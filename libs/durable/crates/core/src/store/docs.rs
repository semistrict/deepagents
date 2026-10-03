//! Documents: typed JSON objects stored next to the transcript.
//!
//! An incarnation's revisions are a base followed by Chord delta batches.
//! Current-only documents drop older revisions whenever a new base lands;
//! rewindable conversation documents keep them, so any committed value can be
//! read back at its sequence.

use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde_json::Value;

use super::rows::{decode, indexed, json, record};
use crate::batch::{Change, DocChange};
use crate::delta::{self, Op};
use crate::error::{Error, Result, invalid};
use crate::records::{DocAddress, DocOptions, DocRecord, ForkPolicy, History, Id, Scope, Seq};

/// Store a fresh base after this many deltas, bounding replay on reads.
const BASE_EVERY: usize = 32;

/// A materialized incarnation.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredDoc {
    pub record: DocRecord,
    /// The definition version its newest base was written with.
    pub version: i64,
    pub value: Value,
    /// Delta batches replayed after that base.
    pub deltas_since_base: usize,
}

/// The indexed columns of an address: kind, scope kind, owner, family flag, key.
fn columns(address: &DocAddress) -> (String, &'static str, Id, bool, String) {
    let (scope, owner) = address.scope.columns();
    (indexed(&address.kind), scope, owner, address.key.is_some(), indexed(address.key.as_deref().unwrap_or("")))
}

/// The incarnation at an address that is current, or alive at commit `at`.
pub(super) fn find(conn: &Connection, address: &DocAddress, at: Option<Seq>) -> Result<Option<DocRecord>> {
    let (kind, scope, owner, family, key) = columns(address);
    let found = match at {
        None => conn
            .prepare_cached(
                "SELECT record FROM documents
                 WHERE kind = ?1 AND scope_kind = ?2 AND owner_id = ?3 AND family = ?4 AND key_value = ?5
                 AND retired_at IS NULL ORDER BY created_at DESC LIMIT 1",
            )?
            .query_row(params![kind, scope, owner, family, key], record)
            .optional()?,
        Some(seq) => conn
            .prepare_cached(
                "SELECT record FROM documents
                 WHERE kind = ?1 AND scope_kind = ?2 AND owner_id = ?3 AND family = ?4 AND key_value = ?5
                 AND created_at <= ?6 AND (retired_at IS NULL OR retired_at > ?6)
                 ORDER BY created_at DESC LIMIT 1",
            )?
            .query_row(params![kind, scope, owner, family, key, seq], record)
            .optional()?,
    };
    Ok(found)
}

/// Rebuild an incarnation's value from its newest base at or before `at`.
pub(super) fn materialize(conn: &Connection, record: DocRecord, at: Option<Seq>) -> Result<StoredDoc> {
    let id = record.id;
    if at.is_some() && record.current_only() {
        return Err(invalid(format!("document {id} does not retain historical content")));
    }
    let upper = at.unwrap_or(Seq::MAX);
    let (base_seq, version, mut value): (Seq, i64, Value) = conn
        .prepare_cached(
            "SELECT seq, version, content FROM document_revisions
             WHERE document_id = ?1 AND kind = 'base' AND seq <= ?2 ORDER BY seq DESC LIMIT 1",
        )?
        .query_row(params![id, upper], |row| Ok((row.get(0)?, row.get(1)?, decode(row, 2)?)))
        .optional()?
        .ok_or_else(|| Error::Corrupt(format!("document {id} is missing a required base")))?;
    let mut tail = conn.prepare_cached(
        "SELECT kind, version, content FROM document_revisions
         WHERE document_id = ?1 AND seq > ?2 AND seq <= ?3 ORDER BY seq",
    )?;
    let mut deltas_since_base = 0;
    for revision in tail.query_map(params![id, base_seq, upper], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, decode::<Vec<Op>>(row, 2)?))
    })? {
        let (kind, revision_version, ops) = revision?;
        if kind != "delta" || revision_version != version {
            return Err(Error::Corrupt(format!("document {id} crosses a stored version boundary without a base")));
        }
        delta::apply(&mut value, &ops)?;
        deltas_since_base += 1;
    }
    Ok(StoredDoc { record, version, value, deltas_since_base })
}

pub(super) fn read(conn: &Connection, address: &DocAddress, at: Option<Seq>) -> Result<Option<StoredDoc>> {
    find(conn, address, at)?.map(|record| materialize(conn, record, at)).transpose()
}

pub(super) fn list(conn: &Connection, scope: Scope, kind: Option<&str>) -> Result<Vec<DocRecord>> {
    let (scope_kind, owner) = scope.columns();
    let mut stmt = conn.prepare_cached(
        "SELECT record FROM documents
         WHERE scope_kind = ?1 AND owner_id = ?2 AND retired_at IS NULL AND (?3 IS NULL OR kind = ?3) ORDER BY id",
    )?;
    let rows = stmt.query_map(params![scope_kind, owner, kind.map(indexed)], record)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Replace a document's value, creating incarnation `id` when the address has none.
/// Returns `None` when nothing changed. A batch writes each incarnation at most once.
pub(super) fn put(tx: &Transaction, seq: Seq, address: &DocAddress, options: DocOptions, id: Id, value: Value) -> Result<Option<DocChange>> {
    if !value.is_object() {
        return Err(invalid(format!("document {} must be a JSON object", address.kind)));
    }
    let Some(current) = read(tx, address, None)? else { return create(tx, seq, address, options, id, value).map(Some) };
    let ops = delta::diff(&current.value, &value);
    if ops.is_empty() && current.version == options.version {
        return Ok(None);
    }
    let encoded_ops = json(&ops)?;
    let encoded_value = json(&value)?;
    let rebase = current.version != options.version
        || current.deltas_since_base + 1 >= BASE_EVERY
        || encoded_ops.len() >= encoded_value.len();
    if rebase {
        revise(tx, &current.record, seq, "base", options.version, &encoded_value)?;
    } else {
        revise(tx, &current.record, seq, "delta", options.version, &encoded_ops)?;
    }
    Ok(Some(DocChange { record: current.record, change: Change::Updated { ops } }))
}

fn create(tx: &Transaction, seq: Seq, address: &DocAddress, options: DocOptions, id: Id, value: Value) -> Result<DocChange> {
    let (history, fork) = match address.scope {
        Scope::Conversation { .. } => {
            let history = options.history.unwrap_or(History::Latest);
            let fork = options.fork.unwrap_or(ForkPolicy::Current);
            if history == History::Latest && fork == ForkPolicy::AsOf {
                return Err(invalid(format!("document {} cannot fork as-of without history", address.kind)));
            }
            (Some(history), Some(fork))
        }
        Scope::Session | Scope::Task { .. } => (None, None),
    };
    let record = DocRecord {
        id,
        kind: address.kind.clone(),
        key: address.key.clone(),
        scope: address.scope,
        history,
        fork,
        created_at: seq,
        retired_at: None,
    };
    let (kind, scope, owner, family, key) = columns(address);
    super::claim(tx, id, "document")?;
    tx.prepare_cached(
        "INSERT INTO documents (id, kind, family, key_value, scope_kind, owner_id, created_at, retired_at, record)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8)",
    )?
    .execute(params![id, kind, family, key, scope, owner, seq, json(&record)?])?;
    revise(tx, &record, seq, "base", options.version, &json(&value)?)?;
    Ok(DocChange { record, change: Change::Created { value } })
}

fn revise(tx: &Transaction, record: &DocRecord, seq: Seq, kind: &str, version: i64, content: &str) -> Result<()> {
    if kind == "base" && record.current_only() {
        tx.prepare_cached("DELETE FROM document_revisions WHERE document_id = ?1")?.execute([record.id])?;
    }
    let inserted = tx
        .prepare_cached("INSERT OR IGNORE INTO document_revisions (document_id, seq, kind, version, content) VALUES (?1, ?2, ?3, ?4, ?5)")?
        .execute(params![record.id, seq, kind, version, content])?;
    if inserted == 0 {
        return Err(invalid(format!("document {} changed twice in one commit", record.id)));
    }
    Ok(())
}

pub(super) fn retire(tx: &Transaction, seq: Seq, address: &DocAddress) -> Result<Option<DocChange>> {
    let Some(mut record) = find(tx, address, None)? else { return Ok(None) };
    record.retired_at = Some(seq);
    tx.prepare_cached("UPDATE documents SET retired_at = ?2, record = ?3 WHERE id = ?1")?
        .execute(params![record.id, seq, json(&record)?])?;
    if record.current_only() {
        tx.prepare_cached("DELETE FROM document_revisions WHERE document_id = ?1")?.execute([record.id])?;
    }
    Ok(Some(DocChange { record, change: Change::Retired }))
}

/// Retire every current document of a scope, such as a task that became terminal.
pub(super) fn retire_scope(tx: &Transaction, seq: Seq, scope: Scope) -> Result<Vec<DocChange>> {
    let mut changes = Vec::new();
    for record in list(tx, scope, None)? {
        changes.extend(retire(tx, seq, &record.address())?);
    }
    Ok(changes)
}
