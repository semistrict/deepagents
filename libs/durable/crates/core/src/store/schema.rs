//! pi-durable's SQLite layout, statement for statement, so either
//! implementation opens the other's files.

use rusqlite::{Connection, OptionalExtension};

use crate::error::{Error, Result};

/// `packages/durable/src/storage/sqlite/migrations.ts`, version 1.
const INITIAL_SCHEMA: &[&str] = &[
    "CREATE TABLE durable_metadata (
		singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
		next_id TEXT NOT NULL,
		next_seq INTEGER NOT NULL
	) STRICT",
    "INSERT INTO durable_metadata (singleton, next_id, next_seq) VALUES (1, '2', 1)",
    "CREATE TABLE record_ids (
		id INTEGER PRIMARY KEY,
		record_type TEXT NOT NULL CHECK (record_type IN ('conversation', 'entry', 'task', 'submission', 'document'))
	) STRICT",
    "CREATE TABLE conversations (
		id INTEGER PRIMARY KEY,
		owner_conversation_id INTEGER,
		owner_task_id INTEGER,
		record TEXT NOT NULL CHECK (json_valid(record))
	) STRICT",
    "CREATE INDEX conversations_by_owner_conversation ON conversations (owner_conversation_id, id)",
    "CREATE INDEX conversations_by_owner_task ON conversations (owner_task_id, id)",
    "CREATE TABLE entries (
		id INTEGER PRIMARY KEY,
		conversation_id INTEGER NOT NULL,
		head INTEGER,
		commit_seq INTEGER NOT NULL,
		record TEXT NOT NULL CHECK (json_valid(record))
	) STRICT",
    "CREATE INDEX entries_by_conversation ON entries (conversation_id, id DESC)",
    "CREATE INDEX entry_heads_by_conversation ON entries (conversation_id, id DESC) WHERE head IS NOT NULL",
    "CREATE TABLE tasks (
		id INTEGER PRIMARY KEY,
		conversation_id INTEGER NOT NULL,
		kind TEXT NOT NULL,
		status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'waiting', 'completing', 'terminal')),
		abort_requested INTEGER NOT NULL CHECK (abort_requested IN (0, 1)),
		background INTEGER NOT NULL CHECK (background IN (0, 1)),
		record TEXT NOT NULL CHECK (json_valid(record))
	) STRICT",
    "CREATE INDEX tasks_by_status ON tasks (status, id)",
    "CREATE INDEX tasks_by_conversation ON tasks (conversation_id, id)",
    "CREATE INDEX tasks_by_kind ON tasks (kind, id)",
    "CREATE INDEX tasks_by_abort_requested ON tasks (abort_requested, id)",
    "CREATE INDEX tasks_by_background ON tasks (background, id)",
    "CREATE TABLE submissions (
		id INTEGER PRIMARY KEY,
		conversation_id INTEGER NOT NULL,
		request_id TEXT,
		status TEXT NOT NULL CHECK (status IN ('queued', 'placed', 'done', 'unanswered')),
		record TEXT NOT NULL CHECK (json_valid(record))
	) STRICT",
    "CREATE INDEX submissions_by_request ON submissions (conversation_id, request_id)",
    "CREATE INDEX submissions_by_conversation ON submissions (conversation_id, id)",
    "CREATE INDEX submissions_by_status ON submissions (status, id)",
    "CREATE TABLE documents (
		id INTEGER PRIMARY KEY,
		kind TEXT NOT NULL,
		family INTEGER NOT NULL CHECK (family IN (0, 1)),
		key_value TEXT NOT NULL,
		scope_kind TEXT NOT NULL CHECK (scope_kind IN ('session', 'conversation', 'task')),
		owner_id INTEGER NOT NULL,
		created_at INTEGER NOT NULL,
		retired_at INTEGER,
		record TEXT NOT NULL CHECK (json_valid(record))
	) STRICT",
    "CREATE INDEX documents_by_address
		ON documents (kind, scope_kind, owner_id, family, key_value, created_at DESC, retired_at)",
    "CREATE INDEX documents_by_scope ON documents (scope_kind, owner_id, id)",
    "CREATE INDEX documents_by_scope_kind ON documents (scope_kind, owner_id, kind, id)",
    "CREATE TABLE document_revisions (
		document_id INTEGER NOT NULL,
		seq INTEGER NOT NULL,
		kind TEXT NOT NULL CHECK (kind IN ('base', 'delta')),
		version INTEGER NOT NULL,
		content TEXT NOT NULL CHECK (json_valid(content)),
		PRIMARY KEY (document_id, seq)
	) STRICT",
    "CREATE INDEX document_revisions_by_kind ON document_revisions (document_id, kind, seq DESC)",
];

/// Immutable, ordered schema history, mirroring pi's `SQLITE_MIGRATIONS`.
const MIGRATIONS: &[&[&str]] = &[INITIAL_SCHEMA];

pub(crate) fn migrate(conn: &mut Connection) -> Result<()> {
    // A JavaScript host's file system has no shared memory for the WAL index.
    // The session is the file's one owner, so it keeps the index in its own
    // memory under an exclusive lock, which SQLite allows without shared memory.
    #[cfg(js)]
    conn.pragma_update(None, "locking_mode", "exclusive")?;
    conn.pragma_update(None, "journal_mode", "wal")?;
    conn.pragma_update(None, "synchronous", "normal")?;
    let tx = conn.transaction()?;
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS durable_schema (
			singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
			version INTEGER NOT NULL CHECK (version >= 0)
		) STRICT;
		INSERT OR IGNORE INTO durable_schema (singleton, version) VALUES (1, 0);",
    )?;
    let version: i64 = tx
        .query_row("SELECT version FROM durable_schema WHERE singleton = 1", [], |row| row.get(0))
        .optional()?
        .ok_or_else(|| Error::Corrupt("schema metadata is missing".into()))?;
    let supported = MIGRATIONS.len() as i64;
    if version > supported {
        return Err(Error::Corrupt(format!("schema version {version} is newer than supported version {supported}")));
    }
    for (index, statements) in MIGRATIONS.iter().enumerate().skip(version as usize) {
        for statement in *statements {
            tx.execute_batch(statement)?;
        }
        tx.execute("UPDATE durable_schema SET version = ?1 WHERE singleton = 1", [index as i64 + 1])?;
    }
    tx.commit()?;
    Ok(())
}
