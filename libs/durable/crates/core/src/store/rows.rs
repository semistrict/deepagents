//! Record columns. Every table keeps its record as JSON; a few fields are
//! mirrored into indexed columns.

use rusqlite::Row;
use rusqlite::types::Type;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::Result;
use crate::records::{Id, Seq, StoredEntry};

pub(super) fn json<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}

/// Indexed strings are stored JSON-encoded, as pi does, so lone surrogates survive.
pub(super) fn indexed(value: &str) -> String {
    serde_json::to_string(value).expect("strings encode")
}

pub(super) fn decode<T: DeserializeOwned>(row: &Row, index: usize) -> rusqlite::Result<T> {
    let raw: String = row.get(index)?;
    serde_json::from_str(&raw).map_err(|error| rusqlite::Error::FromSqlConversionFailure(index, Type::Text, Box::new(error)))
}

/// Decode a `SELECT record ...` row.
pub(super) fn record<T: DeserializeOwned>(row: &Row) -> rusqlite::Result<T> {
    decode(row, 0)
}

/// Decode a `SELECT record, commit_seq FROM entries` row.
pub(super) fn entry(row: &Row) -> rusqlite::Result<StoredEntry> {
    Ok(StoredEntry { entry: decode(row, 0)?, seq: row.get(1)? })
}

/// An entry's stored record, undecoded.
pub(super) struct RawEntry {
    pub record: String,
    pub seq: Seq,
    pub head: Option<Id>,
}

/// Decode a `SELECT record, commit_seq, head FROM entries` row without parsing the record.
pub(super) fn raw_entry(row: &Row) -> rusqlite::Result<RawEntry> {
    Ok(RawEntry { record: row.get(0)?, seq: row.get(1)?, head: row.get(2)? })
}
