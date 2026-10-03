use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("storage: {0}")]
    Storage(#[from] rusqlite::Error),
    #[error("encoding: {0}")]
    Encoding(#[from] serde_json::Error),
    /// Stored state contradicts itself. The session must not continue.
    #[error("corrupt storage: {0}")]
    Corrupt(String),
    /// A write broke a record contract; nothing was committed.
    #[error("invalid write: {0}")]
    Invalid(String),
    /// A table was read after a table write in the same transaction.
    #[error("table read after a table write in the same transaction")]
    ReadAfterWrite,
    /// The transaction was already committed or abandoned.
    #[error("transaction is finished")]
    Finished,
    #[error("session is closed")]
    Closed,
    /// Another session holds the file open.
    #[error("session file is open elsewhere: {0}")]
    Locked(String),
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    /// An invocation may no longer commit: its task finished, moved, or was marked for abort.
    #[error("invocation ended: {0}")]
    InvocationEnded(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid(msg: impl Into<String>) -> Error {
    Error::Invalid(msg.into())
}
