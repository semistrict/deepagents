//! Kernel errors as JavaScript errors, told apart by `name`.

use js_sys::Error as JsError;
use wasm_bindgen::prelude::*;

/// Every kernel error's `name`, as the host's error classes match them.
pub mod names {
    pub const INVALID_WRITE: &str = "InvalidWriteError";
    pub const READ_AFTER_WRITE: &str = "ReadAfterWriteError";
    pub const TRANSACTION_FINISHED: &str = "TransactionFinishedError";
    pub const SESSION_CLOSED: &str = "SessionClosedError";
    pub const SESSION_LOCKED: &str = "SessionLockedError";
    pub const CORRUPT_STORAGE: &str = "CorruptStorageError";
    pub const INVOCATION_ENDED: &str = "InvocationEndedError";
    pub const FRAMES_LAGGED: &str = "FramesLaggedError";
}

pub fn named(name: &str, message: &str) -> JsValue {
    let error = JsError::new(message);
    error.set_name(name);
    error.into()
}

pub fn type_error(message: &str) -> JsValue {
    js_sys::TypeError::new(message).into()
}

pub fn to_js(error: durable_core::Error) -> JsValue {
    use durable_core::Error;
    let message = error.to_string();
    let name = match error {
        Error::Invalid(_) => names::INVALID_WRITE,
        Error::ReadAfterWrite => names::READ_AFTER_WRITE,
        Error::Finished => names::TRANSACTION_FINISHED,
        Error::Closed => names::SESSION_CLOSED,
        Error::Locked(_) => names::SESSION_LOCKED,
        Error::InvocationEnded(_) => names::INVOCATION_ENDED,
        Error::Corrupt(_) | Error::Storage(_) | Error::Encoding(_) | Error::Io(_) => names::CORRUPT_STORAGE,
    };
    named(name, &message)
}

/// The `name` of a thrown JavaScript value, if it is an error.
pub fn name_of(thrown: &JsValue) -> Option<String> {
    thrown.dyn_ref::<JsError>().map(|error| String::from(error.name()))
}

/// A thrown JavaScript value as text: `Name: message` for an error, else the value itself.
pub fn describe(thrown: &JsValue) -> String {
    if let Some(error) = thrown.dyn_ref::<JsError>() {
        return format!("{}: {}", String::from(error.name()), String::from(error.message()));
    }
    thrown.as_string().unwrap_or_else(|| format!("{thrown:?}"))
}
