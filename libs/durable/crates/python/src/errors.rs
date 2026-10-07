//! Kernel errors as Python exceptions.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_core, DurableError, PyException, "Base class of durable kernel errors.");
create_exception!(_core, InvalidWrite, DurableError, "A write broke a record contract; nothing was committed.");
create_exception!(_core, ReadAfterWrite, DurableError, "A table was read after a table write in the same transaction.");
create_exception!(_core, TransactionFinished, DurableError, "The transaction was already committed or rolled back.");
create_exception!(_core, SessionClosed, DurableError, "The session is closed.");
create_exception!(_core, SessionLocked, DurableError, "Another session holds the file open.");
create_exception!(_core, CorruptStorage, DurableError, "Stored state contradicts itself; the session must not continue.");
create_exception!(_core, InvocationEnded, DurableError, "The invocation may no longer commit: its task finished, moved, or was marked for abort.");
create_exception!(_core, FramesLagged, DurableError, "A frame subscriber fell behind and must resynchronize.");

pub fn to_py(error: durable_core::Error) -> PyErr {
    use durable_core::Error;
    let message = error.to_string();
    match error {
        Error::Invalid(_) => InvalidWrite::new_err(message),
        Error::ReadAfterWrite => ReadAfterWrite::new_err(message),
        Error::Finished => TransactionFinished::new_err(message),
        Error::Closed => SessionClosed::new_err(message),
        Error::Locked(_) => SessionLocked::new_err(message),
        Error::InvocationEnded(_) => InvocationEnded::new_err(message),
        Error::Corrupt(_) | Error::Storage(_) | Error::Encoding(_) | Error::Io(_) => CorruptStorage::new_err(message),
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("DurableError", py.get_type::<DurableError>())?;
    module.add("InvalidWrite", py.get_type::<InvalidWrite>())?;
    module.add("ReadAfterWrite", py.get_type::<ReadAfterWrite>())?;
    module.add("TransactionFinished", py.get_type::<TransactionFinished>())?;
    module.add("SessionClosed", py.get_type::<SessionClosed>())?;
    module.add("SessionLocked", py.get_type::<SessionLocked>())?;
    module.add("CorruptStorage", py.get_type::<CorruptStorage>())?;
    module.add("InvocationEnded", py.get_type::<InvocationEnded>())?;
    module.add("FramesLagged", py.get_type::<FramesLagged>())?;
    Ok(())
}
