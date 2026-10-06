//! Python bindings for the durable kernel. Every I/O method returns an
//! awaitable that runs on the kernel's tokio runtime; records cross as the
//! JSON dicts pi-durable stores.

mod convert;
mod errors;
mod scheduler;

use std::path::PathBuf;
use std::sync::Arc;

use durable_core::{
    ConversationOwner, DocAddress, DocOptions, Fork, ForkPolicy, Frame, Head, History, Id, Mode, Scope, Seq, StoredDoc, StoredEntry, Submission,
    SubmissionStatus, TaskFilter, TaskState,
};
use pyo3::exceptions::{PyStopAsyncIteration, PyTypeError};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use serde_json::{Value, json};
use tokio::sync::{Mutex, broadcast};

use convert::{Json, decode, object};
use errors::to_py;

pub(crate) type Awaitable<'py> = PyResult<Bound<'py, PyAny>>;

fn stored_entry(stored: &StoredEntry) -> PyResult<Value> {
    Ok(json!({"entry": Json::of(&stored.entry)?.0, "commitSeq": stored.seq}))
}

fn stored_entries(entries: &[StoredEntry]) -> PyResult<Json> {
    entries.iter().map(stored_entry).collect::<PyResult<Vec<_>>>().map(|items| Json(Value::Array(items)))
}

fn stored_doc(doc: Option<StoredDoc>) -> PyResult<Json> {
    match doc {
        None => Ok(Json(Value::Null)),
        Some(doc) => Ok(Json(json!({"record": Json::of(&doc.record)?.0, "version": doc.version, "value": doc.value}))),
    }
}

pub(crate) fn frame(frame: &Frame) -> PyResult<Json> {
    Json::of(frame)
}

fn address(kind: String, scope: &Bound<'_, PyAny>, key: Option<String>) -> PyResult<DocAddress> {
    Ok(DocAddress { kind, scope: decode(scope)?, key })
}

fn status(name: Option<&str>) -> PyResult<Option<SubmissionStatus>> {
    name.map(|name| serde_json::from_value(Value::String(name.to_owned())).map_err(|_| PyTypeError::new_err(format!("unknown status {name}"))))
        .transpose()
}

fn filter(
    conversation: Option<Id>,
    kind: Option<String>,
    status: Option<String>,
    abort_requested: Option<bool>,
    background: Option<bool>,
    live: bool,
) -> TaskFilter {
    TaskFilter { conversation, kind, status, abort_requested, background, live }
}

/// An open durable session.
#[pyclass(frozen, name = "Session")]
pub(crate) struct PySession {
    pub(crate) session: durable_core::Session,
}

#[pymethods]
impl PySession {
    /// Open a session over a SQLite file, or in memory without a path.
    #[staticmethod]
    #[pyo3(signature = (path=None))]
    fn open(py: Python<'_>, path: Option<PathBuf>) -> Awaitable<'_> {
        future_into_py(py, async move {
            let session = durable_core::Session::open(path).await.map_err(to_py)?;
            Ok(PySession { session })
        })
    }

    /// Close once the current transaction settles; frame streams end.
    fn close<'py>(&self, py: Python<'py>) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move {
            session.close().await;
            Ok(())
        })
    }

    /// Start a transaction; it takes the mutation line at its first read or at commit.
    fn begin(&self) -> PyTx {
        PyTx { tx: Arc::new(self.session.begin()) }
    }

    /// Observe every commit from now on, in order.
    fn subscribe(&self) -> Frames {
        Frames { frames: Arc::new(Mutex::new(self.session.subscribe())) }
    }

    /// Reserve the oldest eligible pending task of one of `kinds`: `(task, "run" | "abort")` or `None`.
    fn reserve<'py>(&self, py: Python<'py>, kinds: Vec<String>) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move {
            let reserved = session.reserve(kinds).await.map_err(to_py)?;
            Ok(match reserved {
                None => Json(Value::Null),
                Some((task, mode)) => {
                    let mode = match mode {
                        Mode::Run => "run",
                        Mode::Abort => "abort",
                    };
                    Json(json!([Json::of(&task)?.0, mode]))
                }
            })
        })
    }

    fn seq<'py>(&self, py: Python<'py>) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { session.seq().await.map_err(to_py) })
    }

    fn conversation<'py>(&self, py: Python<'py>, id: Id) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { Json::of(&session.conversation(id).await.map_err(to_py)?) })
    }

    #[pyo3(signature = (after=None, limit=1000))]
    fn conversations<'py>(&self, py: Python<'py>, after: Option<Id>, limit: usize) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { Json::of(&session.conversations(after, limit).await.map_err(to_py)?) })
    }

    fn owned_conversations<'py>(&self, py: Python<'py>, task: Id) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { Json::of(&session.owned_conversations(task).await.map_err(to_py)?) })
    }

    fn entry<'py>(&self, py: Python<'py>, id: Id) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move {
            match session.entry(id).await.map_err(to_py)? {
                None => Ok(Json(Value::Null)),
                Some(stored) => stored_entry(&stored).map(Json),
            }
        })
    }

    /// Entries visible from a conversation after `after`, oldest first.
    #[pyo3(signature = (conversation, after=None, limit=usize::MAX))]
    fn entries<'py>(&self, py: Python<'py>, conversation: Id, after: Option<Id>, limit: usize) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { stored_entries(&session.entries(conversation, after, limit).await.map_err(to_py)?) })
    }

    /// The active transcript, now or as of commit `at`.
    #[pyo3(signature = (conversation, at=None))]
    fn context<'py>(&self, py: Python<'py>, conversation: Id, at: Option<Seq>) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { Ok(convert::JsonText(session.context_json(conversation, at).await.map_err(to_py)?)) })
    }

    fn task<'py>(&self, py: Python<'py>, id: Id) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { Json::of(&session.task(id).await.map_err(to_py)?) })
    }

    #[pyo3(signature = (*, conversation=None, kind=None, status=None, abort_requested=None, background=None, live=false))]
    #[allow(clippy::too_many_arguments)]
    fn tasks<'py>(
        &self,
        py: Python<'py>,
        conversation: Option<Id>,
        kind: Option<String>,
        status: Option<String>,
        abort_requested: Option<bool>,
        background: Option<bool>,
        live: bool,
    ) -> Awaitable<'py> {
        let session = self.session.clone();
        let filter = filter(conversation, kind, status, abort_requested, background, live);
        future_into_py(py, async move { Json::of(&session.tasks(filter).await.map_err(to_py)?) })
    }

    fn submission<'py>(&self, py: Python<'py>, id: Id) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { Json::of(&session.submission(id).await.map_err(to_py)?) })
    }

    fn submission_by_request<'py>(&self, py: Python<'py>, conversation: Id, request_id: String) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { Json::of(&session.submission_by_request(conversation, request_id).await.map_err(to_py)?) })
    }

    #[pyo3(signature = (conversation, status=None))]
    fn submissions<'py>(&self, py: Python<'py>, conversation: Id, status: Option<&str>) -> Awaitable<'py> {
        let session = self.session.clone();
        let status = self::status(status)?;
        future_into_py(py, async move { Json::of(&session.submissions(conversation, status).await.map_err(to_py)?) })
    }

    /// `{"record", "version", "value"}` of a document now or as of commit `at`, or `None`.
    #[pyo3(signature = (kind, scope, key=None, at=None))]
    fn doc<'py>(&self, py: Python<'py>, kind: String, scope: &Bound<'py, PyAny>, key: Option<String>, at: Option<Seq>) -> Awaitable<'py> {
        let session = self.session.clone();
        let address = address(kind, scope, key)?;
        future_into_py(py, async move { stored_doc(session.doc(address, at).await.map_err(to_py)?) })
    }

    #[pyo3(signature = (scope, kind=None))]
    fn docs<'py>(&self, py: Python<'py>, scope: &Bound<'py, PyAny>, kind: Option<String>) -> Awaitable<'py> {
        let session = self.session.clone();
        let scope: Scope = decode(scope)?;
        future_into_py(py, async move { Json::of(&session.docs(scope, kind).await.map_err(to_py)?) })
    }

    fn wait_task<'py>(&self, py: Python<'py>, id: Id) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { Json::of(&session.wait_task(id).await.map_err(to_py)?) })
    }

    fn wait_submission<'py>(&self, py: Python<'py>, id: Id) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { Json::of(&session.wait_submission(id).await.map_err(to_py)?) })
    }

    /// Every record and document as canonical JSON, in pi's interop shape.
    fn dump<'py>(&self, py: Python<'py>) -> Awaitable<'py> {
        let session = self.session.clone();
        future_into_py(py, async move { Ok(Json(durable_core::inspect::dump(&session).await.map_err(to_py)?)) })
    }
}

/// Committed frames, in commit order.
#[pyclass(frozen)]
struct Frames {
    frames: Arc<Mutex<broadcast::Receiver<Arc<Frame>>>>,
}

#[pymethods]
impl Frames {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> Awaitable<'py> {
        let frames = self.frames.clone();
        future_into_py(py, async move {
            match frames.lock().await.recv().await {
                Ok(next) => frame(&next),
                Err(broadcast::error::RecvError::Lagged(missed)) => Err(errors::FramesLagged::new_err(format!("missed {missed} frames"))),
                Err(broadcast::error::RecvError::Closed) => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

/// One atomic change. Table reads must precede table writes.
#[pyclass(frozen, name = "Tx")]
pub(crate) struct PyTx {
    pub(crate) tx: Arc<durable_core::Tx>,
}

#[pymethods]
impl PyTx {
    fn conversation<'py>(&self, py: Python<'py>, id: Id) -> Awaitable<'py> {
        let tx = self.tx.clone();
        future_into_py(py, async move { Json::of(&tx.conversation(id).await.map_err(to_py)?) })
    }

    fn owned_conversations<'py>(&self, py: Python<'py>, task: Id) -> Awaitable<'py> {
        let tx = self.tx.clone();
        future_into_py(py, async move { Json::of(&tx.owned_conversations(task).await.map_err(to_py)?) })
    }

    fn entry<'py>(&self, py: Python<'py>, id: Id) -> Awaitable<'py> {
        let tx = self.tx.clone();
        future_into_py(py, async move {
            match tx.entry(id).await.map_err(to_py)? {
                None => Ok(Json(Value::Null)),
                Some(stored) => stored_entry(&stored).map(Json),
            }
        })
    }

    fn context<'py>(&self, py: Python<'py>, conversation: Id) -> Awaitable<'py> {
        let tx = self.tx.clone();
        future_into_py(py, async move { stored_entries(&tx.context(conversation).await.map_err(to_py)?) })
    }

    fn task<'py>(&self, py: Python<'py>, id: Id) -> Awaitable<'py> {
        let tx = self.tx.clone();
        future_into_py(py, async move { Json::of(&tx.task(id).await.map_err(to_py)?) })
    }

    #[pyo3(signature = (*, conversation=None, kind=None, status=None, abort_requested=None, background=None, live=false))]
    #[allow(clippy::too_many_arguments)]
    fn tasks<'py>(
        &self,
        py: Python<'py>,
        conversation: Option<Id>,
        kind: Option<String>,
        status: Option<String>,
        abort_requested: Option<bool>,
        background: Option<bool>,
        live: bool,
    ) -> Awaitable<'py> {
        let tx = self.tx.clone();
        let filter = filter(conversation, kind, status, abort_requested, background, live);
        future_into_py(py, async move { Json::of(&tx.tasks(filter).await.map_err(to_py)?) })
    }

    fn submission<'py>(&self, py: Python<'py>, id: Id) -> Awaitable<'py> {
        let tx = self.tx.clone();
        future_into_py(py, async move { Json::of(&tx.submission(id).await.map_err(to_py)?) })
    }

    fn submission_by_request<'py>(&self, py: Python<'py>, conversation: Id, request_id: String) -> Awaitable<'py> {
        let tx = self.tx.clone();
        future_into_py(py, async move { Json::of(&tx.submission_by_request(conversation, request_id).await.map_err(to_py)?) })
    }

    #[pyo3(signature = (conversation, status=None))]
    fn submissions<'py>(&self, py: Python<'py>, conversation: Id, status: Option<&str>) -> Awaitable<'py> {
        let tx = self.tx.clone();
        let status = self::status(status)?;
        future_into_py(py, async move { Json::of(&tx.submissions(conversation, status).await.map_err(to_py)?) })
    }

    /// A document's value as this transaction would leave it, or `None`.
    #[pyo3(signature = (kind, scope, key=None))]
    fn doc<'py>(&self, py: Python<'py>, kind: String, scope: &Bound<'py, PyAny>, key: Option<String>) -> Awaitable<'py> {
        let tx = self.tx.clone();
        let address = address(kind, scope, key)?;
        future_into_py(py, async move { Ok(Json(tx.doc(address).await.map_err(to_py)?.unwrap_or(Value::Null))) })
    }

    fn create_root(&self) -> PyResult<Id> {
        self.tx.create_root().map_err(to_py)
    }

    /// `parent` is `{"conversationId", "at"}`; `owner` is `{"conversationId", "taskId"}`.
    #[pyo3(signature = (*, parent=None, owner=None))]
    fn create_conversation(&self, parent: Option<&Bound<'_, PyAny>>, owner: Option<&Bound<'_, PyAny>>) -> PyResult<Id> {
        let parent: Option<Fork> = parent.map(decode).transpose()?;
        let owner: Option<ConversationOwner> = owner.map(decode).transpose()?;
        self.tx.create_conversation(parent, owner).map_err(to_py)
    }

    /// `content` holds `model`, `data`, or other fields. `head` is an entry ID or `"self"`.
    #[pyo3(signature = (conversation, kind, content=None, *, head=None, by_task=None))]
    fn append_entry(
        &self,
        conversation: Id,
        kind: String,
        content: Option<&Bound<'_, PyAny>>,
        head: Option<&Bound<'_, PyAny>>,
        by_task: Option<Id>,
    ) -> PyResult<Id> {
        let head = match head {
            None => None,
            Some(head) if head.extract::<&str>().is_ok_and(|text| text == "self") => Some(Head::SelfEntry),
            Some(head) => Some(Head::Entry(head.extract()?)),
        };
        self.tx.append_entry(conversation, kind, object(content)?, head, by_task).map_err(to_py)
    }

    #[pyo3(signature = (conversation, kind, input, checkpoint, *, version=1, owner=None, background=false))]
    #[allow(clippy::too_many_arguments)]
    fn create_task(
        &self,
        conversation: Id,
        kind: String,
        input: &Bound<'_, PyAny>,
        checkpoint: &Bound<'_, PyAny>,
        version: i64,
        owner: Option<Id>,
        background: bool,
    ) -> PyResult<Id> {
        let (input, checkpoint) = (convert::from_object(input)?, convert::from_object(checkpoint)?);
        self.tx.create_task(conversation, kind, version, input, checkpoint, owner, background).map_err(to_py)
    }

    /// A task replaces its own state: `{"status": "running" | "waiting" | "terminal", ...}`.
    fn set_task_state(&self, id: Id, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let state: TaskState = decode(state)?;
        self.tx.set_task_state(id, state).map_err(to_py)
    }

    fn release_task(&self, id: Id) -> PyResult<()> {
        self.tx.release_task(id).map_err(to_py)
    }

    fn abort_task(&self, id: Id) -> PyResult<()> {
        self.tx.abort_task(id).map_err(to_py)
    }

    #[pyo3(signature = (conversation, content, *, request_id=None, status="queued"))]
    fn create_submission(&self, conversation: Id, content: &Bound<'_, PyAny>, request_id: Option<String>, status: &str) -> PyResult<Id> {
        let status = self::status(Some(status))?.expect("a status was given");
        self.tx.create_submission(conversation, request_id, status, object(Some(content))?).map_err(to_py)
    }

    /// Replace a submission record read earlier in this transaction.
    fn put_submission(&self, record: &Bound<'_, PyAny>) -> PyResult<()> {
        let submission: Submission = decode(record)?;
        self.tx.put_submission(submission).map_err(to_py)
    }

    /// Set a document. `version`, `history`, and `fork` apply when this creates an incarnation.
    #[pyo3(signature = (kind, scope, value, *, key=None, version=1, history=None, fork=None))]
    #[allow(clippy::too_many_arguments)]
    fn put_doc(
        &self,
        kind: String,
        scope: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
        key: Option<String>,
        version: i64,
        history: Option<&Bound<'_, PyAny>>,
        fork: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let history: Option<History> = history.map(decode).transpose()?;
        let fork: Option<ForkPolicy> = fork.map(decode).transpose()?;
        let options = DocOptions { version, history, fork };
        self.tx.put_doc(address(kind, scope, key)?, options, convert::from_object(value)?).map_err(to_py)
    }

    #[pyo3(signature = (kind, scope, key=None))]
    fn retire_doc(&self, kind: String, scope: &Bound<'_, PyAny>, key: Option<String>) -> PyResult<()> {
        self.tx.retire_doc(address(kind, scope, key)?).map_err(to_py)
    }

    /// Commit atomically and return the published frame.
    fn commit<'py>(&self, py: Python<'py>) -> Awaitable<'py> {
        let tx = self.tx.clone();
        future_into_py(py, async move { frame(&*tx.commit().await.map_err(to_py)?) })
    }

    fn rollback(&self) {
        self.tx.rollback();
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySession>()?;
    module.add_class::<PyTx>()?;
    module.add_class::<Frames>()?;
    scheduler::register(module)?;
    errors::register(module)
}
