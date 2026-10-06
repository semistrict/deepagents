//! The kernel's scheduler driving Python task handlers.
//!
//! A handler is any Python object with async `run(invocation)` and
//! `abort(invocation)` methods, and optionally async `fault(tx, task, message)`.
//! Each call runs as a task on the event loop that registered the handler;
//! when the scheduler cancels an invocation, that task is cancelled and the
//! scheduler waits for it to unwind.

use std::sync::{Arc, Mutex};

use durable_core::scheduler::BoxFuture;
use durable_core::{HandlerError, Invocation, JoinPolicy, Mode, Step, Task, TaskHandler, Tx};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use pyo3_async_runtimes::tokio::future_into_py;
use tokio::sync::oneshot;

use crate::convert::{Json, from_object};
use crate::errors::{InvocationEnded, to_py};
use crate::{Awaitable, PySession, PyTx, frame};

type Outcome = Result<(), HandlerError>;

/// Delivers a finished `concurrent.futures.Future` back to the scheduler.
#[pyclass(frozen)]
struct Done {
    sender: Mutex<Option<oneshot::Sender<Outcome>>>,
}

#[pymethods]
impl Done {
    fn __call__(&self, future: &Bound<'_, PyAny>) -> PyResult<()> {
        let outcome = if future.call_method0("cancelled")?.is_truthy()? {
            Err(HandlerError::Cancelled)
        } else {
            let error = future.call_method0("exception")?;
            if error.is_none() {
                Ok(())
            } else if error.is_instance_of::<InvocationEnded>() {
                Err(HandlerError::Ended)
            } else if error.is_instance(&future.py().import("asyncio")?.getattr("CancelledError")?)? {
                Err(HandlerError::Cancelled)
            } else {
                Err(HandlerError::Failed(error.repr()?.to_string()))
            }
        };
        if let Some(sender) = self.sender.lock().expect("sender lock poisoned").take() {
            // The scheduler stopped waiting only if it was dropped; nothing to tell.
            let _ = sender.send(outcome);
        }
        Ok(())
    }
}

/// A Python handler registered from an event loop.
pub struct PyHandler {
    handler: Py<PyAny>,
    locals: TaskLocals,
}

impl PyHandler {
    pub fn new(py: Python<'_>, handler: Py<PyAny>) -> PyResult<PyHandler> {
        Ok(PyHandler { handler, locals: pyo3_async_runtimes::tokio::get_current_locals(py)? })
    }

    /// Schedule `method(*args)` on the handler's loop; returns its future and the result channel.
    fn schedule(
        &self,
        method: &str,
        args: impl for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>>,
    ) -> Result<(Py<PyAny>, oneshot::Receiver<Outcome>), HandlerError> {
        Python::attach(|py| {
            let coroutine = self.handler.bind(py).call_method1(method, args(py)?)?;
            let asyncio = py.import("asyncio")?;
            let future = asyncio.call_method1("run_coroutine_threadsafe", (coroutine, self.locals.event_loop(py)))?;
            let (sender, receiver) = oneshot::channel();
            future.call_method1("add_done_callback", (Done { sender: Mutex::new(Some(sender)) },))?;
            Ok((future.unbind(), receiver))
        })
        .map_err(|error: PyErr| HandlerError::Failed(error.to_string()))
    }

    fn call(&self, method: &'static str, invocation: Invocation) -> BoxFuture<Outcome> {
        let scheduled = self.schedule(method, |py| {
            let invocation = Py::new(py, PyInvocation { invocation: invocation.clone() })?;
            pyo3::types::PyTuple::new(py, [invocation])
        });
        Box::pin(async move {
            let (future, mut receiver) = scheduled?;
            tokio::select! {
                outcome = &mut receiver => return outcome.unwrap_or(Err(HandlerError::Cancelled)),
                () = invocation.cancelled() => {}
            }
            // Cancel the Python task and wait for it to unwind before the scheduler moves on.
            Python::attach(|py| future.bind(py).call_method0("cancel").map(drop)).map_err(|error| HandlerError::Failed(error.to_string()))?;
            receiver.await.unwrap_or(Err(HandlerError::Cancelled))
        })
    }
}

impl TaskHandler for PyHandler {
    fn run(&self, invocation: Invocation) -> BoxFuture<Outcome> {
        self.call("run", invocation)
    }

    fn abort(&self, invocation: Invocation) -> BoxFuture<Outcome> {
        self.call("abort", invocation)
    }

    fn fault(&self, tx: Arc<Tx>, task: Task, message: String) -> BoxFuture<Outcome> {
        let has_fault = Python::attach(|py| self.handler.bind(py).hasattr("fault").unwrap_or(false));
        if !has_fault {
            return Box::pin(async { Ok(()) });
        }
        let scheduled = self.schedule("fault", |py| {
            let tx = Py::new(py, PyTx { tx: tx.clone() })?;
            let task = Json::of(&task)?.into_pyobject(py)?;
            pyo3::types::PyTuple::new(py, [tx.into_any().into_bound(py), task, message.into_pyobject(py)?.into_any()])
        });
        Box::pin(async move {
            let (_, receiver) = scheduled?;
            receiver.await.unwrap_or(Err(HandlerError::Cancelled))
        })
    }
}

/// Reserves and runs tasks of registered kinds.
#[pyclass(frozen, name = "Scheduler")]
pub struct PyScheduler {
    scheduler: Arc<durable_core::Scheduler>,
}

#[pymethods]
impl PyScheduler {
    #[new]
    fn new(session: &PySession) -> PyScheduler {
        PyScheduler { scheduler: durable_core::Scheduler::new(session.session.clone()) }
    }

    /// Run tasks of `kind` with `handler`. Call from the event loop the handler runs on.
    fn register(&self, py: Python<'_>, kind: String, handler: Py<PyAny>) -> PyResult<()> {
        let handler = Arc::new(PyHandler::new(py, handler)?);
        let scheduler = self.scheduler.clone();
        // Registration spawns the scheduler's loops, which needs the kernel runtime.
        pyo3_async_runtimes::tokio::get_runtime().block_on(async move { scheduler.register(kind, handler) });
        Ok(())
    }

    /// Stop every invocation without writing anything.
    fn stop<'py>(&self, py: Python<'py>) -> Awaitable<'py> {
        let scheduler = self.scheduler.clone();
        future_into_py(py, async move {
            scheduler.stop().await;
            Ok(())
        })
    }
}

/// One reserved run or abort of a task.
#[pyclass(frozen, name = "Invocation")]
pub struct PyInvocation {
    invocation: Invocation,
}

#[pymethods]
impl PyInvocation {
    /// The task record as of this invocation's last commit.
    #[getter]
    fn task(&self) -> PyResult<Json> {
        Json::of(&self.invocation.task())
    }

    #[getter]
    fn mode(&self) -> &'static str {
        match self.invocation.mode() {
            Mode::Run => "run",
            Mode::Abort => "abort",
        }
    }

    #[getter]
    fn session(&self) -> PySession {
        PySession { session: self.invocation.session().clone() }
    }

    /// Open a gated transaction for the task's next state; its commit fails unless
    /// the task is still running under this invocation.
    fn step(&self) -> PyResult<PyStep> {
        let step = self.invocation.step().map_err(to_py)?;
        Ok(PyStep { tx: step.tx().clone(), step: Mutex::new(Some(step)) })
    }
}

/// A gated transaction plus the task-state change it commits.
#[pyclass(frozen, name = "Step")]
pub struct PyStep {
    tx: Arc<Tx>,
    step: Mutex<Option<Step>>,
}

impl PyStep {
    fn with<R>(&self, f: impl FnOnce(&mut Step) -> R) -> PyResult<R> {
        let mut step = self.step.lock().expect("step lock poisoned");
        step.as_mut().map(f).ok_or_else(|| to_py(durable_core::Error::Finished))
    }
}

#[pymethods]
impl PyStep {
    #[getter]
    fn tx(&self) -> PyTx {
        PyTx { tx: self.tx.clone() }
    }

    /// The task as committed when the step began.
    #[getter]
    fn task(&self) -> PyResult<Json> {
        self.with(|step| Json::of(step.task()))?
    }

    fn advance(&self, checkpoint: &Bound<'_, PyAny>) -> PyResult<()> {
        let checkpoint = from_object(checkpoint)?;
        self.with(|step| step.advance(checkpoint))
    }

    #[pyo3(signature = (on, checkpoint, *, policy="allSettled"))]
    fn wait(&self, on: Vec<i64>, checkpoint: &Bound<'_, PyAny>, policy: &str) -> PyResult<()> {
        let checkpoint = from_object(checkpoint)?;
        let policy: JoinPolicy = serde_json::from_value(serde_json::Value::String(policy.to_owned()))
            .map_err(|_| PyTypeError::new_err(format!("unknown join policy {policy}")))?;
        self.with(|step| step.wait(on, checkpoint, policy))
    }

    #[pyo3(signature = (result=None))]
    fn finish(&self, result: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let result = result.map(from_object).transpose()?.unwrap_or(serde_json::Value::Null);
        self.with(|step| step.finish(result))
    }

    #[pyo3(signature = (message, detail=None))]
    fn fail(&self, message: String, detail: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let detail = detail.map(from_object).transpose()?;
        self.with(|step| step.fail(message, detail))
    }

    #[pyo3(signature = (reason=None))]
    fn aborted(&self, reason: Option<String>) -> PyResult<()> {
        self.with(|step| step.aborted(reason))
    }

    /// Commit the writes and the new task state atomically.
    fn commit<'py>(&self, py: Python<'py>) -> Awaitable<'py> {
        let step = self.step.lock().expect("step lock poisoned").take().ok_or_else(|| to_py(durable_core::Error::Finished))?;
        future_into_py(py, async move { frame(&*step.commit().await.map_err(to_py)?) })
    }

    fn rollback(&self) {
        if let Some(step) = self.step.lock().expect("step lock poisoned").take() {
            step.rollback();
        }
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyScheduler>()?;
    module.add_class::<PyInvocation>()?;
    module.add_class::<PyStep>()?;
    Ok(())
}
