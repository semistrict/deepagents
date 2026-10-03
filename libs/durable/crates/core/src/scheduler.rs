//! Runs durable tasks: reserves pending work and drives each task's phases.
//!
//! A host registers one [`TaskHandler`] per task kind. The scheduler reserves
//! eligible tasks of registered kinds, calls the handler once per checkpoint
//! phase, and keeps going while each phase commits progress through
//! [`Invocation::step`]. It stops a run invocation when its task gets an abort
//! mark, releases the task, and later runs the handler's abort path once the
//! work the task owns has drained. A phase that errors, or returns without
//! committing progress, faults its task.
//!
//! Cancellation is cooperative: [`Invocation::cancelled`] resolves when the
//! scheduler wants the invocation to stop, and the scheduler waits for the
//! handler to return. Handlers backed by another runtime (such as Python's
//! asyncio) forward the signal to their own cancellation.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde_json::Value;
use tokio::sync::{Notify, broadcast, watch};
use tokio::task::JoinHandle;

use crate::batch::Frame;
use crate::error::{Error, Result};
use crate::records::{Id, JoinPolicy, Outcome, OutcomeError, Task, TaskState};
use crate::session::{Session, Tx};
use crate::store::Mode;

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// Why a handler stopped without finishing its phase normally.
#[derive(Debug)]
pub enum HandlerError {
    /// The task may no longer commit through this invocation; not a fault.
    Ended,
    /// The handler honored a cancellation; not a fault.
    Cancelled,
    /// Anything else. The task faults with this message.
    Failed(String),
}

impl From<Error> for HandlerError {
    fn from(error: Error) -> Self {
        match error {
            Error::InvocationEnded(_) => HandlerError::Ended,
            other => HandlerError::Failed(other.to_string()),
        }
    }
}

/// The code behind one task kind.
pub trait TaskHandler: Send + Sync + 'static {
    /// Run the phase named by the invocation's checkpoint. It must commit progress
    /// through [`Invocation::step`] before returning.
    fn run(&self, invocation: Invocation) -> BoxFuture<std::result::Result<(), HandlerError>>;

    /// Decide an abort-marked task's outcome. Runs once the task's owned work drained.
    fn abort(&self, invocation: Invocation) -> BoxFuture<std::result::Result<(), HandlerError>>;

    /// Settle what a faulting task was responsible for, inside the commit that faults it.
    fn fault(&self, _tx: Arc<Tx>, _task: Task, _message: String) -> BoxFuture<std::result::Result<(), HandlerError>> {
        Box::pin(async { Ok(()) })
    }
}

/// One reserved run or abort of a task, and the only way its code commits.
#[derive(Clone)]
pub struct Invocation {
    session: Session,
    task: Arc<Mutex<Task>>,
    mode: Mode,
    cancel: watch::Receiver<bool>,
}

impl Invocation {
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// The task record as of this invocation's last commit.
    pub fn task(&self) -> Task {
        self.task.lock().expect("task lock poisoned").clone()
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Resolves once the scheduler asks this invocation to stop.
    pub async fn cancelled(&self) {
        let mut cancel = self.cancel.clone();
        // An error means the scheduler is gone, which also means stop.
        let _ = cancel.wait_for(|cancelled| *cancelled).await;
    }

    pub fn is_cancelled(&self) -> bool {
        *self.cancel.borrow()
    }

    /// Open a gated transaction: its commit fails, writing nothing, unless the task is
    /// still running under this invocation (and, in run mode, carries no abort mark).
    pub fn step(&self) -> Result<Step> {
        let task = self.task();
        let tx = self.session.begin();
        tx.require_running(task.id, self.mode == Mode::Run)?;
        Ok(Step { tx: Arc::new(tx), task, state: None, owner: self.task.clone() })
    }
}

/// A gated transaction plus the task-state change it commits with its writes.
pub struct Step {
    tx: Arc<Tx>,
    task: Task,
    state: Option<TaskState>,
    owner: Arc<Mutex<Task>>,
}

impl Step {
    pub fn tx(&self) -> &Arc<Tx> {
        &self.tx
    }

    /// The task as this invocation last saw it committed.
    pub fn task(&self) -> &Task {
        &self.task
    }

    /// Keep running, continuing at `checkpoint`.
    pub fn advance(&mut self, checkpoint: Value) {
        self.state = Some(TaskState::Running { checkpoint });
    }

    /// Park until every task in `on` is terminal, then resume at `checkpoint`.
    pub fn wait(&mut self, on: Vec<Id>, checkpoint: Value, policy: JoinPolicy) {
        self.state = Some(TaskState::Waiting { checkpoint, on, policy });
    }

    pub fn finish(&mut self, result: Value) {
        self.state = Some(TaskState::Terminal { outcome: Outcome::Completed { result } });
    }

    pub fn fail(&mut self, message: String, detail: Option<Value>) {
        let error = OutcomeError { message, detail };
        self.state = Some(TaskState::Terminal { outcome: Outcome::Failed { error, result: None } });
    }

    /// End the task from its abort handler.
    pub fn aborted(&mut self, reason: Option<String>) {
        self.state = Some(TaskState::Terminal { outcome: Outcome::Aborted { reason, result: None } });
    }

    /// Commit the writes and the new state atomically.
    pub async fn commit(self) -> Result<Arc<Frame>> {
        let id = self.task.id;
        if let Some(state) = self.state {
            self.tx.set_task_state(id, state)?;
        }
        let frame = self.tx.commit().await?;
        if let Some(task) = frame.tasks.iter().find(|task| task.id == id) {
            *self.owner.lock().expect("task lock poisoned") = task.clone();
        }
        Ok(frame)
    }

    pub fn rollback(&self) {
        self.tx.rollback();
    }
}

struct Running {
    mode: Mode,
    cancel: watch::Sender<bool>,
    join: JoinHandle<()>,
}

/// Reserves and runs tasks of registered kinds on the current tokio runtime.
pub struct Scheduler {
    session: Session,
    handlers: RwLock<HashMap<String, Arc<dyn TaskHandler>>>,
    running: Mutex<HashMap<Id, Running>>,
    wake: Notify,
    closing: AtomicBool,
    loops: Mutex<Vec<JoinHandle<()>>>,
}

impl Scheduler {
    pub fn new(session: Session) -> Arc<Scheduler> {
        Arc::new(Scheduler {
            session,
            handlers: RwLock::new(HashMap::new()),
            running: Mutex::new(HashMap::new()),
            wake: Notify::new(),
            closing: AtomicBool::new(false),
            loops: Mutex::new(Vec::new()),
        })
    }

    /// Run tasks of `kind` with `handler`; a later registration replaces an earlier one.
    /// The first registration starts the scheduler on the current tokio runtime.
    pub fn register(self: &Arc<Self>, kind: impl Into<String>, handler: Arc<dyn TaskHandler>) {
        self.handlers.write().expect("handlers lock poisoned").insert(kind.into(), handler);
        let mut loops = self.loops.lock().expect("loops lock poisoned");
        if loops.is_empty() {
            loops.push(tokio::spawn(self.clone().reserve_loop()));
            loops.push(tokio::spawn(self.clone().watch_loop(self.session.subscribe())));
        }
        self.wake.notify_one();
    }

    /// Stop every invocation and loop without writing anything; interrupted work resumes on reopen.
    pub async fn stop(&self) {
        self.closing.store(true, Ordering::SeqCst);
        for handle in self.loops.lock().expect("loops lock poisoned").drain(..) {
            handle.abort();
        }
        let running: Vec<Running> = self.running.lock().expect("running lock poisoned").drain().map(|(_, running)| running).collect();
        for invocation in &running {
            let _ = invocation.cancel.send(true);
        }
        for invocation in running {
            let _ = invocation.join.await;
        }
    }

    fn kinds(&self) -> Vec<String> {
        self.handlers.read().expect("handlers lock poisoned").keys().cloned().collect()
    }

    fn handler(&self, kind: &str) -> Option<Arc<dyn TaskHandler>> {
        self.handlers.read().expect("handlers lock poisoned").get(kind).cloned()
    }

    async fn reserve_loop(self: Arc<Self>) {
        loop {
            self.wake.notified().await;
            loop {
                match self.session.reserve(self.kinds()).await {
                    Ok(Some((task, mode))) => self.spawn(task, mode),
                    Ok(None) => break,
                    Err(_) => return,
                }
            }
        }
    }

    fn spawn(self: &Arc<Self>, task: Task, mode: Mode) {
        let (cancel, cancelled) = watch::channel(false);
        let id = task.id;
        let invocation = Invocation { session: self.session.clone(), task: Arc::new(Mutex::new(task)), mode, cancel: cancelled };
        // Hold the map lock across the spawn so the invocation cannot finish before it is listed.
        let mut running = self.running.lock().expect("running lock poisoned");
        let join = tokio::spawn(self.clone().invoke(invocation));
        running.insert(id, Running { mode, cancel, join });
    }

    /// Wake on every commit, and stop run invocations whose task was marked for abort.
    async fn watch_loop(self: Arc<Self>, mut frames: broadcast::Receiver<Arc<Frame>>) {
        loop {
            match frames.recv().await {
                Ok(frame) => {
                    let running = self.running.lock().expect("running lock poisoned");
                    for task in frame.tasks.iter().filter(|task| task.abort_requested) {
                        if let Some(invocation) = running.get(&task.id).filter(|invocation| invocation.mode == Mode::Run) {
                            let _ = invocation.cancel.send(true);
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    frames = self.session.subscribe();
                    self.cancel_marked().await;
                }
                Err(broadcast::error::RecvError::Closed) => return,
            }
            self.wake.notify_one();
        }
    }

    /// After missing frames, find marked run invocations by reading their tasks.
    async fn cancel_marked(&self) {
        let ids: Vec<Id> = self.running.lock().expect("running lock poisoned").keys().copied().collect();
        for id in ids {
            if let Ok(Some(task)) = self.session.task(id).await
                && task.abort_requested
                && let Some(invocation) = self.running.lock().expect("running lock poisoned").get(&id)
                && invocation.mode == Mode::Run
            {
                let _ = invocation.cancel.send(true);
            }
        }
    }

    async fn invoke(self: Arc<Self>, invocation: Invocation) {
        let task = invocation.task();
        let outcome = match self.handler(&task.kind) {
            None => Err(HandlerError::Failed(format!("no handler for task kind {}", task.kind))),
            Some(handler) => match invocation.mode {
                Mode::Abort => self.abort(&handler, &invocation).await,
                Mode::Run => self.run_phases(&handler, &invocation).await,
            },
        };
        let result = match outcome {
            Ok(()) | Err(HandlerError::Ended) => Ok(()),
            Err(HandlerError::Cancelled) if self.closing.load(Ordering::SeqCst) => Ok(()),
            Err(HandlerError::Cancelled) => self.release(task.id).await,
            Err(HandlerError::Failed(message)) => self.fault(&task, message).await,
        };
        // Storage failing here means the session closed or broke; the task stays as committed.
        let _ = result;
        self.running.lock().expect("running lock poisoned").remove(&task.id);
        self.wake.notify_one();
    }

    async fn run_phases(&self, handler: &Arc<dyn TaskHandler>, invocation: &Invocation) -> std::result::Result<(), HandlerError> {
        loop {
            let before = invocation.task();
            handler.run(invocation.clone()).await?;
            if invocation.is_cancelled() {
                return Err(HandlerError::Cancelled);
            }
            let current = self.session.task(before.id).await?.ok_or_else(|| HandlerError::Failed(format!("task {} disappeared", before.id)))?;
            if !matches!(current.state, TaskState::Running { .. }) {
                return Ok(());
            }
            if current.state == before.state {
                return Err(HandlerError::Failed(format!("task {} returned from a phase without committing progress", before.id)));
            }
            if current.abort_requested {
                return Err(HandlerError::Cancelled);
            }
            *invocation.task.lock().expect("task lock poisoned") = current;
        }
    }

    async fn abort(&self, handler: &Arc<dyn TaskHandler>, invocation: &Invocation) -> std::result::Result<(), HandlerError> {
        let id = invocation.task().id;
        handler.abort(invocation.clone()).await?;
        match self.session.task(id).await? {
            Some(task) if matches!(task.state, TaskState::Running { .. }) => {
                Err(HandlerError::Failed(format!("task {id} returned from its abort handler without an outcome")))
            }
            _ => Ok(()),
        }
    }

    /// Return a still-running task to pending so its abort handler can be reserved.
    async fn release(&self, id: Id) -> std::result::Result<(), HandlerError> {
        let tx = self.session.begin();
        if !matches!(tx.task(id).await?, Some(Task { state: TaskState::Running { .. }, .. })) {
            return Ok(());
        }
        tx.release_task(id)?;
        tx.commit().await?;
        Ok(())
    }

    async fn fault(&self, task: &Task, message: String) -> std::result::Result<(), HandlerError> {
        let tx = Arc::new(self.session.begin());
        let current = match tx.task(task.id).await? {
            Some(current) if matches!(current.state, TaskState::Pending { .. } | TaskState::Running { .. }) => current,
            _ => return Ok(()),
        };
        if let Some(handler) = self.handler(&task.kind) {
            handler.fault(tx.clone(), current, message.clone()).await?;
        }
        let outcome = Outcome::Faulted { error: OutcomeError { message, detail: None } };
        tx.set_task_state(task.id, TaskState::Terminal { outcome })?;
        tx.commit().await?;
        Ok(())
    }
}
