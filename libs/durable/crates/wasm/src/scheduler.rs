//! The kernel's scheduler driving JavaScript task handlers.
//!
//! A handler is an object with `run(invocation)` and `abort(invocation)`,
//! which may return promises, and optionally `fault(tx, task, message)`.
//! Cancellation is cooperative: when the scheduler stops an invocation, its
//! `signal` aborts, and the scheduler waits for the handler to settle.

use std::cell::RefCell;
use std::pin::pin;
use std::rc::Rc;
use std::sync::Arc;

use durable_core::{BoxFuture, HandlerError, JoinPolicy, Mode, Task, TaskHandler, Tx as CoreTx};
use futures_util::future::{Either, select};
use js_sys::{Function, Promise, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, future_to_promise};

use crate::convert::{decode, to_js, value};
use crate::errors::{describe, name_of, names, to_js as error, type_error};
use crate::session::{Claim, Session, Tx, frame};

#[wasm_bindgen(typescript_custom_section)]
const HANDLER: &str = r#"
/** The code behind one task kind. */
export interface TaskHandler {
  /** Run the phase named by the invocation's checkpoint; it must commit progress through a step. */
  run(invocation: Invocation): Promise<void> | void;
  /** Decide an abort-marked task's outcome, once the work it owns has drained. */
  abort(invocation: Invocation): Promise<void> | void;
  /** Settle what a faulting task was responsible for, inside the commit that faults it. */
  fault?(tx: Tx, task: Task, message: string): Promise<void> | void;
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "TaskHandler")]
    pub type JsTaskHandler;

    #[wasm_bindgen(method, catch)]
    fn run(this: &JsTaskHandler, invocation: Invocation) -> Result<JsValue, JsValue>;
    #[wasm_bindgen(method, catch)]
    fn abort(this: &JsTaskHandler, invocation: Invocation) -> Result<JsValue, JsValue>;

    type AbortController;
    #[wasm_bindgen(constructor)]
    fn new() -> AbortController;
    #[wasm_bindgen(method, getter)]
    fn signal(this: &AbortController) -> JsValue;
    #[wasm_bindgen(method, js_name = abort)]
    fn cancel(this: &AbortController);
}

type Outcome = Result<(), HandlerError>;

/// How a handler's thrown value ends its invocation.
fn outcome(thrown: &JsValue, cancelled: bool) -> HandlerError {
    match name_of(thrown).as_deref() {
        Some(names::INVOCATION_ENDED) => HandlerError::Ended,
        Some("AbortError") => HandlerError::Cancelled,
        _ if cancelled => HandlerError::Cancelled,
        _ => HandlerError::Failed(describe(thrown)),
    }
}

/// Wait for what a handler returned, aborting its signal if the scheduler cancels first.
async fn settle(returned: JsValue, invocation: durable_core::Invocation, controller: AbortController) -> Outcome {
    let mut settled = JsFuture::from(Promise::resolve(&returned));
    let cancelled = pin!(invocation.cancelled());
    let (result, cancelled) = match select(&mut settled, cancelled).await {
        Either::Left((result, _)) => (result, false),
        Either::Right(((), settled)) => {
            controller.cancel();
            (settled.await, true)
        }
    };
    result.map(drop).map_err(|thrown| outcome(&thrown, cancelled))
}

/// A JavaScript handler, run on the event loop it was registered from.
struct JsHandler {
    handler: JsTaskHandler,
    claim: Rc<Claim>,
}

impl JsHandler {
    fn call(&self, mode: Mode, invocation: durable_core::Invocation) -> BoxFuture<Outcome> {
        let controller = AbortController::new();
        let exposed = Invocation { invocation: invocation.clone(), signal: controller.signal(), claim: self.claim.clone() };
        let returned = match mode {
            Mode::Run => self.handler.run(exposed),
            Mode::Abort => self.handler.abort(exposed),
        };
        Box::pin(async move {
            match returned {
                Ok(returned) => settle(returned, invocation, controller).await,
                Err(thrown) => Err(outcome(&thrown, false)),
            }
        })
    }
}

impl TaskHandler for JsHandler {
    fn run(&self, invocation: durable_core::Invocation) -> BoxFuture<Outcome> {
        self.call(Mode::Run, invocation)
    }

    fn abort(&self, invocation: durable_core::Invocation) -> BoxFuture<Outcome> {
        self.call(Mode::Abort, invocation)
    }

    fn fault(&self, tx: Arc<CoreTx>, task: Task, message: String) -> BoxFuture<Outcome> {
        let fault = Reflect::get(&self.handler, &"fault".into()).ok().and_then(|fault| fault.dyn_into::<Function>().ok());
        let Some(fault) = fault else {
            return Box::pin(async { Ok(()) });
        };
        let returned = to_js(&task).and_then(|task| fault.call3(&self.handler, &Tx { tx }.into(), &task, &JsValue::from_str(&message)));
        Box::pin(async move {
            let returned = returned.map_err(|thrown| HandlerError::Failed(describe(&thrown)))?;
            JsFuture::from(Promise::resolve(&returned)).await.map(drop).map_err(|thrown| HandlerError::Failed(describe(&thrown)))
        })
    }
}

/// Reserves and runs tasks of registered kinds on the host's event loop.
#[wasm_bindgen]
pub struct Scheduler {
    scheduler: Arc<durable_core::Scheduler>,
    claim: Rc<Claim>,
}

#[wasm_bindgen]
impl Scheduler {
    #[wasm_bindgen(constructor)]
    pub fn new(session: &Session) -> Scheduler {
        Scheduler { scheduler: durable_core::Scheduler::new(session.session.clone()), claim: session.claim.clone() }
    }

    /// Run tasks of `kind` with `handler`; a later registration replaces an earlier one.
    pub fn register(&self, kind: String, handler: JsTaskHandler) {
        // In a JavaScript host the handler never leaves the thread it was registered on.
        #[allow(clippy::arc_with_non_send_sync)]
        let handler = Arc::new(JsHandler { handler, claim: self.claim.clone() });
        self.scheduler.register(kind, handler);
    }

    /// Stop every invocation without writing anything; interrupted work resumes on reopen.
    #[wasm_bindgen(unchecked_return_type = "Promise<void>")]
    pub fn stop(&self) -> Promise {
        let scheduler = self.scheduler.clone();
        future_to_promise(async move {
            scheduler.stop().await;
            Ok(JsValue::UNDEFINED)
        })
    }
}

/// One reserved run or abort of a task, and the only way its code commits.
#[wasm_bindgen]
pub struct Invocation {
    invocation: durable_core::Invocation,
    signal: JsValue,
    claim: Rc<Claim>,
}

#[wasm_bindgen]
impl Invocation {
    /// The task record as of this invocation's last commit.
    #[wasm_bindgen(getter, unchecked_return_type = "Task")]
    pub fn task(&self) -> Result<JsValue, JsValue> {
        to_js(&self.invocation.task())
    }

    #[wasm_bindgen(getter, unchecked_return_type = "\"run\" | \"abort\"")]
    pub fn mode(&self) -> String {
        match self.invocation.mode() {
            Mode::Run => "run".into(),
            Mode::Abort => "abort".into(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn session(&self) -> Session {
        Session { session: self.invocation.session().clone(), claim: self.claim.clone() }
    }

    /// Aborts when the scheduler wants this invocation to stop.
    #[wasm_bindgen(getter, unchecked_return_type = "AbortSignal")]
    pub fn signal(&self) -> JsValue {
        self.signal.clone()
    }

    /// Open a step holding the mutation line: a transaction whose commit fails,
    /// writing nothing, unless the task is still running under this invocation.
    #[wasm_bindgen(unchecked_return_type = "Promise<Step>")]
    pub fn step(&self) -> Promise {
        let step = self.invocation.step();
        future_to_promise(async move {
            let step = step.map_err(error)?;
            step.hold().await.map_err(error)?;
            Ok(Step { tx: step.tx().clone(), step: RefCell::new(Some(step)) }.into())
        })
    }
}

/// A gated transaction plus the task-state change it commits.
#[wasm_bindgen]
pub struct Step {
    tx: Arc<CoreTx>,
    step: RefCell<Option<durable_core::Step>>,
}

impl Step {
    fn with<R>(&self, f: impl FnOnce(&mut durable_core::Step) -> R) -> Result<R, JsValue> {
        self.step.borrow_mut().as_mut().map(f).ok_or_else(|| error(durable_core::Error::Finished))
    }
}

#[wasm_bindgen]
impl Step {
    #[wasm_bindgen(getter)]
    pub fn tx(&self) -> Tx {
        Tx { tx: self.tx.clone() }
    }

    /// The task as committed when the step began.
    #[wasm_bindgen(getter, unchecked_return_type = "Task")]
    pub fn task(&self) -> Result<JsValue, JsValue> {
        self.with(|step| to_js(step.task()))?
    }

    /// Keep running, continuing at `checkpoint`.
    pub fn advance(&self, checkpoint: JsValue) -> Result<(), JsValue> {
        let checkpoint = value(&checkpoint)?;
        self.with(|step| step.advance(checkpoint))
    }

    /// Park until every task in `on` is terminal, then resume at `checkpoint`.
    pub fn wait(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Id[]")] on: JsValue,
        checkpoint: JsValue,
        #[wasm_bindgen(unchecked_optional_param_type = "JoinPolicy")] policy: Option<String>,
    ) -> Result<(), JsValue> {
        let on: Vec<i64> = decode(&on)?;
        let checkpoint = value(&checkpoint)?;
        let policy = policy.unwrap_or_else(|| "allSettled".into());
        let policy: JoinPolicy =
            serde_json::from_value(serde_json::Value::String(policy.clone())).map_err(|_| type_error(&format!("unknown join policy {policy}")))?;
        self.with(|step| step.wait(on, checkpoint, policy))
    }

    pub fn finish(&self, #[wasm_bindgen(unchecked_optional_param_type = "unknown")] result: JsValue) -> Result<(), JsValue> {
        let result = value(&result)?;
        self.with(|step| step.finish(result))
    }

    pub fn fail(&self, message: String, #[wasm_bindgen(unchecked_optional_param_type = "unknown")] detail: JsValue) -> Result<(), JsValue> {
        let detail = match value(&detail)? {
            serde_json::Value::Null => None,
            detail => Some(detail),
        };
        self.with(|step| step.fail(message, detail))
    }

    /// End the task from its abort handler.
    pub fn aborted(&self, reason: Option<String>) -> Result<(), JsValue> {
        self.with(|step| step.aborted(reason))
    }

    /// Commit the writes and the new task state atomically.
    #[wasm_bindgen(unchecked_return_type = "Frame")]
    pub fn commit(&self) -> Result<JsValue, JsValue> {
        let step = self.step.borrow_mut().take().ok_or_else(|| error(durable_core::Error::Finished))?;
        frame(&*step.commit_now().map_err(error)?)
    }

    pub fn rollback(&self) {
        if let Some(step) = self.step.borrow_mut().take() {
            step.rollback();
        }
    }
}
