//! Sessions and transactions.
//!
//! Storage runs inline on the host's event loop, so reads and commits are
//! synchronous. Only what can actually wait returns a promise: opening and
//! closing, taking the mutation line for a transaction, waiting for a task or
//! submission to settle, and the next committed frame.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use durable_core::{
    ConversationOwner, DocAddress, DocOptions, Fork, ForkPolicy, Frame, Head, History, Id, Scope, StoredDoc, StoredEntry, Submission,
    SubmissionStatus, TaskFilter, TaskState,
};
use js_sys::{Promise, Reflect};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{Mutex, broadcast};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use crate::convert::{decode, object, parse, to_js, value};
use crate::errors::{named, names, to_js as error, type_error};
use crate::files::{self, SessionFiles};

#[wasm_bindgen(typescript_custom_section)]
const RECORDS: &str = r#"
/** An ID from the session's one namespace. IDs are never reused. */
export type Id = number;
export interface Fork { conversationId: Id; at: Id }
export interface ConversationOwner { conversationId: Id; taskId: Id }
export interface Conversation { id: Id; parent?: Fork; owner?: ConversationOwner }
/** An immutable transcript record; `model`, `data`, `edits`, and other payload fields sit beside the typed ones. */
export interface Entry { id: Id; conversationId: Id; kind: string; head?: Id; byTaskId?: Id; [field: string]: unknown }
export interface StoredEntry { entry: Entry; commitSeq: number }
export type JoinPolicy = "failFast" | "allSettled";
export interface OutcomeError { message: string; detail?: unknown }
export type Outcome =
  | { status: "completed"; result: any }
  | { status: "failed"; error: OutcomeError; result?: any }
  | { status: "aborted"; reason?: string; result?: any }
  | { status: "orphaned"; reason: string }
  | { status: "faulted"; error: OutcomeError };
export type TaskState =
  | { status: "pending"; checkpoint: any }
  | { status: "running"; checkpoint: any }
  | { status: "waiting"; checkpoint: any; on: Id[]; policy: JoinPolicy }
  | { status: "completing"; outcome: Outcome }
  | { status: "terminal"; outcome: Outcome };
export interface Task<Input = any> {
  id: Id;
  conversationId: Id;
  kind: string;
  version: number;
  input: Input;
  owner?: Id;
  background: boolean;
  abortRequested: boolean;
  state: TaskState;
  memos?: Record<string, unknown>;
}
export type SubmissionStatus = "queued" | "placed" | "done" | "unanswered";
/** Something handed to a conversation that callers can wait for; `type` and other fields sit beside the typed ones. */
export interface Submission { id: Id; conversationId: Id; requestId?: string; status: SubmissionStatus; [field: string]: any }
export type Scope = { kind: "session" } | { kind: "conversation"; conversationId: Id } | { kind: "task"; taskId: Id };
export type History = "latest" | "rewindable";
export type ForkPolicy = "current" | "initial" | "asOf";
export interface DocRecord {
  id: Id;
  kind: string;
  key?: string;
  scope: Scope;
  history?: History;
  fork?: ForkPolicy;
  createdAt: number;
  retiredAt?: number;
}
export interface StoredDoc<T = any> { record: DocRecord; version: number; value: T }
export type DocChange = { record: DocRecord } & (
  | { type: "created"; value: any }
  | { type: "updated"; ops: unknown[] }
  | { type: "retired" }
);
/** What one commit wrote, published to observers in commit order. */
export interface Frame {
  seq: number;
  conversations: Conversation[];
  entries: Entry[];
  tasks: Task[];
  submissions: Submission[];
  docs: DocChange[];
}
export interface TaskFilter {
  conversation?: Id;
  kind?: string;
  status?: TaskState["status"];
  abortRequested?: boolean;
  background?: boolean;
  /** Only tasks that are not terminal. */
  live?: boolean;
}
export interface OpenOptions {
  /** The session's SQLite file; without one the session lives in memory. */
  path?: string;
  /** Where the file and its companions live. Required with `path`. */
  files?: SessionFiles;
}
export interface AppendOptions { head?: Id | "self"; byTask?: Id }
export interface CreateTaskOptions { version?: number; owner?: Id; background?: boolean }
export interface CreateSubmissionOptions { requestId?: string; status?: SubmissionStatus }
export interface PutDocOptions { key?: string; version?: number; history?: History; fork?: ForkPolicy }
"#;

/// A session's hold on its files, by the name SQLite opened them as; released once when it closes.
pub(crate) struct Claim {
    name: RefCell<Option<String>>,
}

impl Claim {
    fn release(&self) {
        if let Some(name) = self.name.borrow_mut().take() {
            files::release(&name);
        }
    }
}

fn id(raw: f64) -> Id {
    raw as Id
}

fn optional(raw: Option<f64>) -> Option<i64> {
    raw.map(|raw| raw as i64)
}

fn stored_entry(stored: &StoredEntry) -> Result<Value, JsValue> {
    let entry = serde_json::to_value(&stored.entry).map_err(|error| type_error(&error.to_string()))?;
    Ok(json!({"entry": entry, "commitSeq": stored.seq}))
}

fn stored_entries(entries: &[StoredEntry]) -> Result<JsValue, JsValue> {
    to_js(&entries.iter().map(stored_entry).collect::<Result<Vec<_>, _>>()?)
}

fn stored_doc(doc: Option<StoredDoc>) -> Result<JsValue, JsValue> {
    match doc {
        None => Ok(JsValue::NULL),
        Some(doc) => to_js(&json!({"record": doc.record, "version": doc.version, "value": doc.value})),
    }
}

fn address(kind: String, scope: &JsValue, key: Option<String>) -> Result<DocAddress, JsValue> {
    Ok(DocAddress { kind, scope: decode(scope)?, key })
}

fn status(name: Option<String>) -> Result<Option<SubmissionStatus>, JsValue> {
    name.map(|name| serde_json::from_value(Value::String(name.clone())).map_err(|_| type_error(&format!("unknown status {name}")))).transpose()
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Filter {
    conversation: Option<Id>,
    kind: Option<String>,
    status: Option<String>,
    abort_requested: Option<bool>,
    background: Option<bool>,
    #[serde(default)]
    live: bool,
}

fn filter(raw: &JsValue) -> Result<TaskFilter, JsValue> {
    let Filter { conversation, kind, status, abort_requested, background, live } = options(raw)?;
    Ok(TaskFilter { conversation, kind, status, abort_requested, background, live })
}

pub(crate) fn frame(frame: &Frame) -> Result<JsValue, JsValue> {
    to_js(frame)
}

/// An open durable session.
#[wasm_bindgen]
pub struct Session {
    pub(crate) session: durable_core::Session,
    pub(crate) claim: Rc<Claim>,
}

#[wasm_bindgen]
impl Session {
    /// Open a session over a SQLite file kept by `files`, or in memory without a path.
    pub async fn open(#[wasm_bindgen(unchecked_optional_param_type = "OpenOptions")] options: Option<js_sys::Object>) -> Result<Session, JsValue> {
        let (path, files) = match &options {
            None => (None, None),
            Some(options) => {
                let path = Reflect::get(options, &"path".into())?.as_string();
                let files = Reflect::get(options, &"files".into())?;
                (path, (!files.is_undefined() && !files.is_null()).then(|| files.unchecked_into::<SessionFiles>()))
            }
        };
        let name = match (path, files) {
            (Some(path), Some(files)) => Some(files::claim(&path, files)?),
            (Some(_), None) => return Err(type_error("a session with a path needs `files` to keep it")),
            (None, Some(_)) => return Err(type_error("`files` need a `path` to keep")),
            (None, None) => None,
        };
        let claim = Rc::new(Claim { name: RefCell::new(name.clone()) });
        match durable_core::Session::open(name.map(PathBuf::from)).await {
            Ok(session) => Ok(Session { session, claim }),
            Err(failure) => {
                claim.release();
                Err(error(failure))
            }
        }
    }

    /// Close once the current transaction settles; frame streams end and the files are released.
    #[wasm_bindgen(unchecked_return_type = "Promise<void>")]
    pub fn close(&self) -> Promise {
        let (session, claim) = (self.session.clone(), self.claim.clone());
        future_to_promise(async move {
            session.close().await;
            claim.release();
            Ok(JsValue::UNDEFINED)
        })
    }

    /// Start a transaction holding the mutation line. It waits only while another
    /// transaction holds the line; everything in it is then synchronous.
    #[wasm_bindgen(unchecked_return_type = "Promise<Tx>")]
    pub fn transaction(&self) -> Promise {
        let tx = self.session.begin();
        future_to_promise(async move {
            tx.hold().await.map_err(error)?;
            Ok(Tx { tx: Arc::new(tx) }.into())
        })
    }

    /// Observe every commit from now on, in order.
    pub fn subscribe(&self) -> Frames {
        Frames { frames: Rc::new(Mutex::new(self.session.subscribe())) }
    }

    /// The last committed sequence.
    pub fn seq(&self) -> Result<f64, JsValue> {
        self.session.read_now(|store| Ok(store.seq() as f64)).map_err(error)
    }

    #[wasm_bindgen(unchecked_return_type = "Conversation | null")]
    pub fn conversation(&self, id: f64) -> Result<JsValue, JsValue> {
        to_js(&self.session.read_now(|store| store.conversation(self::id(id))).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "Conversation[]")]
    pub fn conversations(&self, after: Option<f64>, limit: Option<f64>) -> Result<JsValue, JsValue> {
        let limit = limit.map_or(1000, |limit| limit as usize);
        to_js(&self.session.read_now(|store| store.conversations(optional(after), limit)).map_err(error)?)
    }

    #[wasm_bindgen(js_name = ownedConversations, unchecked_return_type = "Conversation[]")]
    pub fn owned_conversations(&self, task: f64) -> Result<JsValue, JsValue> {
        to_js(&self.session.read_now(|store| store.owned_conversations(id(task))).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "StoredEntry | null")]
    pub fn entry(&self, id: f64) -> Result<JsValue, JsValue> {
        match self.session.read_now(|store| store.entry(self::id(id))).map_err(error)? {
            None => Ok(JsValue::NULL),
            Some(stored) => to_js(&stored_entry(&stored)?),
        }
    }

    /// Entries visible from a conversation after `after`, oldest first.
    #[wasm_bindgen(unchecked_return_type = "StoredEntry[]")]
    pub fn entries(&self, conversation: f64, after: Option<f64>, limit: Option<f64>) -> Result<JsValue, JsValue> {
        let limit = limit.map_or(usize::MAX, |limit| limit as usize);
        stored_entries(&self.session.read_now(|store| store.entries(id(conversation), optional(after), limit)).map_err(error)?)
    }

    /// The active transcript, now or as of commit `at`.
    #[wasm_bindgen(unchecked_return_type = "StoredEntry[]")]
    pub fn context(&self, conversation: f64, at: Option<f64>) -> Result<JsValue, JsValue> {
        parse(&self.session.read_now(|store| store.context_json(id(conversation), optional(at))).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "Task | null")]
    pub fn task(&self, id: f64) -> Result<JsValue, JsValue> {
        to_js(&self.session.read_now(|store| store.task(self::id(id))).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "Task[]")]
    pub fn tasks(&self, #[wasm_bindgen(unchecked_optional_param_type = "TaskFilter")] filter: JsValue) -> Result<JsValue, JsValue> {
        let filter = self::filter(&filter)?;
        to_js(&self.session.read_now(|store| store.tasks(&filter)).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "Submission | null")]
    pub fn submission(&self, id: f64) -> Result<JsValue, JsValue> {
        to_js(&self.session.read_now(|store| store.submission(self::id(id))).map_err(error)?)
    }

    #[wasm_bindgen(js_name = submissionByRequest, unchecked_return_type = "Submission | null")]
    pub fn submission_by_request(&self, conversation: f64, request_id: String) -> Result<JsValue, JsValue> {
        to_js(&self.session.read_now(|store| store.submission_by_request(id(conversation), &request_id)).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "Submission[]")]
    pub fn submissions(
        &self,
        conversation: f64,
        #[wasm_bindgen(unchecked_optional_param_type = "SubmissionStatus")] status: Option<String>,
    ) -> Result<JsValue, JsValue> {
        let status = self::status(status)?;
        to_js(&self.session.read_now(|store| store.submissions(id(conversation), status)).map_err(error)?)
    }

    /// A document now or as of commit `at`.
    #[wasm_bindgen(unchecked_return_type = "StoredDoc | null")]
    pub fn doc(
        &self,
        kind: String,
        #[wasm_bindgen(unchecked_param_type = "Scope")] scope: JsValue,
        key: Option<String>,
        at: Option<f64>,
    ) -> Result<JsValue, JsValue> {
        let address = address(kind, &scope, key)?;
        stored_doc(self.session.read_now(|store| store.doc(&address, optional(at))).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "DocRecord[]")]
    pub fn docs(&self, #[wasm_bindgen(unchecked_param_type = "Scope")] scope: JsValue, kind: Option<String>) -> Result<JsValue, JsValue> {
        let scope: Scope = decode(&scope)?;
        to_js(&self.session.read_now(|store| store.docs(scope, kind.as_deref())).map_err(error)?)
    }

    /// Wait until a task is terminal and return its final record.
    #[wasm_bindgen(js_name = waitTask, unchecked_return_type = "Promise<Task>")]
    pub fn wait_task(&self, id: f64) -> Promise {
        let session = self.session.clone();
        future_to_promise(async move { to_js(&session.wait_task(self::id(id)).await.map_err(error)?) })
    }

    /// Wait until a submission is done or unanswered.
    #[wasm_bindgen(js_name = waitSubmission, unchecked_return_type = "Promise<Submission>")]
    pub fn wait_submission(&self, id: f64) -> Promise {
        let session = self.session.clone();
        future_to_promise(async move { to_js(&session.wait_submission(self::id(id)).await.map_err(error)?) })
    }

    /// Every record and document as canonical JSON, in pi's interop shape.
    #[wasm_bindgen(unchecked_return_type = "Promise<any>")]
    pub fn dump(&self) -> Promise {
        let session = self.session.clone();
        future_to_promise(async move { to_js(&durable_core::inspect::dump(&session).await.map_err(error)?) })
    }
}

/// Committed frames, in commit order.
#[wasm_bindgen]
pub struct Frames {
    frames: Rc<Mutex<broadcast::Receiver<Arc<Frame>>>>,
}

#[wasm_bindgen]
impl Frames {
    /// The next frame, or `undefined` once the session closed.
    #[wasm_bindgen(unchecked_return_type = "Promise<Frame | undefined>")]
    pub fn next(&self) -> Promise {
        let frames = self.frames.clone();
        future_to_promise(async move {
            match frames.lock().await.recv().await {
                Ok(next) => frame(&next),
                Err(broadcast::error::RecvError::Lagged(missed)) => Err(named(names::FRAMES_LAGGED, &format!("missed {missed} frames"))),
                Err(broadcast::error::RecvError::Closed) => Ok(JsValue::UNDEFINED),
            }
        })
    }
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppendOptions {
    head: Option<HeadOption>,
    by_task: Option<Id>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum HeadOption {
    Entry(Id),
    Named(String),
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateTaskOptions {
    version: Option<i64>,
    owner: Option<Id>,
    #[serde(default)]
    background: bool,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateSubmissionOptions {
    request_id: Option<String>,
    status: Option<SubmissionStatus>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PutDocOptions {
    key: Option<String>,
    version: Option<i64>,
    history: Option<History>,
    fork: Option<ForkPolicy>,
}

/// Decode an optional options object; `undefined` takes every default.
fn options<T: Default + for<'de> Deserialize<'de>>(raw: &JsValue) -> Result<T, JsValue> {
    if raw.is_undefined() || raw.is_null() { Ok(T::default()) } else { decode(raw) }
}

/// One atomic change, holding the mutation line until it commits or rolls back.
/// Table reads see committed state, so they must come before its first table
/// write; document reads see its own writes.
#[wasm_bindgen]
pub struct Tx {
    pub(crate) tx: Arc<durable_core::Tx>,
}

#[wasm_bindgen]
impl Tx {
    #[wasm_bindgen(unchecked_return_type = "Conversation | null")]
    pub fn conversation(&self, id: f64) -> Result<JsValue, JsValue> {
        to_js(&self.tx.read_now(|store| store.conversation(self::id(id))).map_err(error)?)
    }

    #[wasm_bindgen(js_name = ownedConversations, unchecked_return_type = "Conversation[]")]
    pub fn owned_conversations(&self, task: f64) -> Result<JsValue, JsValue> {
        to_js(&self.tx.read_now(|store| store.owned_conversations(id(task))).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "StoredEntry | null")]
    pub fn entry(&self, id: f64) -> Result<JsValue, JsValue> {
        match self.tx.read_now(|store| store.entry(self::id(id))).map_err(error)? {
            None => Ok(JsValue::NULL),
            Some(stored) => to_js(&stored_entry(&stored)?),
        }
    }

    #[wasm_bindgen(unchecked_return_type = "StoredEntry[]")]
    pub fn context(&self, conversation: f64) -> Result<JsValue, JsValue> {
        stored_entries(&self.tx.read_now(|store| store.context(id(conversation), None)).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "Task | null")]
    pub fn task(&self, id: f64) -> Result<JsValue, JsValue> {
        to_js(&self.tx.read_now(|store| store.task(self::id(id))).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "Task[]")]
    pub fn tasks(&self, #[wasm_bindgen(unchecked_optional_param_type = "TaskFilter")] filter: JsValue) -> Result<JsValue, JsValue> {
        let filter = self::filter(&filter)?;
        to_js(&self.tx.read_now(|store| store.tasks(&filter)).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "Submission | null")]
    pub fn submission(&self, id: f64) -> Result<JsValue, JsValue> {
        to_js(&self.tx.read_now(|store| store.submission(self::id(id))).map_err(error)?)
    }

    #[wasm_bindgen(js_name = submissionByRequest, unchecked_return_type = "Submission | null")]
    pub fn submission_by_request(&self, conversation: f64, request_id: String) -> Result<JsValue, JsValue> {
        to_js(&self.tx.read_now(|store| store.submission_by_request(id(conversation), &request_id)).map_err(error)?)
    }

    #[wasm_bindgen(unchecked_return_type = "Submission[]")]
    pub fn submissions(
        &self,
        conversation: f64,
        #[wasm_bindgen(unchecked_optional_param_type = "SubmissionStatus")] status: Option<String>,
    ) -> Result<JsValue, JsValue> {
        let status = self::status(status)?;
        to_js(&self.tx.read_now(|store| store.submissions(id(conversation), status)).map_err(error)?)
    }

    /// A document's value as this transaction would leave it, or `null`.
    #[wasm_bindgen(unchecked_return_type = "any")]
    pub fn doc(&self, kind: String, #[wasm_bindgen(unchecked_param_type = "Scope")] scope: JsValue, key: Option<String>) -> Result<JsValue, JsValue> {
        to_js(&self.tx.doc_now(address(kind, &scope, key)?).map_err(error)?)
    }

    /// Create the root conversation, which has a fixed ID.
    #[wasm_bindgen(js_name = createRoot)]
    pub fn create_root(&self) -> Result<f64, JsValue> {
        self.tx.create_root().map(|id| id as f64).map_err(error)
    }

    #[wasm_bindgen(js_name = createConversation)]
    pub fn create_conversation(
        &self,
        #[wasm_bindgen(unchecked_optional_param_type = "{ parent?: Fork; owner?: ConversationOwner }")] options: JsValue,
    ) -> Result<f64, JsValue> {
        #[derive(Default, Deserialize)]
        struct Options {
            parent: Option<Fork>,
            owner: Option<ConversationOwner>,
        }
        let Options { parent, owner } = self::options(&options)?;
        self.tx.create_conversation(parent, owner).map(|id| id as f64).map_err(error)
    }

    /// Append an entry. `content` holds `model`, `data`, `edits`, or other fields.
    #[wasm_bindgen(js_name = appendEntry)]
    pub fn append_entry(
        &self,
        conversation: f64,
        kind: String,
        #[wasm_bindgen(unchecked_optional_param_type = "Record<string, unknown> | null")] content: JsValue,
        #[wasm_bindgen(unchecked_optional_param_type = "AppendOptions")] options: JsValue,
    ) -> Result<f64, JsValue> {
        let AppendOptions { head, by_task } = self::options(&options)?;
        let head = match head {
            None => None,
            Some(HeadOption::Entry(entry)) => Some(Head::Entry(entry)),
            Some(HeadOption::Named(name)) if name == "self" => Some(Head::SelfEntry),
            Some(HeadOption::Named(name)) => return Err(type_error(&format!("head must be an entry ID or \"self\", not {name}"))),
        };
        self.tx.append_entry(id(conversation), kind, object(&content)?, head, by_task).map(|id| id as f64).map_err(error)
    }

    #[wasm_bindgen(js_name = createTask)]
    pub fn create_task(
        &self,
        conversation: f64,
        kind: String,
        input: JsValue,
        checkpoint: JsValue,
        #[wasm_bindgen(unchecked_optional_param_type = "CreateTaskOptions")] options: JsValue,
    ) -> Result<f64, JsValue> {
        let CreateTaskOptions { version, owner, background } = self::options(&options)?;
        self.tx
            .create_task(id(conversation), kind, version.unwrap_or(1), value(&input)?, value(&checkpoint)?, owner, background)
            .map(|id| id as f64)
            .map_err(error)
    }

    /// A task replaces its own state: running, waiting, or terminal.
    #[wasm_bindgen(js_name = setTaskState)]
    pub fn set_task_state(&self, id: f64, #[wasm_bindgen(unchecked_param_type = "TaskState")] state: JsValue) -> Result<(), JsValue> {
        let state: TaskState = decode(&state)?;
        self.tx.set_task_state(self::id(id), state).map_err(error)
    }

    #[wasm_bindgen(js_name = releaseTask)]
    pub fn release_task(&self, id: f64) -> Result<(), JsValue> {
        self.tx.release_task(self::id(id)).map_err(error)
    }

    #[wasm_bindgen(js_name = abortTask)]
    pub fn abort_task(&self, id: f64) -> Result<(), JsValue> {
        self.tx.abort_task(self::id(id)).map_err(error)
    }

    /// Admit a submission. `content` holds `type` and any other fields.
    #[wasm_bindgen(js_name = createSubmission)]
    pub fn create_submission(
        &self,
        conversation: f64,
        #[wasm_bindgen(unchecked_param_type = "Record<string, unknown>")] content: JsValue,
        #[wasm_bindgen(unchecked_optional_param_type = "CreateSubmissionOptions")] options: JsValue,
    ) -> Result<f64, JsValue> {
        let CreateSubmissionOptions { request_id, status } = self::options(&options)?;
        let status = status.unwrap_or(SubmissionStatus::Queued);
        self.tx.create_submission(id(conversation), request_id, status, object(&content)?).map(|id| id as f64).map_err(error)
    }

    /// Replace a submission record.
    #[wasm_bindgen(js_name = putSubmission)]
    pub fn put_submission(&self, #[wasm_bindgen(unchecked_param_type = "Submission")] record: JsValue) -> Result<(), JsValue> {
        let submission: Submission = decode(&record)?;
        self.tx.put_submission(submission).map_err(error)
    }

    /// Set a document. `version`, `history`, and `fork` apply when this creates an incarnation.
    #[wasm_bindgen(js_name = putDoc)]
    pub fn put_doc(
        &self,
        kind: String,
        #[wasm_bindgen(unchecked_param_type = "Scope")] scope: JsValue,
        #[wasm_bindgen(unchecked_param_type = "Record<string, unknown>")] value: JsValue,
        #[wasm_bindgen(unchecked_optional_param_type = "PutDocOptions")] options: JsValue,
    ) -> Result<(), JsValue> {
        let PutDocOptions { key, version, history, fork } = self::options(&options)?;
        let options = DocOptions { version: version.unwrap_or(1), history, fork };
        self.tx.put_doc(address(kind, &scope, key)?, options, self::value(&value)?).map_err(error)
    }

    #[wasm_bindgen(js_name = retireDoc)]
    pub fn retire_doc(
        &self,
        kind: String,
        #[wasm_bindgen(unchecked_param_type = "Scope")] scope: JsValue,
        key: Option<String>,
    ) -> Result<(), JsValue> {
        self.tx.retire_doc(address(kind, &scope, key)?).map_err(error)
    }

    /// Commit atomically and return the published frame.
    #[wasm_bindgen(unchecked_return_type = "Frame")]
    pub fn commit(&self) -> Result<JsValue, JsValue> {
        frame(&*self.tx.commit_now().map_err(error)?)
    }

    /// Abandon the transaction without writing, releasing the line.
    pub fn rollback(&self) {
        self.tx.rollback();
    }
}
