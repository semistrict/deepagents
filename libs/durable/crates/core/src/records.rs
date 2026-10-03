//! Durable records, encoded exactly as pi-durable stores them.
//!
//! Each record keeps the fields the kernel interprets typed, and carries any
//! other field through untouched, so files written by either implementation
//! round-trip.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// An ID from the one session-global namespace. IDs are never reused.
pub type Id = i64;

/// A commit sequence number. Strictly increases between commits.
pub type Seq = i64;

/// The conversation `root()` creates on first use.
pub const ROOT_CONVERSATION: Id = 1;

/// A history parent: the child sees the parent's entries up to and including `at`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Fork {
    pub conversation_id: Id,
    pub at: Id,
}

/// The task that owns a conversation, and that task's conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationOwner {
    pub conversation_id: Id,
    pub task_id: Id,
}

/// A transcript scope. Immutable once created.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Conversation {
    pub id: Id,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<Fork>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<ConversationOwner>,
}

/// Where a new entry points the start of the active transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    /// The entry itself starts the new context.
    SelfEntry,
    /// An earlier entry starts the new context.
    Entry(Id),
}

/// An immutable transcript record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Entry {
    pub id: Id,
    pub conversation_id: Id,
    pub kind: String,
    /// First entry of the active context this entry selects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<Id>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by_task_id: Option<Id>,
    /// `model`, `data`, `edits`, and any other payload.
    #[serde(flatten)]
    pub content: Map<String, Value>,
}

/// An entry with the commit that stored it.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredEntry {
    pub entry: Entry,
    pub seq: Seq,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JoinPolicy {
    /// The first non-completed outcome aborts the rest of `on`.
    FailFast,
    AllSettled,
}

/// A JSON-safe error snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutcomeError {
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
}

/// How a task ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum Outcome {
    Completed {
        result: Value,
    },
    /// An expected failure the task committed itself.
    Failed {
        error: OutcomeError,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
    },
    /// Cancellation decided by the task's abort handler.
    Aborted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
    },
    /// Aborted while no definition could run its abort handler.
    Orphaned {
        reason: String,
    },
    /// The runtime ended the task because its code broke a contract.
    Faulted {
        error: OutcomeError,
    },
}

impl Outcome {
    pub fn completed(&self) -> bool {
        matches!(self, Outcome::Completed { .. })
    }
}

/// A task's durable state machine position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum TaskState {
    Pending {
        checkpoint: Value,
    },
    Running {
        checkpoint: Value,
    },
    /// Parked until every task in `on` is terminal, then resumes at `checkpoint`.
    Waiting {
        checkpoint: Value,
        on: Vec<Id>,
        policy: JoinPolicy,
    },
    /// Outcome decided; terminal once no ordinary owned work is live.
    Completing {
        outcome: Outcome,
    },
    Terminal {
        outcome: Outcome,
    },
}

impl TaskState {
    pub fn status(&self) -> &'static str {
        match self {
            TaskState::Pending { .. } => "pending",
            TaskState::Running { .. } => "running",
            TaskState::Waiting { .. } => "waiting",
            TaskState::Completing { .. } => "completing",
            TaskState::Terminal { .. } => "terminal",
        }
    }

    pub fn live(&self) -> bool {
        !matches!(self, TaskState::Terminal { .. })
    }

    /// Pending, running, or waiting: the task can still run code.
    pub fn open(&self) -> bool {
        matches!(self, TaskState::Pending { .. } | TaskState::Running { .. } | TaskState::Waiting { .. })
    }

    pub fn outcome(&self) -> Option<&Outcome> {
        match self {
            TaskState::Completing { outcome } | TaskState::Terminal { outcome } => Some(outcome),
            _ => None,
        }
    }
}

/// A durable state machine attached to one conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Task {
    pub id: Id,
    pub conversation_id: Id,
    pub kind: String,
    /// Definition version, for migrating live input and checkpoints.
    pub version: i64,
    pub input: Value,
    /// Owning task; absent for a task its conversation owns. Immutable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<Id>,
    /// Excluded from conversation abort, idle, and cascades.
    pub background: bool,
    pub abort_requested: bool,
    pub state: TaskState,
    /// Small first-writer-wins values kept while the task can run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memos: Option<Map<String, Value>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SubmissionStatus {
    Queued,
    Placed,
    Done,
    Unanswered,
}

impl SubmissionStatus {
    pub fn settled(self) -> bool {
        matches!(self, SubmissionStatus::Done | SubmissionStatus::Unanswered)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SubmissionStatus::Queued => "queued",
            SubmissionStatus::Placed => "placed",
            SubmissionStatus::Done => "done",
            SubmissionStatus::Unanswered => "unanswered",
        }
    }
}

/// Something handed to a conversation that callers can wait for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Submission {
    pub id: Id,
    pub conversation_id: Id,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub status: SubmissionStatus,
    /// `type`, `entry`, `answer`, `reason`, `detail`, and any other field.
    #[serde(flatten)]
    pub content: Map<String, Value>,
}

/// What a document belongs to; it retires with that owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Scope {
    Session,
    Conversation {
        #[serde(rename = "conversationId")]
        conversation_id: Id,
    },
    Task {
        #[serde(rename = "taskId")]
        task_id: Id,
    },
}

impl Scope {
    pub(crate) fn columns(self) -> (&'static str, Id) {
        match self {
            Scope::Session => ("session", 0),
            Scope::Conversation { conversation_id } => ("conversation", conversation_id),
            Scope::Task { task_id } => ("task", task_id),
        }
    }
}

/// The logical address of one document: a singleton, or a member of a keyed family.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocAddress {
    pub kind: String,
    pub scope: Scope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

/// Whether old values of a conversation document stay readable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum History {
    Latest,
    Rewindable,
}

/// What a fork of the conversation starts with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForkPolicy {
    #[serde(rename = "current")]
    Current,
    #[serde(rename = "initial")]
    Initial,
    #[serde(rename = "asOf")]
    AsOf,
}

/// One incarnation of a document. Recreating an address makes a new incarnation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocRecord {
    pub id: Id,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    pub scope: Scope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history: Option<History>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork: Option<ForkPolicy>,
    pub created_at: Seq,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retired_at: Option<Seq>,
}

impl DocRecord {
    pub fn address(&self) -> DocAddress {
        DocAddress { kind: self.kind.clone(), scope: self.scope, key: self.key.clone() }
    }

    /// Only rewindable conversation documents keep their history.
    pub fn current_only(&self) -> bool {
        !matches!(self.scope, Scope::Conversation { .. }) || self.history != Some(History::Rewindable)
    }
}

/// Lifetime options of a document, fixed when an incarnation is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DocOptions {
    /// Definition version stored with every revision.
    pub version: i64,
    /// Conversation documents only.
    pub history: Option<History>,
    /// Conversation documents only.
    pub fork: Option<ForkPolicy>,
}

impl Default for DocOptions {
    fn default() -> Self {
        DocOptions { version: 1, history: None, fork: None }
    }
}
