//! What one commit writes, and what it publishes.

use serde::Serialize;
use serde_json::Value;

use crate::delta::Op;
use crate::records::{Conversation, DocAddress, DocOptions, DocRecord, Entry, Id, Seq, Submission, Task, TaskState};

/// One buffered write. A batch of them commits atomically.
#[derive(Clone, Debug)]
pub enum Write {
    /// Fail the whole batch with `InvocationEnded` unless `task` is running
    /// (and, with `unmarked`, has no abort mark). Writes nothing.
    Require { task: Id, unmarked: bool },
    Conversation(Conversation),
    Entry(Entry),
    Task(Task),
    /// A task replaces its own state. A terminal state is held while owned work is live.
    TaskState { id: Id, state: TaskState },
    /// The scheduler ends a running task's invocation without progress: it becomes
    /// pending at its checkpoint, keeping its abort mark and memos.
    Release { id: Id },
    /// Mark a live task for abort; the mark cascades to its owned work.
    Abort { id: Id },
    /// Insert or replace a submission.
    Submission(Submission),
    /// Replace a document's value. With no current incarnation at the address,
    /// creates incarnation `id` with `options`.
    Doc { address: DocAddress, options: DocOptions, id: Id, value: Value },
    /// Retire a document's current incarnation.
    Retire { address: DocAddress },
}

/// The committed result of one batch, published to observers in commit order.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Frame {
    pub seq: Seq,
    pub conversations: Vec<Conversation>,
    pub entries: Vec<Entry>,
    /// The final record of every task the commit changed, including by reconciliation.
    pub tasks: Vec<Task>,
    pub submissions: Vec<Submission>,
    pub docs: Vec<DocChange>,
}

impl Frame {
    pub fn is_empty(&self) -> bool {
        self.conversations.is_empty()
            && self.entries.is_empty()
            && self.tasks.is_empty()
            && self.submissions.is_empty()
            && self.docs.is_empty()
    }
}

/// How one document changed in a commit.
#[derive(Clone, Debug, Serialize)]
pub struct DocChange {
    pub record: DocRecord,
    #[serde(flatten)]
    pub change: Change,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Change {
    /// A new incarnation with this value.
    Created { value: Value },
    /// Operations that turn the previous value into the new one.
    Updated { ops: Vec<Op> },
    Retired,
}
