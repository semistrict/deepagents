//! A durable agent kernel: conversations, immutable entries, durable tasks,
//! submissions, and JSON documents, committed atomically through one
//! mutation line and published as ordered frames.

mod batch;
pub mod delta;
mod error;
pub mod inspect;
mod records;
pub mod scheduler;
mod session;
mod store;

pub use batch::{Change, DocChange, Frame, Write};
pub use error::{Error, Result};
pub use records::{
    Conversation, ConversationOwner, DocAddress, DocOptions, DocRecord, Entry, Fork, ForkPolicy, Head, History, Id,
    JoinPolicy, Outcome, OutcomeError, ROOT_CONVERSATION, Scope, Seq, StoredEntry, Submission, SubmissionStatus, Task,
    TaskState,
};
pub use scheduler::{HandlerError, Invocation, Scheduler, Step, TaskHandler};
pub use session::{Session, Tx};
pub use store::{Mode, Store, StoredDoc, TaskFilter};
