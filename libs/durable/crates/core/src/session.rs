//! The asynchronous session: one storage, one mutation line, published frames.
//!
//! SQLite runs on a dedicated thread. Reads go straight to it. Writes go
//! through a [`Tx`], which takes the mutation line at its first read, or at
//! commit if it only writes, and holds it until it commits or is dropped, so
//! a commit sees exactly the state its reads saw. Committed frames are
//! broadcast in commit order.

use std::collections::HashMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Map, Value};
use tokio::sync::{Mutex as Line, OwnedMutexGuard, broadcast, oneshot};

use crate::batch::{Frame, Write};
use crate::error::{Error, Result, invalid};
use crate::records::{
    Conversation, ConversationOwner, DocAddress, DocOptions, DocRecord, Entry, Fork, Head, Id, ROOT_CONVERSATION, Scope, Seq,
    StoredEntry, Submission, SubmissionStatus, Task, TaskState,
};
use crate::store::{Mode, Store, StoredDoc, TaskFilter};

/// Frames a slow observer may fall behind before it must resynchronize.
const FRAME_BACKLOG: usize = 1024;

type Job = Box<dyn FnOnce(&mut Store) + Send>;

/// The storage thread. Closing drops the job sender, so the thread drains
/// queued jobs, closes SQLite, and exits.
struct Db {
    jobs: Mutex<Option<std::sync::mpsc::Sender<Job>>>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
}

/// Hold the exclusive lock of a session file: `<file>.lock`, beside SQLite's own files.
///
/// A session has one owner. The owner allocates IDs and sequence numbers in
/// memory and keeps committed state warm, so a second process writing the
/// same file would corrupt it.
fn lock(path: &Path) -> Result<File> {
    let mut name = path.as_os_str().to_owned();
    name.push(".lock");
    let file = OpenOptions::new().create(true).truncate(false).write(true).open(PathBuf::from(name))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(Error::Locked(path.display().to_string())),
        Err(TryLockError::Error(error)) => Err(error.into()),
    }
}

impl Db {
    fn spawn(path: Option<PathBuf>) -> Result<Db> {
        let (jobs, inbox) = std::sync::mpsc::channel::<Job>();
        let (opened, ready) = std::sync::mpsc::channel();
        let thread = thread::Builder::new()
            .name("durable-store".into())
            .spawn(move || {
                let opening = path.as_deref().map(lock).transpose().and_then(|held| Ok((held, Store::open(path.as_deref())?)));
                // The lock is released only after SQLite has closed.
                let (_held, mut store) = match opening {
                    Ok(opened) => opened,
                    Err(error) => {
                        let _ = opened.send(Err(error));
                        return;
                    }
                };
                let _ = opened.send(Ok(()));
                for job in inbox {
                    job(&mut store);
                }
                drop(store);
            })
            .map_err(|error| Error::Corrupt(format!("cannot start the storage thread: {error}")))?;
        ready.recv().map_err(|_| Error::Closed)??;
        Ok(Db { jobs: Mutex::new(Some(jobs)), thread: Mutex::new(Some(thread)) })
    }

    async fn call<R: Send + 'static>(&self, job: impl FnOnce(&mut Store) -> Result<R> + Send + 'static) -> Result<R> {
        let (reply, result) = oneshot::channel();
        let job: Job = Box::new(move |store| {
            let _ = reply.send(job(store));
        });
        self.send(job)?;
        result.await.map_err(|_| Error::Closed)?
    }

    fn send(&self, job: Job) -> Result<()> {
        let jobs = self.jobs.lock().expect("job sender lock poisoned");
        jobs.as_ref().ok_or(Error::Closed)?.send(job).map_err(|_| Error::Closed)
    }

    /// Stop accepting jobs and wait until the storage thread has closed SQLite.
    async fn close(&self) {
        self.jobs.lock().expect("job sender lock poisoned").take();
        let thread = self.thread.lock().expect("thread lock poisoned").take();
        if let Some(thread) = thread {
            // A panicked storage thread has nothing left to close.
            let _ = tokio::task::spawn_blocking(move || thread.join()).await;
        }
    }
}

impl Drop for Db {
    /// A session dropped without closing still closes SQLite and releases its file before it is gone.
    fn drop(&mut self) {
        self.jobs.get_mut().expect("job sender lock poisoned").take();
        if let Some(thread) = self.thread.get_mut().expect("thread lock poisoned").take() {
            let _ = thread.join();
        }
    }
}

struct Inner {
    db: Db,
    line: Arc<Line<()>>,
    next_id: AtomicI64,
    /// Taken on close, which ends every subscriber's stream.
    frames: Mutex<Option<broadcast::Sender<Arc<Frame>>>>,
}

/// An open durable session. Cheap to clone; clones share the session.
#[derive(Clone)]
pub struct Session {
    inner: Arc<Inner>,
}

macro_rules! read {
    ($self:ident, |$store:ident| $body:expr) => {
        $self.inner.db.call(move |$store| $body).await
    };
}

impl Session {
    /// Open a session over a database file, or an in-memory database without a path.
    pub async fn open(path: Option<PathBuf>) -> Result<Session> {
        let db = tokio::task::spawn_blocking(move || Db::spawn(path)).await.map_err(|_| Error::Closed)??;
        let next_id = db.call(|store| Ok(store.next_id())).await?;
        let (frames, _) = broadcast::channel(FRAME_BACKLOG);
        let inner = Inner { db, line: Arc::new(Line::new(())), next_id: AtomicI64::new(next_id), frames: Mutex::new(Some(frames)) };
        Ok(Session { inner: Arc::new(inner) })
    }

    /// Close the session once the current transaction, if any, settles. Later calls
    /// fail with [`Error::Closed`]; observers see their frame stream end.
    pub async fn close(&self) {
        let _line = self.inner.line.lock().await;
        self.inner.db.close().await;
        self.inner.frames.lock().expect("frame sender lock poisoned").take();
    }

    /// Start a transaction. It takes the mutation line at its first read or at commit.
    pub fn begin(&self) -> Tx {
        Tx { inner: self.inner.clone(), state: Mutex::new(Some(TxState::new())) }
    }

    /// Observe every commit from now on, in order.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<Frame>> {
        match self.inner.frames.lock().expect("frame sender lock poisoned").as_ref() {
            Some(frames) => frames.subscribe(),
            // A closed session publishes nothing: hand out an already-ended stream.
            None => broadcast::channel(1).1,
        }
    }

    /// Reserve the oldest eligible pending task of one of `kinds`.
    pub async fn reserve(&self, kinds: Vec<String>) -> Result<Option<(Task, Mode)>> {
        let _line = self.inner.line.lock().await;
        let reserved = self.inner.db.call(move |store| store.reserve(&kinds)).await?;
        Ok(reserved.map(|(task, mode, frame)| {
            self.publish(frame);
            (task, mode)
        }))
    }

    fn publish(&self, frame: Frame) -> Arc<Frame> {
        let frame = Arc::new(frame);
        if let Some(frames) = self.inner.frames.lock().expect("frame sender lock poisoned").as_ref() {
            // Nobody watching is fine.
            let _ = frames.send(frame.clone());
        }
        frame
    }

    /// The last committed sequence.
    pub async fn seq(&self) -> Result<Seq> {
        read!(self, |store| Ok(store.seq()))
    }

    pub async fn conversation(&self, id: Id) -> Result<Option<Conversation>> {
        read!(self, |store| store.conversation(id))
    }

    pub async fn conversations(&self, after: Option<Id>, limit: usize) -> Result<Vec<Conversation>> {
        read!(self, |store| store.conversations(after, limit))
    }

    pub async fn owned_conversations(&self, task: Id) -> Result<Vec<Conversation>> {
        read!(self, |store| store.owned_conversations(task))
    }

    pub async fn entry(&self, id: Id) -> Result<Option<StoredEntry>> {
        read!(self, |store| store.entry(id))
    }

    pub async fn entries(&self, conversation: Id, after: Option<Id>, limit: usize) -> Result<Vec<StoredEntry>> {
        read!(self, |store| store.entries(conversation, after, limit))
    }

    pub async fn context(&self, conversation: Id, at: Option<Seq>) -> Result<Vec<StoredEntry>> {
        read!(self, |store| store.context(conversation, at))
    }

    /// [`Session::context`] as JSON text, for hosts that decode it themselves.
    pub async fn context_json(&self, conversation: Id, at: Option<Seq>) -> Result<String> {
        read!(self, |store| store.context_json(conversation, at))
    }

    pub async fn task(&self, id: Id) -> Result<Option<Task>> {
        read!(self, |store| store.task(id))
    }

    pub async fn tasks(&self, filter: TaskFilter) -> Result<Vec<Task>> {
        read!(self, |store| store.tasks(&filter))
    }

    pub async fn submission(&self, id: Id) -> Result<Option<Submission>> {
        read!(self, |store| store.submission(id))
    }

    pub async fn submission_by_request(&self, conversation: Id, request_id: String) -> Result<Option<Submission>> {
        read!(self, |store| store.submission_by_request(conversation, &request_id))
    }

    pub async fn submissions(&self, conversation: Id, status: Option<SubmissionStatus>) -> Result<Vec<Submission>> {
        read!(self, |store| store.submissions(conversation, status))
    }

    pub async fn doc(&self, address: DocAddress, at: Option<Seq>) -> Result<Option<StoredDoc>> {
        read!(self, |store| store.doc(&address, at))
    }

    pub async fn docs(&self, scope: Scope, kind: Option<String>) -> Result<Vec<DocRecord>> {
        read!(self, |store| store.docs(scope, kind.as_deref()))
    }

    /// Wait until a task is terminal and return its final record.
    pub async fn wait_task(&self, id: Id) -> Result<Task> {
        self.wait(
            move |session| async move { session.task(id).await?.ok_or_else(|| invalid(format!("unknown task: {id}"))) },
            |task: &Task| !task.state.live(),
            move |frame| frame.tasks.iter().find(|task| task.id == id).cloned(),
        )
        .await
    }

    /// Wait until a submission is done or unanswered.
    pub async fn wait_submission(&self, id: Id) -> Result<Submission> {
        self.wait(
            move |session| async move { session.submission(id).await?.ok_or_else(|| invalid(format!("unknown submission: {id}"))) },
            |submission: &Submission| submission.status.settled(),
            move |frame| frame.submissions.iter().find(|submission| submission.id == id).cloned(),
        )
        .await
    }

    /// Subscribe first, then read, so no commit between the two is missed.
    async fn wait<T, F>(&self, read: impl Fn(Session) -> F, done: impl Fn(&T) -> bool, find: impl Fn(&Frame) -> Option<T>) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let mut frames = self.subscribe();
        loop {
            let current = read(self.clone()).await?;
            if done(&current) {
                return Ok(current);
            }
            loop {
                match frames.recv().await {
                    Ok(frame) => {
                        if let Some(record) = find(&frame).filter(|record| done(record)) {
                            return Ok(record);
                        }
                    }
                    // Missed frames may have settled it: read again.
                    Err(broadcast::error::RecvError::Lagged(_)) => break,
                    Err(broadcast::error::RecvError::Closed) => return Err(Error::Closed),
                }
            }
        }
    }
}

/// A document's pending fate inside one transaction.
struct DocWork {
    /// Retire the incarnation that was current when the transaction began.
    retire: bool,
    /// The value to leave, or `None` when the document ends retired.
    value: Option<(DocOptions, Value)>,
}

struct TxState {
    /// Taken at the first read, or at commit.
    line: Option<OwnedMutexGuard<()>>,
    writes: Vec<Write>,
    docs: HashMap<DocAddress, DocWork>,
    /// Document addresses in first-touch order, so commits are deterministic.
    doc_order: Vec<DocAddress>,
    wrote_tables: bool,
}

impl TxState {
    fn new() -> TxState {
        TxState { line: None, writes: Vec::new(), docs: HashMap::new(), doc_order: Vec::new(), wrote_tables: false }
    }

    fn table_write(&mut self, write: Write) {
        self.wrote_tables = true;
        self.writes.push(write);
    }

    fn touch(&mut self, address: &DocAddress) -> &mut DocWork {
        if !self.docs.contains_key(address) {
            self.doc_order.push(address.clone());
        }
        self.docs.entry(address.clone()).or_insert(DocWork { retire: false, value: None })
    }
}

/// One atomic change. It holds the mutation line from its first read to its commit.
///
/// Table reads see committed state, so they must come before the
/// transaction's first table write. Document reads see the transaction's own
/// writes.
pub struct Tx {
    inner: Arc<Inner>,
    state: Mutex<Option<TxState>>,
}

macro_rules! table_read {
    ($self:ident, |$store:ident| $body:expr) => {{
        $self.readable().await?;
        $self.inner.db.call(move |$store| $body).await
    }};
}

impl Tx {
    fn with<R>(&self, f: impl FnOnce(&mut TxState) -> Result<R>) -> Result<R> {
        let mut state = self.state.lock().expect("transaction lock poisoned");
        f(state.as_mut().ok_or(Error::Finished)?)
    }

    fn write(&self, write: Write) -> Result<()> {
        self.with(|state| {
            state.table_write(write);
            Ok(())
        })
    }

    fn mint(&self) -> Id {
        self.inner.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Table reads see committed state, so they must come before table writes,
    /// and they take the line so nothing commits between the read and this commit.
    async fn readable(&self) -> Result<()> {
        let held = self.with(|state| if state.wrote_tables { Err(Error::ReadAfterWrite) } else { Ok(state.line.is_some()) })?;
        if !held {
            self.hold_line().await?;
        }
        Ok(())
    }

    async fn hold_line(&self) -> Result<()> {
        let line = self.inner.line.clone().lock_owned().await;
        self.with(|state| {
            state.line.get_or_insert(line);
            Ok(())
        })
    }

    /// Commit only if `task` is still running, and, with `unmarked`, carries no abort mark.
    /// The check happens on the line, atomically with the commit.
    pub fn require_running(&self, task: Id, unmarked: bool) -> Result<()> {
        self.with(|state| {
            state.writes.insert(0, Write::Require { task, unmarked });
            Ok(())
        })
    }

    pub async fn conversation(&self, id: Id) -> Result<Option<Conversation>> {
        table_read!(self, |store| store.conversation(id))
    }

    pub async fn owned_conversations(&self, task: Id) -> Result<Vec<Conversation>> {
        table_read!(self, |store| store.owned_conversations(task))
    }

    pub async fn entry(&self, id: Id) -> Result<Option<StoredEntry>> {
        table_read!(self, |store| store.entry(id))
    }

    pub async fn context(&self, conversation: Id) -> Result<Vec<StoredEntry>> {
        table_read!(self, |store| store.context(conversation, None))
    }

    pub async fn task(&self, id: Id) -> Result<Option<Task>> {
        table_read!(self, |store| store.task(id))
    }

    pub async fn tasks(&self, filter: TaskFilter) -> Result<Vec<Task>> {
        table_read!(self, |store| store.tasks(&filter))
    }

    pub async fn submission(&self, id: Id) -> Result<Option<Submission>> {
        table_read!(self, |store| store.submission(id))
    }

    pub async fn submission_by_request(&self, conversation: Id, request_id: String) -> Result<Option<Submission>> {
        table_read!(self, |store| store.submission_by_request(conversation, &request_id))
    }

    pub async fn submissions(&self, conversation: Id, status: Option<SubmissionStatus>) -> Result<Vec<Submission>> {
        table_read!(self, |store| store.submissions(conversation, status))
    }

    /// A document's value as this transaction would leave it.
    pub async fn doc(&self, address: DocAddress) -> Result<Option<Value>> {
        if self.with(|state| Ok(state.line.is_none()))? {
            self.hold_line().await?;
        }
        let pending = self.with(|state| Ok(state.docs.get(&address).map(|work| work.value.as_ref().map(|(_, value)| value.clone()))))?;
        match pending {
            Some(value) => Ok(value),
            None => Ok(self.inner.db.call(move |store| store.doc(&address, None)).await?.map(|stored| stored.value)),
        }
    }

    /// Create the root conversation, which has a fixed ID.
    pub fn create_root(&self) -> Result<Id> {
        self.write(Write::Conversation(Conversation { id: ROOT_CONVERSATION, parent: None, owner: None }))?;
        Ok(ROOT_CONVERSATION)
    }

    pub fn create_conversation(&self, parent: Option<Fork>, owner: Option<ConversationOwner>) -> Result<Id> {
        let id = self.mint();
        self.write(Write::Conversation(Conversation { id, parent, owner }))?;
        Ok(id)
    }

    /// Append an entry. `content` holds `model`, `data`, `edits`, or other fields.
    pub fn append_entry(
        &self,
        conversation: Id,
        kind: String,
        content: Map<String, Value>,
        head: Option<Head>,
        by_task: Option<Id>,
    ) -> Result<Id> {
        let id = self.mint();
        let head = head.map(|head| match head {
            Head::SelfEntry => id,
            Head::Entry(entry) => entry,
        });
        if head.is_some_and(|head| head > id) {
            return Err(invalid("a head cannot point past its own entry"));
        }
        self.write(Write::Entry(Entry { id, conversation_id: conversation, kind, head, by_task_id: by_task, content }))?;
        Ok(id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_task(
        &self,
        conversation: Id,
        kind: String,
        version: i64,
        input: Value,
        checkpoint: Value,
        owner: Option<Id>,
        background: bool,
    ) -> Result<Id> {
        let id = self.mint();
        let state = TaskState::Pending { checkpoint };
        let task = Task { id, conversation_id: conversation, kind, version, input, owner, background, abort_requested: false, state, memos: None };
        self.write(Write::Task(task))?;
        Ok(id)
    }

    /// A task replaces its own state: running, waiting, or terminal.
    pub fn set_task_state(&self, id: Id, state: TaskState) -> Result<()> {
        self.write(Write::TaskState { id, state })
    }

    /// End a running task's invocation: it becomes pending at its checkpoint.
    pub fn release_task(&self, id: Id) -> Result<()> {
        self.write(Write::Release { id })
    }

    pub fn abort_task(&self, id: Id) -> Result<()> {
        self.write(Write::Abort { id })
    }

    /// Admit a submission. `content` holds `type` and any other fields.
    pub fn create_submission(
        &self,
        conversation: Id,
        request_id: Option<String>,
        status: SubmissionStatus,
        content: Map<String, Value>,
    ) -> Result<Id> {
        let id = self.mint();
        self.write(Write::Submission(Submission { id, conversation_id: conversation, request_id, status, content }))?;
        Ok(id)
    }

    /// Replace a submission record read earlier in this transaction.
    pub fn put_submission(&self, submission: Submission) -> Result<()> {
        self.write(Write::Submission(submission))
    }

    /// Set a document's value. `options` apply when this creates an incarnation.
    pub fn put_doc(&self, address: DocAddress, options: DocOptions, value: Value) -> Result<()> {
        self.with(|state| {
            state.touch(&address).value = Some((options, value));
            Ok(())
        })
    }

    pub fn retire_doc(&self, address: DocAddress) -> Result<()> {
        self.with(|state| {
            let work = state.touch(&address);
            // A value written earlier in this transaction never existed for anyone else.
            if work.value.take().is_none() {
                work.retire = true;
            }
            Ok(())
        })
    }

    /// Commit atomically and publish the frame. The line is released afterwards.
    pub async fn commit(&self) -> Result<Arc<Frame>> {
        let state = self.state.lock().expect("transaction lock poisoned").take().ok_or(Error::Finished)?;
        let TxState { line, mut writes, mut docs, doc_order, .. } = state;
        let line = match line {
            Some(line) => line,
            None => self.inner.line.clone().lock_owned().await,
        };
        for address in doc_order {
            let work = docs.remove(&address).expect("ordered address has work");
            if work.retire {
                writes.push(Write::Retire { address: address.clone() });
            }
            if let Some((options, value)) = work.value {
                writes.push(Write::Doc { address, options, id: self.mint(), value });
            }
        }
        let next_id = self.inner.next_id.load(Ordering::SeqCst);
        let frame = self.inner.db.call(move |store| store.commit(writes, next_id)).await?;
        let frame = Session { inner: self.inner.clone() }.publish(frame);
        drop(line);
        Ok(frame)
    }

    /// Abandon the transaction without writing. Dropping does the same.
    pub fn rollback(&self) {
        self.state.lock().expect("transaction lock poisoned").take();
    }
}
