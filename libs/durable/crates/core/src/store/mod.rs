//! Synchronous SQLite storage in pi-durable's format. The
//! [`Session`](crate::Session) owns it on one thread.

mod docs;

pub use docs::StoredDoc;
mod reconcile;
mod rows;
mod schema;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, Transaction, params};

use crate::batch::{Frame, Write};
use crate::error::{Error, Result, invalid};
use crate::records::{Conversation, DocAddress, DocRecord, Id, Scope, Seq, StoredEntry, Submission, SubmissionStatus, Task, TaskState};

use reconcile::Graph;
use rows::{indexed, json, record};

/// Whether a reserved task runs its phases or its abort handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Run,
    Abort,
}

/// Which tasks to list. Unset fields match everything.
#[derive(Clone, Debug, Default)]
pub struct TaskFilter {
    pub conversation: Option<Id>,
    pub kind: Option<String>,
    pub status: Option<String>,
    pub abort_requested: Option<bool>,
    pub background: Option<bool>,
    /// Only tasks that are not terminal.
    pub live: bool,
}

pub struct Store {
    conn: Connection,
    /// The sequence the next commit gets.
    next_seq: Seq,
    next_id: Id,
}

impl Store {
    /// Open a database file, or an in-memory database without a path.
    pub fn open(path: Option<&Path>) -> Result<Store> {
        let mut conn = match path {
            Some(path) => Connection::open(path)?,
            None => Connection::open_in_memory()?,
        };
        schema::migrate(&mut conn)?;
        let (next_id, next_seq): (String, Seq) =
            conn.query_row("SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1", [], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let next_id = next_id.parse().map_err(|_| Error::Corrupt(format!("next_id {next_id} is not an integer")))?;
        let mut store = Store { conn, next_seq, next_id };
        store.recover()?;
        Ok(store)
    }

    /// The last committed sequence.
    pub fn seq(&self) -> Seq {
        self.next_seq - 1
    }

    /// The first ID not yet handed out.
    pub fn next_id(&self) -> Id {
        self.next_id
    }

    /// Running tasks did not survive the last process; they become pending again.
    fn recover(&mut self) -> Result<()> {
        let filter = TaskFilter { status: Some("running".into()), ..TaskFilter::default() };
        let writes: Vec<Write> = self.tasks(&filter)?.into_iter().map(|task| Write::Release { id: task.id }).collect();
        if !writes.is_empty() {
            self.commit(writes, self.next_id)?;
        }
        Ok(())
    }

    // Reads.

    pub fn conversation(&self, id: Id) -> Result<Option<Conversation>> {
        Ok(self.conn.prepare_cached("SELECT record FROM conversations WHERE id = ?1")?.query_row([id], record).optional()?)
    }

    /// Conversations with IDs after `after`, in creation order.
    pub fn conversations(&self, after: Option<Id>, limit: usize) -> Result<Vec<Conversation>> {
        let mut stmt = self.conn.prepare_cached("SELECT record FROM conversations WHERE id > ?1 ORDER BY id LIMIT ?2")?;
        collect(stmt.query_map(params![after.unwrap_or(0), i64::try_from(limit).unwrap_or(i64::MAX)], record)?)
    }

    /// Conversations owned by a task, in creation order.
    pub fn owned_conversations(&self, task: Id) -> Result<Vec<Conversation>> {
        let mut stmt = self.conn.prepare_cached("SELECT record FROM conversations WHERE owner_task_id = ?1 ORDER BY id")?;
        collect(stmt.query_map([task], record)?)
    }

    pub fn entry(&self, id: Id) -> Result<Option<StoredEntry>> {
        let mut stmt = self.conn.prepare_cached("SELECT record, commit_seq FROM entries WHERE id = ?1")?;
        Ok(stmt.query_row([id], rows::entry).optional()?)
    }

    /// Entries visible from `conversation` with IDs after `after`, oldest first.
    pub fn entries(&self, conversation: Id, after: Option<Id>, limit: usize) -> Result<Vec<StoredEntry>> {
        let lineage = self.lineage(conversation)?;
        self.visible(&lineage, after.map_or(0, |id| id + 1), Id::MAX, Visible::All, limit, rows::entry)
    }

    /// The active transcript of `conversation` as of commit `at` (default: now).
    ///
    /// It starts at the newest head marker: the marker itself, then every other
    /// entry from the marker's head onward. Head markers inside the range are dropped.
    pub fn context(&self, conversation: Id, at: Option<Seq>) -> Result<Vec<StoredEntry>> {
        self.context_rows(conversation, at, rows::entry, |stored| stored.entry.head)
    }

    /// [`Store::context`] as one JSON array of `{"entry", "commitSeq"}` objects, built
    /// from the stored records without decoding them.
    pub fn context_json(&self, conversation: Id, at: Option<Seq>) -> Result<String> {
        let raw = self.context_rows(conversation, at, rows::raw_entry, |raw| raw.head)?;
        let mut json = String::with_capacity(raw.iter().map(|raw| raw.record.len() + 32).sum::<usize>() + 2);
        json.push('[');
        for (index, raw) in raw.iter().enumerate() {
            if index > 0 {
                json.push(',');
            }
            json.push_str(r#"{"entry":"#);
            json.push_str(&raw.record);
            json.push_str(&format!(r#","commitSeq":{}}}"#, raw.seq));
        }
        json.push(']');
        Ok(json)
    }

    fn context_rows<T>(
        &self,
        conversation: Id,
        at: Option<Seq>,
        row: fn(&rusqlite::Row) -> rusqlite::Result<T>,
        head: fn(&T) -> Option<Id>,
    ) -> Result<Vec<T>> {
        let lineage = self.lineage(conversation)?;
        let cutoff = match at {
            None => Id::MAX,
            Some(seq) => match self.last_visible(&lineage, seq)? {
                Some(id) => id,
                None => return Ok(Vec::new()),
            },
        };
        let Some(marker) = self.visible(&lineage, 0, cutoff, Visible::NewestMarker, 1, row)?.pop() else {
            return self.visible(&lineage, 0, cutoff, Visible::All, usize::MAX, row);
        };
        let start = head(&marker).expect("head markers have a head");
        let mut context = vec![marker];
        context.extend(self.visible(&lineage, start, cutoff, Visible::Unmarked, usize::MAX, row)?);
        Ok(context)
    }

    /// The conversation and its history parents, each with the highest entry ID it contributes.
    fn lineage(&self, conversation: Id) -> Result<Vec<(Id, Id)>> {
        let mut lineage = Vec::new();
        let (mut current, mut cap) = (Some(conversation), Id::MAX);
        while let Some(id) = current {
            let record = self.conversation(id)?.ok_or_else(|| invalid(format!("unknown conversation: {id}")))?;
            lineage.push((id, cap));
            current = record.parent.map(|fork| {
                cap = cap.min(fork.at);
                fork.conversation_id
            });
        }
        Ok(lineage)
    }

    fn visible<T>(
        &self,
        lineage: &[(Id, Id)],
        from: Id,
        through: Id,
        which: Visible,
        limit: usize,
        row: fn(&rusqlite::Row) -> rusqlite::Result<T>,
    ) -> Result<Vec<T>> {
        let (filter, order) = match which {
            Visible::All => ("", "ASC"),
            Visible::Unmarked => ("AND head IS NULL", "ASC"),
            Visible::NewestMarker => ("AND head IS NOT NULL", "DESC"),
        };
        let sql = format!(
            "SELECT record, commit_seq, head FROM entries
             WHERE ({}) AND id >= ?1 AND id <= ?2 {filter} ORDER BY id {order} LIMIT ?3",
            lineage_clause(lineage)
        );
        let mut stmt = self.conn.prepare_cached(&sql)?;
        collect(stmt.query_map(params![from, through, i64::try_from(limit).unwrap_or(i64::MAX)], row)?)
    }

    fn last_visible(&self, lineage: &[(Id, Id)], seq: Seq) -> Result<Option<Id>> {
        let sql = format!("SELECT max(id) FROM entries WHERE ({}) AND commit_seq <= ?1", lineage_clause(lineage));
        Ok(self.conn.prepare_cached(&sql)?.query_row([seq], |row| row.get(0))?)
    }

    pub fn task(&self, id: Id) -> Result<Option<Task>> {
        task_by_id(&self.conn, id)
    }

    pub fn tasks(&self, filter: &TaskFilter) -> Result<Vec<Task>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT record FROM tasks
             WHERE (?1 IS NULL OR conversation_id = ?1) AND (?2 IS NULL OR kind = ?2) AND (?3 IS NULL OR status = ?3)
               AND (?4 IS NULL OR abort_requested = ?4) AND (?5 IS NULL OR background = ?5)
               AND (?6 = 0 OR status IN ('pending', 'running', 'waiting', 'completing'))
             ORDER BY id",
        )?;
        let kind = filter.kind.as_deref().map(indexed);
        let params = params![filter.conversation, kind, filter.status, filter.abort_requested, filter.background, filter.live];
        collect(stmt.query_map(params, record)?)
    }

    pub fn submission(&self, id: Id) -> Result<Option<Submission>> {
        Ok(self.conn.prepare_cached("SELECT record FROM submissions WHERE id = ?1")?.query_row([id], record).optional()?)
    }

    pub fn submission_by_request(&self, conversation: Id, request_id: &str) -> Result<Option<Submission>> {
        let mut stmt = self.conn.prepare_cached("SELECT record FROM submissions WHERE conversation_id = ?1 AND request_id = ?2")?;
        Ok(stmt.query_row(params![conversation, indexed(request_id)], record).optional()?)
    }

    /// Submissions of a conversation, optionally with one status, oldest first.
    pub fn submissions(&self, conversation: Id, status: Option<SubmissionStatus>) -> Result<Vec<Submission>> {
        let mut stmt =
            self.conn.prepare_cached("SELECT record FROM submissions WHERE conversation_id = ?1 AND (?2 IS NULL OR status = ?2) ORDER BY id")?;
        collect(stmt.query_map(params![conversation, status.map(SubmissionStatus::as_str)], record)?)
    }

    /// A document now, or as of commit `at` for a rewindable conversation document.
    pub fn doc(&self, address: &DocAddress, at: Option<Seq>) -> Result<Option<StoredDoc>> {
        docs::read(&self.conn, address, at)
    }

    /// The current documents of a scope, optionally of one kind.
    pub fn docs(&self, scope: Scope, kind: Option<&str>) -> Result<Vec<DocRecord>> {
        docs::list(&self.conn, scope, kind)
    }

    // Writes.

    /// Atomically apply `writes`, reconcile tasks, and return the published frame.
    /// `next_id` is the first ID the writer has not handed out.
    pub fn commit(&mut self, writes: Vec<Write>, next_id: Id) -> Result<Frame> {
        let seq = self.next_seq;
        let tx = self.conn.transaction()?;
        let mut frame = Frame { seq, ..Frame::default() };
        let mut tasks = BTreeSet::new();
        for write in writes {
            apply(&tx, seq, write, &mut frame, &mut tasks)?;
        }
        tasks.extend(settle(&tx, seq, &mut frame)?);
        for id in tasks {
            frame.tasks.push(task_by_id(&tx, id)?.expect("changed task exists"));
        }
        let next_id = next_id.max(self.next_id);
        tx.execute("UPDATE durable_metadata SET next_id = ?1, next_seq = ?2 WHERE singleton = 1", params![next_id.to_string(), seq + 1])?;
        tx.commit()?;
        self.next_seq = seq + 1;
        self.next_id = next_id;
        Ok(frame)
    }

    /// Reserve the oldest eligible pending task of one of `kinds` for an invocation.
    pub fn reserve(&mut self, kinds: &[String]) -> Result<Option<(Task, Mode, Frame)>> {
        let graph = load_graph(&self.conn)?;
        let candidate = graph.tasks.values().find_map(|task| {
            let TaskState::Pending { checkpoint } = &task.state else {
                return None;
            };
            if !kinds.contains(&task.kind) {
                return None;
            }
            let mode = match task.abort_requested {
                false => Mode::Run,
                // Abort runs bottom-up: only once the work below has drained.
                true if !graph.has_owned_work(task.id) => Mode::Abort,
                true => return None,
            };
            Some((task.id, checkpoint.clone(), mode))
        });
        let Some((id, checkpoint, mode)) = candidate else {
            return Ok(None);
        };
        let frame = self.commit(vec![Write::TaskState { id, state: TaskState::Running { checkpoint } }], self.next_id)?;
        let task = frame.tasks.iter().find(|task| task.id == id).cloned().expect("reserved task is in its frame");
        Ok(Some((task, mode, frame)))
    }
}

#[derive(Clone, Copy)]
enum Visible {
    All,
    /// Every entry without a head.
    Unmarked,
    /// Only the newest head marker.
    NewestMarker,
}

fn lineage_clause(lineage: &[(Id, Id)]) -> String {
    lineage.iter().map(|(conversation, cap)| format!("(conversation_id = {conversation} AND id <= {cap})")).collect::<Vec<_>>().join(" OR ")
}

/// Record an ID's table in the global namespace. Immutable records claim a fresh ID.
fn claim(tx: &Transaction, id: Id, table: &str) -> Result<()> {
    let existing: Option<String> =
        tx.prepare_cached("SELECT record_type FROM record_ids WHERE id = ?1")?.query_row([id], |row| row.get(0)).optional()?;
    match existing {
        None => {
            tx.prepare_cached("INSERT INTO record_ids (id, record_type) VALUES (?1, ?2)")?.execute(params![id, table])?;
            Ok(())
        }
        Some(owner) if owner == table && matches!(table, "task" | "submission") => Ok(()),
        Some(owner) => Err(invalid(format!("ID {id} already belongs to {owner}"))),
    }
}

fn apply(tx: &Transaction, seq: Seq, write: Write, frame: &mut Frame, tasks: &mut BTreeSet<Id>) -> Result<()> {
    match write {
        Write::Require { task, unmarked } => {
            let current = task_by_id(tx, task)?;
            let running = matches!(current, Some(Task { state: TaskState::Running { .. }, abort_requested, .. }) if !(unmarked && abort_requested));
            if !running {
                return Err(Error::InvocationEnded(format!("task {task} is no longer running under this invocation")));
            }
        }
        Write::Conversation(conversation) => {
            if let Some(owner) = &conversation.owner
                && check_owner(tx, owner.task_id)?.conversation_id != owner.conversation_id
            {
                return Err(invalid(format!("conversation {} names the wrong owner conversation", conversation.id)));
            }
            claim(tx, conversation.id, "conversation")?;
            let owner = conversation.owner.as_ref();
            tx.prepare_cached("INSERT INTO conversations (id, owner_conversation_id, owner_task_id, record) VALUES (?1, ?2, ?3, ?4)")?.execute(
                params![conversation.id, owner.map(|owner| owner.conversation_id), owner.map(|owner| owner.task_id), json(&conversation)?],
            )?;
            frame.conversations.push(conversation);
        }
        Write::Entry(entry) => {
            claim(tx, entry.id, "entry")?;
            tx.prepare_cached("INSERT INTO entries (id, conversation_id, head, commit_seq, record) VALUES (?1, ?2, ?3, ?4, ?5)")?
                .execute(params![entry.id, entry.conversation_id, entry.head, seq, json(&entry)?])?;
            frame.entries.push(entry);
        }
        Write::Task(task) => {
            if !matches!(task.state, TaskState::Pending { .. }) || task.abort_requested {
                return Err(invalid(format!("task {} must start pending and unmarked", task.id)));
            }
            if let Some(owner) = task.owner {
                if task.background {
                    return Err(invalid(format!("child task {} cannot be background", task.id)));
                }
                if check_owner(tx, owner)?.conversation_id != task.conversation_id {
                    return Err(invalid(format!("child task {} must live in its owner's conversation", task.id)));
                }
            }
            claim(tx, task.id, "task")?;
            put_task(tx, &task)?;
            tasks.insert(task.id);
        }
        Write::TaskState { id, state } => {
            let mut task = task_by_id(tx, id)?.ok_or_else(|| invalid(format!("unknown task: {id}")))?;
            check_transition(tx, &task, &state)?;
            task.state = match state {
                // A task finishes only once its ordinary owned work drained; settle() decides.
                TaskState::Terminal { outcome } => {
                    task.memos = None;
                    TaskState::Completing { outcome }
                }
                state => state,
            };
            put_task(tx, &task)?;
            tasks.insert(id);
        }
        Write::Release { id } => {
            let mut task = task_by_id(tx, id)?.ok_or_else(|| invalid(format!("unknown task: {id}")))?;
            let TaskState::Running { checkpoint } = task.state else {
                return Err(invalid(format!("task {id} is {} and cannot be released", task.state.status())));
            };
            task.state = TaskState::Pending { checkpoint };
            put_task(tx, &task)?;
            tasks.insert(id);
        }
        Write::Abort { id } => {
            let mut task = task_by_id(tx, id)?.ok_or_else(|| invalid(format!("unknown task: {id}")))?;
            if task.state.live() && !task.abort_requested {
                task.abort_requested = true;
                put_task(tx, &task)?;
                tasks.insert(id);
            }
        }
        Write::Submission(submission) => {
            claim(tx, submission.id, "submission")?;
            tx.prepare_cached(
                "INSERT INTO submissions (id, conversation_id, request_id, status, record) VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(id) DO UPDATE SET conversation_id = excluded.conversation_id,
                 request_id = excluded.request_id, status = excluded.status, record = excluded.record",
            )?
            .execute(params![
                submission.id,
                submission.conversation_id,
                submission.request_id.as_deref().map(indexed),
                submission.status.as_str(),
                json(&submission)?
            ])?;
            frame.submissions.push(submission);
        }
        Write::Doc { address, options, id, value } => frame.docs.extend(docs::put(tx, seq, &address, options, id, value)?),
        Write::Retire { address } => frame.docs.extend(docs::retire(tx, seq, &address)?),
    }
    Ok(())
}

/// A task moves itself: from pending or running to running, waiting, or terminal.
fn check_transition(tx: &Transaction, task: &Task, next: &TaskState) -> Result<()> {
    let id = task.id;
    if !matches!(task.state, TaskState::Pending { .. } | TaskState::Running { .. }) {
        return Err(invalid(format!("task {id} is {} and cannot change its state", task.state.status())));
    }
    match next {
        TaskState::Running { .. } | TaskState::Terminal { .. } => Ok(()),
        TaskState::Waiting { on, .. } => {
            for other in on {
                if *other == id {
                    return Err(invalid(format!("task {id} cannot wait on itself")));
                }
                task_by_id(tx, *other)?.ok_or_else(|| invalid(format!("task {id} waits on unknown task {other}")))?;
            }
            let mut owner = task.owner;
            while let Some(above) = owner {
                if on.contains(&above) {
                    return Err(invalid(format!("task {id} cannot wait on its owner {above}")));
                }
                owner = task_by_id(tx, above)?.and_then(|task| task.owner);
            }
            Ok(())
        }
        TaskState::Pending { .. } | TaskState::Completing { .. } => Err(invalid(format!("task {id} cannot move itself to {}", next.status()))),
    }
}

/// New owned work needs an owner that can still own it.
fn check_owner(tx: &Transaction, owner: Id) -> Result<Task> {
    let task = task_by_id(tx, owner)?.ok_or_else(|| invalid(format!("unknown owner task: {owner}")))?;
    if !task.state.open() || task.abort_requested {
        return Err(invalid(format!("owner task {owner} is finishing and cannot own new work")));
    }
    Ok(task)
}

/// Run the reconciliation rules over the candidate records and store what changed.
fn settle(tx: &Transaction, seq: Seq, frame: &mut Frame) -> Result<BTreeSet<Id>> {
    let mut graph = load_graph(tx)?;
    let changed = graph.reconcile();
    for id in &changed {
        let task = &graph.tasks[id];
        put_task(tx, task)?;
        if !task.state.live() {
            frame.docs.extend(docs::retire_scope(tx, seq, Scope::Task { task_id: *id })?);
        }
    }
    Ok(changed)
}

/// Every live task, the terminal tasks live waits name, and their conversations' owners.
fn load_graph(conn: &Connection) -> Result<Graph> {
    let mut stmt = conn.prepare_cached("SELECT record FROM tasks WHERE status IN ('pending', 'running', 'waiting', 'completing') ORDER BY id")?;
    let mut tasks: BTreeMap<Id, Task> = collect(stmt.query_map([], record::<Task>)?)?.into_iter().map(|task| (task.id, task)).collect();
    let awaited: Vec<Id> = tasks
        .values()
        .filter_map(|task| match &task.state {
            TaskState::Waiting { on, .. } => Some(on.clone()),
            _ => None,
        })
        .flatten()
        .filter(|id| !tasks.contains_key(id))
        .collect();
    for id in awaited {
        if let Some(task) = task_by_id(conn, id)? {
            tasks.insert(id, task);
        }
    }
    let mut conversation_owner = HashMap::new();
    let mut owner_stmt = conn.prepare_cached("SELECT owner_task_id FROM conversations WHERE id = ?1")?;
    for conversation in tasks.values().map(|task| task.conversation_id).collect::<BTreeSet<_>>() {
        let owner: Option<Id> = owner_stmt.query_row([conversation], |row| row.get(0)).optional()?.flatten();
        conversation_owner.insert(conversation, owner);
    }
    Ok(Graph { tasks, conversation_owner })
}

fn task_by_id(conn: &Connection, id: Id) -> Result<Option<Task>> {
    Ok(conn.prepare_cached("SELECT record FROM tasks WHERE id = ?1")?.query_row([id], record).optional()?)
}

fn put_task(tx: &Transaction, task: &Task) -> Result<()> {
    tx.prepare_cached(
        "INSERT INTO tasks (id, conversation_id, kind, status, abort_requested, background, record)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(id) DO UPDATE SET conversation_id = excluded.conversation_id, kind = excluded.kind,
         status = excluded.status, abort_requested = excluded.abort_requested,
         background = excluded.background, record = excluded.record",
    )?
    .execute(params![
        task.id,
        task.conversation_id,
        indexed(&task.kind),
        task.state.status(),
        task.abort_requested,
        task.background,
        json(task)?
    ])?;
    Ok(())
}

fn collect<T>(rows: impl Iterator<Item = rusqlite::Result<T>>) -> Result<Vec<T>> {
    Ok(rows.collect::<rusqlite::Result<Vec<T>>>()?)
}
