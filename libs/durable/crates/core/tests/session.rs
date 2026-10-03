//! End-to-end behavior of the session kernel over real SQLite files.

use std::path::PathBuf;

use durable_core::{
    Change, ConversationOwner, DocAddress, DocOptions, Error, Fork, Head, History, JoinPolicy, Mode, Outcome, OutcomeError,
    Scope, Session, SubmissionStatus, TaskFilter, TaskState,
};
use serde_json::{Map, Value, json};

fn data(value: Value) -> Map<String, Value> {
    let mut content = Map::new();
    content.insert("data".into(), value);
    content
}

fn done(result: Value) -> TaskState {
    TaskState::Terminal { outcome: Outcome::Completed { result } }
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("durable-core-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir.join("session.sqlite"))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.0.parent().unwrap());
    }
}

async fn memory() -> Session {
    Session::open(None).await.unwrap()
}

#[tokio::test]
async fn entries_and_context_follow_heads() {
    let session = memory().await;
    let tx = session.begin();
    let root = tx.create_root().unwrap();
    let first = tx.append_entry(root, "msg".into(), data(json!("a")), None, None).unwrap();
    tx.append_entry(root, "msg".into(), data(json!("b")), None, None).unwrap();
    tx.commit().await.unwrap();

    let tx = session.begin();
    let summary = tx.append_entry(root, "summary".into(), data(json!("ab")), Some(Head::Entry(first + 1)), None).unwrap();
    let last = tx.append_entry(root, "msg".into(), data(json!("c")), None, None).unwrap();
    let frame = tx.commit().await.unwrap();

    let ids: Vec<_> = session.context(root, None).await.unwrap().into_iter().map(|stored| stored.entry.id).collect();
    assert_eq!(ids, vec![summary, first + 1, last]);

    // As of the first commit, the summary did not exist yet.
    let before: Vec<_> = session.context(root, Some(frame.seq - 1)).await.unwrap().into_iter().map(|stored| stored.entry.id).collect();
    assert_eq!(before, vec![first, first + 1]);

    let tx = session.begin();
    let reset = tx.append_entry(root, "reset".into(), Map::new(), Some(Head::SelfEntry), None).unwrap();
    tx.commit().await.unwrap();
    let ids: Vec<_> = session.context(root, None).await.unwrap().into_iter().map(|stored| stored.entry.id).collect();
    assert_eq!(ids, vec![reset]);
}

#[tokio::test]
async fn forks_see_parent_entries_up_to_the_fork_point() {
    let session = memory().await;
    let tx = session.begin();
    let root = tx.create_root().unwrap();
    let a = tx.append_entry(root, "msg".into(), data(json!("a")), None, None).unwrap();
    let b = tx.append_entry(root, "msg".into(), data(json!("b")), None, None).unwrap();
    let fork = tx.create_conversation(Some(Fork { conversation_id: root, at: a }), None).unwrap();
    let c = tx.append_entry(fork, "msg".into(), data(json!("c")), None, None).unwrap();
    tx.commit().await.unwrap();

    let ids = |entries: Vec<durable_core::StoredEntry>| entries.into_iter().map(|stored| stored.entry.id).collect::<Vec<_>>();
    assert_eq!(ids(session.entries(fork, None, 100).await.unwrap()), vec![a, c]);
    assert_eq!(ids(session.entries(root, None, 100).await.unwrap()), vec![a, b]);
}

#[tokio::test]
async fn documents_store_deltas_and_rewind() {
    let session = memory().await;
    let tx = session.begin();
    let root = tx.create_root().unwrap();
    tx.commit().await.unwrap();

    let address = DocAddress { kind: "app.live".into(), scope: Scope::Conversation { conversation_id: root }, key: None };
    let options = DocOptions { history: Some(History::Rewindable), ..DocOptions::default() };
    let mut seqs = Vec::new();
    for text in ["Par", "Paris", "Paris is"] {
        let tx = session.begin();
        tx.put_doc(address.clone(), options, json!({"partial": text})).unwrap();
        seqs.push((tx.commit().await.unwrap().clone(), text));
    }
    let (second, _) = &seqs[1];
    let Change::Updated { ops } = &second.docs[0].change else { panic!("expected an update") };
    assert_eq!(serde_json::to_value(ops).unwrap(), json!([["a", ["partial"], "is"]]));

    for (frame, text) in &seqs {
        let value = session.doc(address.clone(), Some(frame.seq)).await.unwrap().unwrap().value;
        assert_eq!(value, json!({"partial": text}));
    }

    let tx = session.begin();
    tx.put_doc(address.clone(), options, json!({"partial": "draft"})).unwrap();
    assert_eq!(tx.doc(address.clone()).await.unwrap(), Some(json!({"partial": "draft"})));
    tx.rollback();
    let value = session.doc(address.clone(), None).await.unwrap().unwrap().value;
    assert_eq!(value, json!({"partial": "Paris is"}));
}

#[tokio::test]
async fn latest_documents_refuse_historical_reads() {
    let session = memory().await;
    let address = DocAddress { kind: "app.counter".into(), scope: Scope::Session, key: Some("a".into()) };
    let tx = session.begin();
    tx.put_doc(address.clone(), DocOptions::default(), json!({"n": 1})).unwrap();
    let frame = tx.commit().await.unwrap();
    assert!(matches!(session.doc(address, Some(frame.seq)).await, Err(Error::Invalid(_))));
}

#[tokio::test]
async fn child_wait_and_held_outcome() {
    let session = memory().await;
    let tx = session.begin();
    let root = tx.create_root().unwrap();
    let parent = tx.create_task(root, "parent".into(), 1, json!({}), json!({"phase": "spawn"}), None, false).unwrap();
    tx.commit().await.unwrap();

    let kinds = vec!["parent".to_string(), "child".to_string()];
    let (task, mode) = session.reserve(kinds.clone()).await.unwrap().unwrap();
    assert_eq!((task.id, mode), (parent, Mode::Run));

    // The parent spawns two children and waits on both.
    let tx = session.begin();
    let a = tx.create_task(root, "child".into(), 1, json!(1), json!({}), Some(parent), false).unwrap();
    let b = tx.create_task(root, "child".into(), 1, json!(2), json!({}), Some(parent), false).unwrap();
    let on = vec![a, b];
    tx.set_task_state(parent, TaskState::Waiting { checkpoint: json!({"phase": "sum"}), on, policy: JoinPolicy::AllSettled }).unwrap();
    tx.commit().await.unwrap();

    for expected in [a, b] {
        let (child, _) = session.reserve(kinds.clone()).await.unwrap().unwrap();
        assert_eq!(child.id, expected);
        let tx = session.begin();
        tx.set_task_state(child.id, done(child.input.clone())).unwrap();
        tx.commit().await.unwrap();
    }

    // Both children are terminal, so the parent is pending at its checkpoint.
    let (resumed, _) = session.reserve(kinds.clone()).await.unwrap().unwrap();
    assert_eq!(resumed.state, TaskState::Running { checkpoint: json!({"phase": "sum"}) });

    let tx = session.begin();
    tx.set_task_state(parent, done(json!(3))).unwrap();
    tx.commit().await.unwrap();
    assert_eq!(session.wait_task(parent).await.unwrap().state, done(json!(3)));
}

#[tokio::test]
async fn abort_cascades_into_owned_conversations_and_runs_bottom_up() {
    let session = memory().await;
    let kinds = vec!["tool".to_string(), "agent".to_string()];
    let tx = session.begin();
    let root = tx.create_root().unwrap();
    let tool = tx.create_task(root, "tool".into(), 1, json!({}), json!({}), None, false).unwrap();
    tx.commit().await.unwrap();
    session.reserve(kinds.clone()).await.unwrap().unwrap();

    // The tool runs a subagent in a conversation it owns.
    let tx = session.begin();
    let child = tx.create_conversation(None, Some(ConversationOwner { conversation_id: root, task_id: tool })).unwrap();
    let agent = tx.create_task(child, "agent".into(), 1, json!({}), json!({}), None, false).unwrap();
    tx.commit().await.unwrap();
    session.reserve(kinds.clone()).await.unwrap().unwrap();

    let tx = session.begin();
    tx.abort_task(tool).unwrap();
    let frame = tx.commit().await.unwrap();
    let marked: Vec<_> = frame.tasks.iter().filter(|task| task.abort_requested).map(|task| task.id).collect();
    assert_eq!(marked, vec![tool, agent]);

    // The scheduler stops both invocations; abort handlers then run bottom-up.
    let tx = session.begin();
    tx.release_task(tool).unwrap();
    tx.release_task(agent).unwrap();
    tx.commit().await.unwrap();
    let (first, mode) = session.reserve(kinds.clone()).await.unwrap().unwrap();
    assert_eq!((first.id, mode), (agent, Mode::Abort));
    assert!(session.reserve(kinds.clone()).await.unwrap().is_none(), "the tool waits for its subagent");

    let tx = session.begin();
    tx.set_task_state(agent, TaskState::Terminal { outcome: Outcome::Aborted { reason: None, result: None } }).unwrap();
    tx.commit().await.unwrap();
    let (second, mode) = session.reserve(kinds.clone()).await.unwrap().unwrap();
    assert_eq!((second.id, mode), (tool, Mode::Abort));
}

#[tokio::test]
async fn reopening_recovers_running_tasks_and_ids() {
    let scratch = Scratch::new("recover");
    let task;
    {
        let session = Session::open(Some(scratch.0.clone())).await.unwrap();
        let tx = session.begin();
        let root = tx.create_root().unwrap();
        task = tx.create_task(root, "work".into(), 1, json!({}), json!({"phase": "effect"}), None, false).unwrap();
        tx.commit().await.unwrap();
        let (reserved, _) = session.reserve(vec!["work".into()]).await.unwrap().unwrap();
        assert!(matches!(reserved.state, TaskState::Running { .. }));
    }
    let session = Session::open(Some(scratch.0.clone())).await.unwrap();
    let recovered = session.task(task).await.unwrap().unwrap();
    assert_eq!(recovered.state, TaskState::Pending { checkpoint: json!({"phase": "effect"}) });
    let tx = session.begin();
    let fresh = tx.create_conversation(None, None).unwrap();
    assert!(fresh > task, "IDs are never reused after reopening");
}

#[tokio::test]
async fn failed_child_aborts_fail_fast_siblings() {
    let session = memory().await;
    let kinds = vec!["checkout".to_string(), "payment".to_string()];
    let tx = session.begin();
    let root = tx.create_root().unwrap();
    let checkout = tx.create_task(root, "checkout".into(), 1, json!({}), json!({}), None, false).unwrap();
    tx.commit().await.unwrap();
    session.reserve(kinds.clone()).await.unwrap();

    let tx = session.begin();
    let declined = tx.create_task(root, "payment".into(), 1, json!({}), json!({}), Some(checkout), false).unwrap();
    let other = tx.create_task(root, "payment".into(), 1, json!({}), json!({}), Some(checkout), false).unwrap();
    let on = vec![declined, other];
    tx.set_task_state(checkout, TaskState::Waiting { checkpoint: json!({"phase": "decide"}), on, policy: JoinPolicy::FailFast }).unwrap();
    tx.commit().await.unwrap();

    session.reserve(kinds.clone()).await.unwrap();
    let tx = session.begin();
    let error = OutcomeError { message: "card declined".into(), detail: None };
    tx.set_task_state(declined, TaskState::Terminal { outcome: Outcome::Failed { error, result: None } }).unwrap();
    let frame = tx.commit().await.unwrap();
    assert!(frame.tasks.iter().any(|task| task.id == other && task.abort_requested));

    // The marked sibling runs its abort handler next.
    let (task, mode) = session.reserve(kinds.clone()).await.unwrap().unwrap();
    assert_eq!((task.id, mode), (other, Mode::Abort));
}

#[tokio::test]
async fn submissions_deduplicate_by_request_and_settle() {
    let session = memory().await;
    let tx = session.begin();
    let root = tx.create_root().unwrap();
    let mut content = Map::new();
    content.insert("type".into(), json!("input"));
    let id = tx.create_submission(root, Some("req-1".into()), SubmissionStatus::Queued, content).unwrap();
    tx.commit().await.unwrap();

    let found = session.submission_by_request(root, "req-1".into()).await.unwrap().unwrap();
    assert_eq!(found.id, id);
    let queued = session.submissions(root, Some(SubmissionStatus::Queued)).await.unwrap();
    assert_eq!(queued.len(), 1);

    let waiter = tokio::spawn({
        let session = session.clone();
        async move { session.wait_submission(id).await.unwrap() }
    });
    let tx = session.begin();
    let mut settled = found;
    settled.status = SubmissionStatus::Unanswered;
    settled.content.insert("reason".into(), json!("aborted"));
    tx.put_submission(settled).unwrap();
    tx.commit().await.unwrap();
    assert_eq!(waiter.await.unwrap().status, SubmissionStatus::Unanswered);
}

#[tokio::test]
async fn table_reads_after_table_writes_are_rejected() {
    let session = memory().await;
    let tx = session.begin();
    let root = tx.create_root().unwrap();
    assert!(matches!(tx.conversation(root).await, Err(Error::ReadAfterWrite)));
}

#[tokio::test]
async fn task_documents_retire_with_their_task() {
    let session = memory().await;
    let tx = session.begin();
    let root = tx.create_root().unwrap();
    let task = tx.create_task(root, "work".into(), 1, json!({}), json!({}), None, false).unwrap();
    tx.commit().await.unwrap();
    session.reserve(vec!["work".into()]).await.unwrap();

    let address = DocAddress { kind: "app.scratch".into(), scope: Scope::Task { task_id: task }, key: None };
    let tx = session.begin();
    tx.put_doc(address.clone(), DocOptions::default(), json!({"x": 1})).unwrap();
    tx.commit().await.unwrap();

    let tx = session.begin();
    tx.set_task_state(task, done(Value::Null)).unwrap();
    let frame = tx.commit().await.unwrap();
    assert!(frame.docs.iter().any(|change| matches!(change.change, Change::Retired)));
    assert_eq!(session.doc(address, None).await.unwrap(), None);
    assert_eq!(session.tasks(TaskFilter { live: true, ..TaskFilter::default() }).await.unwrap(), vec![]);
}

#[tokio::test]
async fn close_ends_frames_and_releases_the_file() {
    let scratch = Scratch::new("close");
    let session = Session::open(Some(scratch.0.clone())).await.unwrap();
    let mut frames = session.subscribe();
    let tx = session.begin();
    tx.create_root().unwrap();
    tx.commit().await.unwrap();
    session.close().await;
    assert_eq!(frames.recv().await.unwrap().seq, 1);
    assert!(frames.recv().await.is_err(), "the stream ends after close");
    assert!(matches!(session.seq().await, Err(Error::Closed)));

    let reopened = Session::open(Some(scratch.0.clone())).await.unwrap();
    assert_eq!(reopened.seq().await.unwrap(), 1);
}

#[tokio::test]
async fn a_session_file_has_one_owner_at_a_time() {
    let dir = std::env::temp_dir().join(format!("durable-lock-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("session.sqlite");
    let owner = Session::open(Some(file.clone())).await.unwrap();
    let second = Session::open(Some(file.clone())).await;
    assert!(matches!(second, Err(durable_core::Error::Locked(_))), "a second opener is refused");
    owner.close().await;
    let reopened = Session::open(Some(file)).await.unwrap();
    reopened.close().await;
    std::fs::remove_dir_all(&dir).unwrap();
}
