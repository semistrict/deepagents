//! End-to-end behavior of the session kernel over real SQLite files.

use std::path::PathBuf;

use durable_core::{Change, ConversationOwner, DocAddress, DocOptions, History, JoinPolicy, Mode, Outcome, OutcomeError, Scope, Session, TaskState};
use serde_json::json;

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
