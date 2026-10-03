//! Cross-checks durable SQLite files against pi-durable.
//!
//! `cargo run --example interop -- dump <file>` prints every record and document
//! as canonical JSON, in the same shape as `libs/durable/interop/interop.ts dump`.
//! `cargo run --example interop -- write <file>` writes a session with this core.

use std::path::PathBuf;

use durable_core::{
    ConversationOwner, DocAddress, DocOptions, Fork, ForkPolicy, Head, History, Id, Outcome, Scope, Session, SubmissionStatus,
    TaskState,
};
use serde_json::{Map, Value, json};

fn content(pairs: Value) -> Map<String, Value> {
    match pairs {
        Value::Object(map) => map,
        _ => unreachable!("content is an object"),
    }
}

/// The scenario `interop.ts write` produces, written with this core.
async fn write(session: &Session) -> durable_core::Result<()> {
    let notes = |conversation: Id| DocAddress {
        kind: "interop.notes".into(),
        scope: Scope::Conversation { conversation_id: conversation },
        key: None,
    };
    let rewindable = DocOptions { version: 1, history: Some(History::Rewindable), fork: Some(ForkPolicy::AsOf) };
    let settings = DocAddress { kind: "interop.settings".into(), scope: Scope::Session, key: None };

    let tx = session.begin();
    let root = tx.create_root()?;
    tx.commit().await?;

    let tx = session.begin();
    let question = json!({"model": [{"role": "user", "content": "What is the capital of France?", "timestamp": 1}]});
    let first = tx.append_entry(root, "pi.user".into(), content(question), None, None)?;
    tx.put_doc(notes(root), rewindable, json!({"text": "Par", "tags": []}))?;
    tx.put_doc(settings.clone(), DocOptions { version: 2, ..DocOptions::default() }, json!({"theme": "light"}))?;
    tx.commit().await?;

    let tx = session.begin();
    tx.append_entry(root, "pi.assistant".into(), content(json!({"data": {"answer": "Paris"}})), None, None)?;
    tx.put_doc(notes(root), rewindable, json!({"text": "Paris", "tags": ["geo", "europe"]}))?;
    tx.commit().await?;

    let tx = session.begin();
    let fork = tx.create_conversation(Some(Fork { conversation_id: root, at: first }), None)?;
    tx.put_doc(notes(fork), rewindable, json!({"text": "Par", "tags": []}))?;
    tx.commit().await?;

    let tx = session.begin();
    let task = tx.create_task(root, "interop.work".into(), 3, json!({"n": 7}), json!({"phase": "start", "n": 7}), None, false)?;
    tx.commit().await?;
    session.reserve(vec!["interop.work".into()]).await?;

    let scratch = DocAddress { kind: "interop.scratch".into(), scope: Scope::Task { task_id: task }, key: None };
    let tx = session.begin();
    tx.create_conversation(None, Some(ConversationOwner { conversation_id: root, task_id: task }))?;
    tx.put_doc(scratch, DocOptions::default(), json!({"lines": ["started"]}))?;
    tx.commit().await?;

    let tx = session.begin();
    tx.set_task_state(task, TaskState::Terminal { outcome: Outcome::Completed { result: json!({"total": 42}) } })?;
    tx.commit().await?;

    let tx = session.begin();
    let submission = tx.create_submission(root, Some("req-1".into()), SubmissionStatus::Queued, content(json!({"type": "input"})))?;
    tx.commit().await?;

    let tx = session.begin();
    let summary = json!({"data": {"summary": "asked about France"}});
    tx.append_entry(fork, "pi.compaction".into(), content(summary), Some(Head::Entry(first)), None)?;
    let settled = durable_core::Submission {
        id: submission,
        conversation_id: root,
        request_id: Some("req-1".into()),
        status: SubmissionStatus::Unanswered,
        content: content(json!({"type": "input", "reason": "aborted"})),
    };
    tx.put_submission(settled)?;
    tx.put_doc(notes(root), rewindable, json!({"text": "Paris", "tags": ["europe"]}))?;
    tx.commit().await?;
    Ok(())
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> durable_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [command, file] = args.as_slice() else { panic!("usage: interop dump|write <file>") };
    let session = Session::open(Some(PathBuf::from(file))).await?;
    match command.as_str() {
        "dump" => println!("{}", serde_json::to_string(&durable_core::inspect::dump(&session).await?)?),
        "write" => write(&session).await?,
        other => panic!("unknown command {other}"),
    }
    Ok(())
}
