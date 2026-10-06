//! A canonical JSON dump of everything a session stores.
//!
//! The shape matches `libs/durable/interop/interop.ts dump`, which reads the
//! same file with pi-durable, so the two implementations can be compared.

use serde_json::{Map, Value, json};

use crate::error::Result;
use crate::records::Scope;
use crate::session::Session;
use crate::store::TaskFilter;

const PAGE: usize = 1000;

fn record<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("records serialize")
}

/// Every conversation, visible entry, active context, task, submission, current
/// document, and every committed value of each rewindable document.
pub async fn dump(session: &Session) -> Result<Value> {
    let seq = session.seq().await?;
    let mut conversations = Vec::new();
    let mut after = None;
    loop {
        let page = session.conversations(after, PAGE).await?;
        after = page.last().map(|conversation| conversation.id);
        let done = page.len() < PAGE;
        conversations.extend(page);
        if done {
            break;
        }
    }
    let mut entries = Map::new();
    let mut contexts = Map::new();
    for conversation in &conversations {
        let visible = session.entries(conversation.id, None, usize::MAX).await?;
        let listed = visible.iter().map(|stored| json!({"commitSeq": stored.seq, "record": record(&stored.entry)})).collect();
        entries.insert(conversation.id.to_string(), Value::Array(listed));
        let context = session.context(conversation.id, None).await?;
        contexts.insert(conversation.id.to_string(), context.iter().map(|stored| json!(stored.entry.id)).collect());
    }
    let tasks = session.tasks(TaskFilter::default()).await?;
    let mut submissions = Vec::new();
    for conversation in &conversations {
        submissions.extend(session.submissions(conversation.id, None).await?);
    }
    submissions.sort_by_key(|submission| submission.id);

    let mut scopes = vec![Scope::Session];
    scopes.extend(conversations.iter().map(|conversation| Scope::Conversation { conversation_id: conversation.id }));
    scopes.extend(tasks.iter().map(|task| Scope::Task { task_id: task.id }));
    let (mut documents, mut history) = (Vec::new(), Vec::new());
    for scope in scopes {
        for doc in session.docs(scope, None).await? {
            let stored = session.doc(doc.address(), None).await?.expect("listed document exists");
            documents.push((doc.id, json!({"record": record(&doc), "value": stored.value, "version": stored.version})));
            if doc.current_only() {
                continue;
            }
            for at in doc.created_at..=seq {
                let past = session.doc(doc.address(), Some(at)).await?.expect("alive document exists");
                history.push(json!({"at": at, "id": doc.id, "value": past.value}));
            }
        }
    }
    documents.sort_by_key(|(id, _)| *id);
    Ok(json!({
        "seq": seq,
        "conversations": conversations.iter().map(record).collect::<Vec<_>>(),
        "entries": entries,
        "context": contexts,
        "tasks": tasks.iter().map(record).collect::<Vec<_>>(),
        "submissions": submissions.iter().map(record).collect::<Vec<_>>(),
        "documents": documents.into_iter().map(|(_, doc)| doc).collect::<Vec<_>>(),
        "history": history,
    }))
}
