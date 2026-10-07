//! The scheduler driving native Rust task handlers.

use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use durable_core::scheduler::BoxFuture;
use durable_core::{HandlerError, Invocation, JoinPolicy, Outcome, Scheduler, Session, TaskHandler, TaskState, Tx};
use serde_json::{Map, Value, json};
use tokio::sync::Notify;

type Result = std::result::Result<(), HandlerError>;

/// A handler from two closures: one for every phase, one for aborts.
struct Handler<R, A> {
    run: R,
    abort: A,
}

impl<R, RF, A, AF> TaskHandler for Handler<R, A>
where
    R: Fn(Invocation) -> RF + Send + Sync + 'static,
    RF: Future<Output = Result> + Send + 'static,
    A: Fn(Invocation) -> AF + Send + Sync + 'static,
    AF: Future<Output = Result> + Send + 'static,
{
    fn run(&self, invocation: Invocation) -> BoxFuture<Result> {
        Box::pin((self.run)(invocation))
    }

    fn abort(&self, invocation: Invocation) -> BoxFuture<Result> {
        Box::pin((self.abort)(invocation))
    }
}

fn handler<R, RF>(run: R) -> Arc<dyn TaskHandler>
where
    R: Fn(Invocation) -> RF + Send + Sync + 'static,
    RF: Future<Output = Result> + Send + 'static,
{
    Arc::new(Handler { run, abort: abort_with(|| ()) })
}

/// An abort handler that records itself and ends the task as aborted.
fn abort_with<F: Fn() + Send + Sync + 'static>(record: F) -> impl Fn(Invocation) -> BoxFuture<Result> + Send + Sync + 'static {
    let record = Arc::new(record);
    move |invocation: Invocation| {
        let record = record.clone();
        Box::pin(async move {
            record();
            let mut step = invocation.step()?;
            step.aborted(Some("stopped".into()));
            step.commit().await?;
            Ok(())
        }) as BoxFuture<Result>
    }
}

async fn start(session: &Session, kind: &str, input: Value, checkpoint: Value) -> i64 {
    let tx = session.begin();
    let root = tx.create_root().unwrap();
    let task = tx.create_task(root, kind.into(), 1, input, checkpoint, None, false).unwrap();
    tx.commit().await.unwrap();
    task
}

fn completed(task: &durable_core::Task) -> Value {
    match &task.state {
        TaskState::Terminal { outcome: Outcome::Completed { result } } => result.clone(),
        other => panic!("task is {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn abort_cancels_running_work_and_runs_handlers_bottom_up() {
    let session = Session::open(None).await.unwrap();
    let scheduler = Scheduler::new(session.clone());
    let order = Arc::new(Mutex::new(Vec::new()));
    let child_started = Arc::new(Notify::new());

    let started = child_started.clone();
    let child_order = order.clone();
    scheduler.register(
        "child",
        Arc::new(Handler {
            run: move |invocation: Invocation| {
                let started = started.clone();
                async move {
                    let mut step = invocation.step()?;
                    step.advance(json!({"phase": "block", "started": true}));
                    step.commit().await?;
                    started.notify_one();
                    invocation.cancelled().await;
                    Err(HandlerError::Cancelled)
                }
            },
            abort: abort_with(move || child_order.lock().unwrap().push("child")),
        }),
    );
    let parent_order = order.clone();
    scheduler.register(
        "parent",
        Arc::new(Handler {
            run: |invocation: Invocation| async move {
                let task = invocation.task();
                let mut step = invocation.step()?;
                let child =
                    step.tx().create_task(task.conversation_id, "child".into(), 1, Value::Null, json!({"phase": "block"}), Some(task.id), false)?;
                step.wait(vec![child], json!({"phase": "never"}), JoinPolicy::AllSettled);
                step.commit().await?;
                Ok(())
            },
            abort: abort_with(move || parent_order.lock().unwrap().push("parent")),
        }),
    );
    let parent = start(&session, "parent", Value::Null, json!({"phase": "spawn"})).await;
    child_started.notified().await;
    let tx = session.begin();
    tx.abort_task(parent).unwrap();
    tx.commit().await.unwrap();

    let settled = session.wait_task(parent).await.unwrap();
    assert_eq!(*order.lock().unwrap(), vec!["child", "parent"]);
    assert_eq!(settled.state, TaskState::Terminal { outcome: Outcome::Aborted { reason: Some("stopped".into()), result: None } });
    scheduler.stop().await;
}

struct Faulty;

impl TaskHandler for Faulty {
    fn run(&self, _: Invocation) -> BoxFuture<Result> {
        Box::pin(async { Err(HandlerError::Failed("cannot continue".into())) })
    }

    fn abort(&self, _: Invocation) -> BoxFuture<Result> {
        Box::pin(async { Ok(()) })
    }

    fn fault(&self, tx: Arc<Tx>, task: durable_core::Task, message: String) -> BoxFuture<Result> {
        Box::pin(async move {
            let mut content = Map::new();
            content.insert("data".into(), json!(message));
            tx.append_entry(task.conversation_id, "fault".into(), content, None, None)?;
            Ok(())
        })
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failing_phase_faults_and_runs_the_fault_hook() {
    let session = Session::open(None).await.unwrap();
    let scheduler = Scheduler::new(session.clone());
    scheduler.register("faulty", Arc::new(Faulty));
    let task = start(&session, "faulty", Value::Null, json!({"phase": "start"})).await;
    let settled = session.wait_task(task).await.unwrap();
    let TaskState::Terminal { outcome: Outcome::Faulted { error } } = settled.state else { panic!("expected a fault") };
    assert_eq!(error.message, "cannot continue");
    let entries = session.entries(settled.conversation_id, None, 10).await.unwrap();
    assert_eq!(entries[0].entry.content["data"], json!("cannot continue"));
    scheduler.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reopened_session_resumes_at_the_last_checkpoint() {
    let dir = std::env::temp_dir().join(format!("durable-scheduler-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file: PathBuf = dir.join("session.sqlite");

    let session = Session::open(Some(file.clone())).await.unwrap();
    let scheduler = Scheduler::new(session.clone());
    let reached = Arc::new(Notify::new());
    let signal = reached.clone();
    scheduler.register(
        "work",
        handler(move |invocation: Invocation| {
            let signal = signal.clone();
            async move {
                let mut step = invocation.step()?;
                step.advance(json!({"phase": "effect", "saved": "before the crash"}));
                step.commit().await?;
                signal.notify_one();
                invocation.cancelled().await;
                Err(HandlerError::Cancelled)
            }
        }),
    );
    let task = start(&session, "work", Value::Null, json!({"phase": "start"})).await;
    reached.notified().await;
    scheduler.stop().await;
    session.close().await;

    let session = Session::open(Some(file)).await.unwrap();
    let scheduler = Scheduler::new(session.clone());
    scheduler.register(
        "work",
        handler(|invocation: Invocation| async move {
            let TaskState::Running { checkpoint } = invocation.task().state else { unreachable!() };
            let mut step = invocation.step()?;
            step.finish(checkpoint["saved"].clone());
            step.commit().await?;
            Ok(())
        }),
    );
    assert_eq!(completed(&session.wait_task(task).await.unwrap()), json!("before the crash"));
    scheduler.stop().await;
    std::fs::remove_dir_all(&dir).unwrap();
}
