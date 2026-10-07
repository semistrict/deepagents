//! Structured-concurrency rules, applied to a commit's candidate records.
//!
//! Every commit ends by running these rules to a fixpoint over the live tasks:
//!
//! 1. Abort flows down: a live owner's cancellation intent marks its ordinary owned work.
//! 2. `failFast` waits abort the rest of `on` after the first non-completed outcome.
//! 3. A satisfied (or abort-marked) wait becomes pending again.
//! 4. A held (`completing`) outcome becomes terminal once no ordinary owned work is live.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::records::{Id, JoinPolicy, Task, TaskState};

/// The tasks a commit can affect: every live task, plus the terminal tasks live waits name.
pub(crate) struct Graph {
    pub tasks: BTreeMap<Id, Task>,
    /// The owning task of each conversation that holds a live task.
    pub conversation_owner: HashMap<Id, Option<Id>>,
}

impl Graph {
    /// Apply the rules until nothing changes. Returns the IDs of tasks it changed.
    pub fn reconcile(&mut self) -> BTreeSet<Id> {
        let mut changed = BTreeSet::new();
        loop {
            let round = self.cascade_aborts() + self.fail_fast() + self.release_waits() + self.finish_holds();
            if round.is_empty() {
                return changed;
            }
            changed.extend(round.0);
        }
    }

    fn live(&self) -> impl Iterator<Item = &Task> {
        self.tasks.values().filter(|task| task.state.live())
    }

    fn owner_of(&self, task: &Task) -> Option<Id> {
        task.owner.or_else(|| self.conversation_owner.get(&task.conversation_id).copied().flatten())
    }

    /// A live owner's abort mark or held non-completed outcome.
    fn cancels(task: &Task) -> bool {
        task.abort_requested || matches!(&task.state, TaskState::Completing { outcome } if !outcome.completed())
    }

    /// Whether a live owner above `task` cancels it. Background tasks without intent are boundaries.
    fn cancelled_above(&self, task: &Task) -> bool {
        let mut owner = self.owner_of(task);
        while let Some(id) = owner {
            let Some(above) = self.tasks.get(&id).filter(|above| above.state.live()) else {
                return false;
            };
            if Self::cancels(above) {
                return true;
            }
            if above.background {
                return false;
            }
            owner = self.owner_of(above);
        }
        false
    }

    /// Whether `owner` has live ordinary owned work: its child tasks, or foreground tasks
    /// in conversations it owns. Deeper work implies a live task at this level.
    pub fn has_owned_work(&self, owner: Id) -> bool {
        self.live().any(|task| {
            task.owner == Some(owner)
                || (task.owner.is_none() && !task.background && self.conversation_owner.get(&task.conversation_id).copied().flatten() == Some(owner))
        })
    }

    fn cascade_aborts(&mut self) -> Round {
        let marked: Vec<Id> =
            self.live().filter(|task| !task.abort_requested && !task.background && self.cancelled_above(task)).map(|task| task.id).collect();
        self.mark(marked)
    }

    fn fail_fast(&mut self) -> Round {
        let mut marked = Vec::new();
        for task in self.live() {
            let TaskState::Waiting { on, policy: JoinPolicy::FailFast, .. } = &task.state else {
                continue;
            };
            let failed = on.iter().any(|id| self.tasks.get(id).and_then(|other| other.state.outcome()).is_some_and(|outcome| !outcome.completed()));
            if failed {
                marked.extend(on.iter().filter(|id| {
                    self.tasks.get(id).is_some_and(|other| {
                        !other.abort_requested
                            && matches!(other.state, TaskState::Pending { .. } | TaskState::Running { .. } | TaskState::Waiting { .. })
                    })
                }));
            }
        }
        self.mark(marked)
    }

    fn release_waits(&mut self) -> Round {
        let released: Vec<Id> = self
            .live()
            .filter(|task| match &task.state {
                TaskState::Waiting { on, .. } => {
                    task.abort_requested || on.iter().all(|id| self.tasks.get(id).is_none_or(|other| !other.state.live()))
                }
                _ => false,
            })
            .map(|task| task.id)
            .collect();
        for id in &released {
            let task = self.tasks.get_mut(id).expect("released task is loaded");
            let TaskState::Waiting { checkpoint, .. } = &task.state else { unreachable!() };
            task.state = TaskState::Pending { checkpoint: checkpoint.clone() };
        }
        Round(released)
    }

    fn finish_holds(&mut self) -> Round {
        let finished: Vec<Id> = self
            .live()
            .filter(|task| matches!(task.state, TaskState::Completing { .. }) && !self.has_owned_work(task.id))
            .map(|task| task.id)
            .collect();
        for id in &finished {
            let task = self.tasks.get_mut(id).expect("finished task is loaded");
            let TaskState::Completing { outcome } = &task.state else { unreachable!() };
            task.state = TaskState::Terminal { outcome: outcome.clone() };
        }
        Round(finished)
    }

    fn mark(&mut self, ids: Vec<Id>) -> Round {
        for id in &ids {
            self.tasks.get_mut(id).expect("marked task is loaded").abort_requested = true;
        }
        Round(ids)
    }
}

/// The task IDs one pass of one rule changed.
struct Round(Vec<Id>);

impl Round {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::ops::Add for Round {
    type Output = Round;

    fn add(mut self, other: Round) -> Round {
        self.0.extend(other.0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::{Outcome, OutcomeError};
    use serde_json::{Value, json};

    fn done() -> Outcome {
        Outcome::Completed { result: Value::Null }
    }

    fn task(id: Id, owner: Option<Id>, state: TaskState) -> Task {
        Task {
            id,
            conversation_id: 1,
            kind: "t".into(),
            version: 1,
            input: Value::Null,
            owner,
            background: false,
            abort_requested: false,
            state,
            memos: None,
        }
    }

    fn running() -> TaskState {
        TaskState::Running { checkpoint: json!({}) }
    }

    fn graph(tasks: Vec<Task>, owners: &[(Id, Id)]) -> Graph {
        let mut conversation_owner: HashMap<Id, Option<Id>> = tasks.iter().map(|task| (task.conversation_id, None)).collect();
        for (conversation, owner) in owners {
            conversation_owner.insert(*conversation, Some(*owner));
        }
        Graph { tasks: tasks.into_iter().map(|task| (task.id, task)).collect(), conversation_owner }
    }

    #[test]
    fn abort_cascades_to_children_and_owned_conversations() {
        let mut parent = task(1, None, running());
        parent.abort_requested = true;
        let child = task(2, Some(1), running());
        let mut subagent = task(3, None, running());
        subagent.conversation_id = 9;
        let mut grandchild = task(4, Some(3), running());
        grandchild.conversation_id = 9;
        let sibling = task(5, None, running());
        let mut graph = graph(vec![parent, child, subagent, grandchild, sibling], &[(9, 1)]);

        let changed = graph.reconcile();

        assert_eq!(changed, BTreeSet::from([2, 3, 4]));
        assert!(!graph.tasks[&5].abort_requested);
    }

    #[test]
    fn background_task_is_a_boundary() {
        let mut parent = task(1, None, running());
        parent.abort_requested = true;
        let mut background = task(2, None, running());
        background.conversation_id = 9;
        background.background = true;
        let mut below = task(3, Some(2), running());
        below.conversation_id = 9;
        let mut graph = graph(vec![parent, background, below], &[(9, 1)]);

        assert!(graph.reconcile().is_empty());
    }

    #[test]
    fn held_outcome_finishes_after_owned_work() {
        let parent = task(1, None, TaskState::Completing { outcome: done() });
        let child = task(2, Some(1), running());
        let mut graph = graph(vec![parent, child], &[]);
        assert!(graph.reconcile().is_empty());

        graph.tasks.get_mut(&2).unwrap().state = TaskState::Terminal { outcome: done() };
        assert_eq!(graph.reconcile(), BTreeSet::from([1]));
        assert_eq!(graph.tasks[&1].state, TaskState::Terminal { outcome: done() });
    }

    #[test]
    fn failed_hold_cancels_owned_work() {
        let failed = Outcome::Failed { error: OutcomeError { message: "boom".into(), detail: None }, result: None };
        let parent = task(1, None, TaskState::Completing { outcome: failed });
        let child = task(2, Some(1), running());
        let mut graph = graph(vec![parent, child], &[]);

        assert_eq!(graph.reconcile(), BTreeSet::from([2]));
        assert!(graph.tasks[&2].abort_requested);
    }

    #[test]
    fn fail_fast_wait_aborts_siblings_then_releases() {
        let waiting = TaskState::Waiting { checkpoint: json!({"phase": "decide"}), on: vec![2, 3], policy: JoinPolicy::FailFast };
        let parent = task(1, None, waiting);
        let failed = task(
            2,
            Some(1),
            TaskState::Terminal { outcome: Outcome::Failed { error: OutcomeError { message: "declined".into(), detail: None }, result: None } },
        );
        let other = task(3, Some(1), running());
        let mut graph = graph(vec![parent, failed, other], &[]);

        assert_eq!(graph.reconcile(), BTreeSet::from([3]));
        assert!(graph.tasks[&3].abort_requested);
        assert!(matches!(graph.tasks[&1].state, TaskState::Waiting { .. }));

        graph.tasks.get_mut(&3).unwrap().state = TaskState::Terminal { outcome: Outcome::Aborted { reason: None, result: None } };
        assert_eq!(graph.reconcile(), BTreeSet::from([1]));
        assert_eq!(graph.tasks[&1].state, TaskState::Pending { checkpoint: json!({"phase": "decide"}) });
    }
}
