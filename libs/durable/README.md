# deepagents-durable

Experimental. The Deep Agents middleware stack running on a durable kernel instead of LangGraph's Pregel engine.

- `crates/core`: the Rust kernel. It stores conversations, immutable entries, durable tasks, submissions, and JSON documents with Chord deltas, all written atomically through one commit line. It reads and writes pi-durable's SQLite format (`make interop`).
- `crates/python`: PyO3 bindings that expose the kernel as asyncio awaitables.
- `python/deepagents_durable`: the agent runtime built on the kernel.
