# deepagents-durable

Experimental. The Deep Agents middleware stack running on a durable kernel instead of LangGraph's Pregel engine.

- `crates/core`: the Rust kernel. It stores conversations, immutable entries, durable tasks, submissions, and JSON documents with Chord deltas, all written atomically through one commit line. It reads and writes pi-durable's SQLite format (`make interop`). It also builds for `wasm32-unknown-unknown`, where it runs on a JavaScript host's event loop with storage inline and reads and commits synchronous (`cfg(js)`).
- `crates/python`: PyO3 bindings that expose the kernel as asyncio awaitables.
- `crates/wasm`: wasm-bindgen bindings for JavaScript hosts (Node and Cloudflare Durable Objects), with SQLite's files kept by the host. `make wasm` builds them into the `durable-wasm` npm package in `crates/wasm/pkg`; a `durable-wasm-v<version>` tag publishes its tarball as a release asset, which deepagentsjs's `deepagents-durable` depends on.
- `python/deepagents_durable`: the agent runtime built on the kernel.
