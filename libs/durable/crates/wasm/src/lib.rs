//! The deepagents durable kernel for JavaScript hosts: Node and Cloudflare
//! Durable Objects.
//!
//! The kernel runs on the host's event loop with SQLite inline, so reads and
//! commits are synchronous calls. Records cross as the JSON pi-durable
//! stores. SQLite's files are kept by the host (see `SessionFiles`).
//!
//! Built with `make wasm`; on other targets the crate is empty.

#![cfg(all(target_family = "wasm", target_os = "unknown"))]

mod convert;
mod errors;
mod files;
mod scheduler;
mod session;

pub use scheduler::{Invocation, Scheduler, Step};
pub use session::{Frames, Session, Tx};
