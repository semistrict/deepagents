//! Files written by pi-durable read back exactly as pi reads them.
//!
//! `fixtures/pi-session.sqlite` was written by `libs/durable/interop/interop.ts write`
//! with pi-durable itself, and `fixtures/pi-session.json` is pi's own dump of it.
//! Regenerate both with `make pi-fixture`.

use std::path::Path;

use durable_core::Session;
use serde_json::Value;

#[tokio::test]
async fn reads_a_pi_written_session_like_pi_does() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let dir = std::env::temp_dir().join(format!("durable-pi-interop-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("session.sqlite");
    std::fs::copy(fixtures.join("pi-session.sqlite"), &file).unwrap();

    let session = Session::open(Some(file)).await.unwrap();
    let ours = durable_core::inspect::dump(&session).await.unwrap();
    let theirs: Value = serde_json::from_str(&std::fs::read_to_string(fixtures.join("pi-session.json")).unwrap()).unwrap();
    drop(session);
    std::fs::remove_dir_all(&dir).unwrap();

    assert_eq!(ours, theirs);
}
