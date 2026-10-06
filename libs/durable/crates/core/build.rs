//! `cfg(js)`: the kernel runs inside a JavaScript host (`wasm32-unknown-unknown`),
//! on its event loop, with storage called inline instead of on a thread.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(js)");
    let family = std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if family.split(',').any(|family| family == "wasm") && os == "unknown" {
        println!("cargo::rustc-cfg=js");
    }
}
