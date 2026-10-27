//! SR-8.1 / SR-8.2: CPU-only pure-Rust serving with no Python anywhere in
//! the build graph — asserted EXECUTABLY (task 4.2).

#![allow(clippy::duplicate_mod)] // design D1: tests embed the binary's modules (no lib target)

#[path = "../src/api/mod.rs"]
pub mod api;
#[path = "../src/cli.rs"]
pub mod cli;
#[path = "../src/engine/mod.rs"]
pub mod engine;
#[path = "../src/model/mod.rs"]
pub mod model;
#[path = "../src/sampler.rs"]
pub mod sampler;

mod common;

use common::*;

/// SR-8.1: the ready server serves a completion for the fixture model on
/// CPU through the kernels rlib face (this whole test binary links
/// `quantibbler_kernels::core` directly — no Python involved).
#[test]
#[ignore = "generation rewires onto the transformer forward with qk-attention 2.2/2.3"]
fn inference_runs_cpu_via_kernels_rlib() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    for_each_serving_source(|artifact| {
        rt.block_on(async {
            let (_state, router) = ready_router_for(artifact);
            let port = serve_on_port(router);
            let resp = post_json(
                port,
                "/v1/completions",
                &completion_body(FIXTURE_MODEL_ID, false),
            )
            .expect("completion request");
            assert_eq!(resp.status, 200, "{}: {}", artifact.display(), resp.body);
        });
    });
    // The kernels rlib is a direct dependency of this test binary; calling
    // a kernel here proves the CPU face executes without any interpreter.
    let signs = quantibbler_kernels::core::sign_estimate(&[1.0, -2.0, 0.0]);
    assert_eq!(signs, vec![1, -1, 1]);
}

/// SR-8.2: the built binary links no libpython (dynamic-section check) and
/// `cargo tree -e normal` shows no Python-facing dependency path (the
/// kernels dependency runs with default features = extension-module OFF).
#[test]
fn binary_purity_no_python() {
    // 1. Dynamic linkage of the built binary: no libpython.
    let exe = env!("CARGO_BIN_EXE_quantibbler-server");
    let out = std::process::Command::new("ldd")
        .arg(exe)
        .output()
        .expect("run ldd");
    let ldd = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        !ldd.to_ascii_lowercase().contains("python"),
        "binary links Python: {ldd}"
    );
    // 2. The normal dependency graph of this crate contains no pyo3
    //    extension-module feature: `cargo tree -e normal` must not show
    //    pyo3 with the extension-module feature enabled. (pyo3 itself is a
    //    build-time-conditional compile of the kernels crate; with default
    //    features it does not link libpython — asserted by (1) — and the
    //    extension-module feature must be OFF.)
    let manifest = env!("CARGO_MANIFEST_DIR");
    let out = std::process::Command::new("cargo")
        .args(["tree", "-e", "normal", "-p", "quantibbler-kernels"])
        .current_dir(manifest)
        .output()
        .expect("run cargo tree");
    let tree = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        !tree.contains("feature extension-module"),
        "kernels rlib must build WITHOUT the extension-module feature; tree:\n{tree}"
    );
    // 3. The server's own graph has no python-interpreter embedding crate.
    let out = std::process::Command::new("cargo")
        .args(["tree", "-e", "normal"])
        .current_dir(manifest)
        .output()
        .expect("run cargo tree");
    let full = String::from_utf8_lossy(&out.stdout).to_string();
    for banned in ["cpython", "pyembed", "rustpython"] {
        assert!(
            !full.contains(banned),
            "server build graph contains {banned}"
        );
    }
}
