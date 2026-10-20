//! Invariant (3) of the cubek adoption (change `qk-flash-attention`, design.md
//! "Substrate decision 2026-10-19"): the attention kernels ARE reachable under
//! the `gpu` feature, and the shipped server never enables cubek's
//! `cpu-reference` — which ships inside `cubek-test-utils`.
//!
//! A link-based test cannot see a feature flag, so this reads the RESOLVED
//! graph with `cargo tree`, package-scoped, never workspace-wide and never
//! `--all-features` (the kernels crate's own test build legitimately carries
//! cubek-attention as the third parity face; resolver 2 keeps dev-dependency
//! features out of the server graph). `--locked` so a lock drift is a failure
//! here rather than a surprise in a release build. Exactly as designed: this
//! guard lands WITH the dependency, so the `cpu-reference` hazard window never
//! opens — it is not deferred to the task that first enables that feature.
//!
//! The pair is self-guarding in both directions:
//! - dropping `cubek-attention`, or moving it out from under `gpu`, reds the
//!   POSITIVE half;
//! - the NEGATIVE half is asserted on a graph that demonstrably contains
//!   `cubek-attention` (checked in the same invocation), so "no
//!   `cpu-reference` anywhere" can never pass by resolving an empty or
//!   attention-free graph;
//! - and it is asserted at the FEATURE level, not only by crate name: the
//!   crate-absence backstop alone would be vacuous here, because
//!   `cubek-test-utils` does not appear in `Cargo.lock` at all until someone
//!   turns `cpu-reference` on.

use std::process::Command;

const ATTENTION_PKG: &str = "cubek-attention";
const TEST_UTILS_PKG: &str = "cubek-test-utils";
const CPU_REFERENCE_FEATURE: &str = "cpu-reference";

/// Resolved graph for `-p <pkg>` with the given feature set. `--format
/// "{p} {f}"` renders each package WITH its enabled features, which is what
/// makes the negative half a live assertion.
fn resolved_tree(pkg: &str, features: &[&str], with_features: bool) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let mut cmd = Command::new("cargo");
    cmd.args(["tree", "-e", "normal,build", "--locked", "-p", pkg]);
    for feature in features {
        cmd.args(["-F", feature]);
    }
    if with_features {
        cmd.args(["--format", "{p} {f}"]);
    }
    let out = cmd
        .current_dir(manifest)
        .output()
        .expect("run cargo tree (is cargo on PATH?)");
    assert!(
        out.status.success(),
        "cargo tree -p {pkg} -F {features:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let tree = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        tree.contains("quantibbler"),
        "cargo tree resolved no usable graph for {pkg} -F {features:?}:\n{tree}"
    );
    tree
}

/// Lines rendering one package in a `--format "{p} {f}"` tree.
fn package_lines(tree: &str, package: &str) -> Vec<String> {
    tree.lines()
        .filter(|line| line.contains(package))
        .map(|line| line.trim().to_string())
        .collect()
}

/// POSITIVE half: the attention face is reachable from the kernels crate under
/// `gpu`, and from the server once the server's own `gpu` feature is on.
#[test]
fn cubek_attention_is_reachable_under_the_gpu_feature() {
    let kernels = resolved_tree("quantibbler-kernels", &["gpu"], false);
    assert!(
        kernels.contains(ATTENTION_PKG),
        "positive half: {ATTENTION_PKG} not reachable from quantibbler-kernels -F gpu:\n{kernels}"
    );

    let server = resolved_tree("quantibbler-server", &["gpu"], false);
    assert!(
        server.contains(ATTENTION_PKG),
        "positive half: the server's gpu graph does not reach {ATTENTION_PKG}:\n{server}"
    );
}

/// NEGATIVE half: `cpu-reference` — and therefore `cubek-test-utils` — is never
/// enabled in the shipped server's graph, in either feature state.
#[test]
fn shipped_server_never_enables_cubek_cpu_reference() {
    for (label, features, expect_attention) in
        [("default", &[][..], false), ("gpu", &["gpu"][..], true)]
    {
        let by_name = resolved_tree("quantibbler-server", features, false);
        assert!(
            !by_name.contains(TEST_UTILS_PKG),
            "negative half ({label}): {TEST_UTILS_PKG} is reachable from quantibbler-server:\n{by_name}"
        );

        let with_features = resolved_tree("quantibbler-server", features, true);
        let lines = package_lines(&with_features, ATTENTION_PKG);
        if expect_attention {
            // Non-vacuity: the very graph the exclusion is asserted on carries
            // the attention crate, WITH a rendered feature list.
            assert!(
                !lines.is_empty(),
                "negative half ({label}) would be vacuous: {ATTENTION_PKG} is not in this graph:\n{with_features}"
            );
            assert!(
                lines.iter().any(|line| line.contains("std")),
                "negative half ({label}) would be vacuous: no feature list rendered:\n{with_features}"
            );
        }
        for line in &lines {
            assert!(
                !line.contains(CPU_REFERENCE_FEATURE),
                "negative half ({label}): {ATTENTION_PKG} is built WITH {CPU_REFERENCE_FEATURE} \
                 in the server graph (that feature ships {TEST_UTILS_PKG}): {line}"
            );
        }
        assert!(
            !with_features.contains(CPU_REFERENCE_FEATURE),
            "negative half ({label}): {CPU_REFERENCE_FEATURE} appears in the server graph:\n{with_features}"
        );
    }
}
