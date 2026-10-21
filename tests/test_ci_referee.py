"""qk-ci-referee: the referee's own logic, refereed in the lean tier.

CI-only logic that never runs anywhere else is exactly the disease this repo just
burned, so the pieces the workflow calls are loaded and exercised here: the
pytest summary parser (the skip-count discipline both python jobs rest on), the
report assertions of the smoke and abort legs (fed real-shaped payloads,
including the shapes that must fail), the abort leg's fail-closed cache check,
and the workflow's own structural pins (SHA-pinned actions, no privileged
trigger, no masked gate rcs, absence-before-counts ordering, cache key naming the
revisions the tooling asserts).

No torch, no weights, no network: this file runs in the lean tier.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_CI = REPO_ROOT / "tools" / "ci"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load(name: str, path: Path):  # noqa: ANN202
    """Load a tools/ci script by path (they are scripts, not package modules)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pytest_summary = _load("qk_ci_pytest_summary", TOOLS_CI / "pytest_summary.py")
report_checks = _load("report_checks", TOOLS_CI / "report_checks.py")
abort_leg = _load("abort_leg", TOOLS_CI / "abort_leg.py")


# ---------------------------------------------------------------------------
# The skip-count discipline (both python jobs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "passed", "skipped", "failed", "errors"),
    [
        ("80 passed, 16 skipped in 1.31s", 80, 16, 0, 0),
        ("160 passed in 70.62s (0:01:10)", 160, 0, 0, 0),
        ("1 failed, 79 passed, 16 skipped in 4.00s", 79, 16, 1, 0),
        ("2 errors in 0.52s", 0, 0, 0, 2),
        ("16 skipped, 2 warnings in 0.40s", 0, 16, 0, 0),
        ("=== 80 passed, 16 skipped in 1.31s ===", 80, 16, 0, 0),
    ],
)
def test_parse_summary_reads_pytests_own_line(line, passed, skipped, failed, errors):  # noqa: ANN001
    summary = pytest_summary.parse_summary(line)
    assert (summary.passed, summary.skipped, summary.failed, summary.errors) == (
        passed,
        skipped,
        failed,
        errors,
    )


def test_parse_summary_handles_rs_detail_lines_and_ansi():
    log = (
        "\x1b[31mSKIPPED [16] tests/test_kernels_parity.py:45: Rust leg unavailable\x1b[0m\n"
        "\n"
        "80 passed, 16 skipped in 1.74s\n"
    )
    summary = pytest_summary.parse_summary(log)
    assert (summary.passed, summary.skipped) == (80, 16)


@pytest.mark.parametrize(
    "log",
    [
        "",
        "collected 0 items\n",
        "the driver printed a report but pytest never ran\n",
        "80 passed, 16 skipped\n",  # no timing clause: not pytest's summary line
    ],
)
def test_parse_summary_fails_closed_without_a_summary_line(log):  # noqa: ANN001
    with pytest.raises(pytest_summary.SummaryParseError):
        pytest_summary.parse_summary(log)


def test_findings_flag_every_way_a_tier_can_lie():
    ok = pytest_summary.Summary(
        line="80 passed, 16 skipped in 1.0s", passed=80, failed=0, errors=0, skipped=16
    )
    assert pytest_summary.findings(ok, 0, 16) == []

    # The silent-skip disease: a pre-extras suite that looks green.
    lie = pytest_summary.Summary(
        line="62 passed, 16 skipped in 1.0s", passed=62, failed=0, errors=0, skipped=16
    )
    assert any("expected exactly 0" in finding for finding in pytest_summary.findings(lie, 0, 0))

    assert any("did not pass" in f for f in pytest_summary.findings(ok, 1, 16))
    empty = pytest_summary.Summary(
        line="16 skipped in 0.4s", passed=0, failed=0, errors=0, skipped=16
    )
    assert any("no test passed" in finding for finding in pytest_summary.findings(empty, 0, 16))
    broken = pytest_summary.Summary(
        line="1 failed, 79 passed, 16 skipped in 4.0s", passed=79, failed=1, errors=0, skipped=16
    )
    assert any("1 failed" in finding for finding in pytest_summary.findings(broken, 1, 16))


def test_summary_cli_exit_codes(tmp_path):  # noqa: ANN001
    """The workflow calls this as a process; its exit codes are the contract."""
    log = tmp_path / "pytest.log"

    def run(text: str, rc: int, expect: int) -> subprocess.CompletedProcess:
        log.write_text(text)
        return subprocess.run(
            [
                sys.executable,
                str(TOOLS_CI / "pytest_summary.py"),
                "--log",
                str(log),
                "--pytest-rc",
                str(rc),
                "--expect-skipped",
                str(expect),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

    good = run("80 passed, 16 skipped in 1.31s\n", 0, 16)
    assert good.returncode == 0, good.stdout + good.stderr
    assert "skip discipline holds" in good.stdout

    wrong_count = run("62 passed, 16 skipped in 1.31s\n", 0, 0)
    assert wrong_count.returncode == 1
    assert "::error::" in wrong_count.stdout

    unparseable = run("nothing useful here\n", 0, 16)
    assert unparseable.returncode == 2

    missing = subprocess.run(
        [
            sys.executable,
            str(TOOLS_CI / "pytest_summary.py"),
            "--log",
            str(tmp_path / "absent.log"),
            "--pytest-rc",
            "0",
            "--expect-skipped",
            "16",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert missing.returncode == 2


def test_workflow_lean_job_proves_absence_before_believing_skips():
    text = WORKFLOW.read_text()
    lean = _job_block(text, "lean-python")
    absence = lean.index("import torch, transformers, tokenizers")
    parse = lean.index("tools/ci/pytest_summary.py")
    assert absence < parse, "the absence assertion must precede the skip-count parse"
    assert "uv pip install -e . pytest" in lean
    assert re.search(r"uv venv --python 3\.12 \.venv-lean", lean)
    assert "--expect-skipped 16" in lean

    full = _job_block(text, "full-python")
    assert "--expect-skipped 0" in full
    assert "uv sync --extra torch --extra hf --extra kernels" in full


# ---------------------------------------------------------------------------
# The report assertions (smoke + abort leg)
# ---------------------------------------------------------------------------

SMOKE_EXPECT = {
    "model": "/tmp/tiny-qwen3",
    "rank": 4,
    "solver": "svd",
    "scale_dtype": "float32",
    "seq_len": 32,
}


def _healthy_report(**overrides) -> str:  # noqa: ANN003
    """A real-shaped healthy tiny-fixture report (numbers from the recorded run)."""
    report = {
        "model": "/tmp/tiny-qwen3",
        "rank": 4,
        "solver": "svd",
        "scale_dtype": "float32",
        "seq_len": 32,
        "ppl_baseline_bf16": 151949.86,
        "ppl_quant_native": 151949.86,
        "ppl_ratio": 1.0,
        "effective_bpw": 0.5,
        "qpk1_layer_bytes": 123456,
        "probe": {
            "interval": 1,
            "batches": 1,
            "fp_prefix_ppl": 151949.86356532486,
            "history": [
                {
                    "block": "layers.1",
                    "depth": 2,
                    "prefix_ppl": 151949.9,
                    "fp_prefix_ppl": 151949.86356532486,
                    "ratio": 1.0,
                }
            ],
        },
        "probe_abort": None,
        "artifact_exported": True,
        "status": "ok",
        "wall_seconds": 48.2,
    }
    report.update(overrides)
    return json.dumps(report, allow_nan=False)


def _aborted_report(**overrides) -> str:  # noqa: ANN003
    """A real-shaped abort report (Qwen3-0.6B, rank 1: ratio ~35,000 at depth 2)."""
    report = {
        "model": "Qwen/Qwen3-0.6B",
        "rank": 1,
        "solver": "svd",
        "scale_dtype": "float32",
        "seq_len": 64,
        "ppl_baseline_bf16": 83.08454639156159,
        "ppl_quant_native": None,
        "ppl_ratio": None,
        "effective_bpw": None,
        "qpk1_layer_bytes": None,
        "probe": {
            "interval": 1,
            "batches": 1,
            "fp_prefix_ppl": 83.08454639156159,
            "history": [
                {
                    "block": "layers.1",
                    "depth": 2,
                    "prefix_ppl": 2906604.684589879,
                    "fp_prefix_ppl": 83.08454639156159,
                    "ratio": 34983.69806211141,
                }
            ],
        },
        "probe_abort": {
            "block": "layers.1",
            "depth": 2,
            "prefix_ppl": 2906604.684589879,
            "fp_prefix_ppl": 83.08454639156159,
            "ratio": 34983.69806211141,
            "line": 40.0,
        },
        "artifact_exported": False,
        "status": "failed",
        "wall_seconds": 69.6,
    }
    report.update(overrides)
    return json.dumps(report, allow_nan=False)


def test_smoke_contract_holds_on_a_healthy_run():
    assert report_checks.smoke_findings(_healthy_report(), 0, SMOKE_EXPECT) == []


@pytest.mark.parametrize(
    ("payload", "rc", "expected"),
    [
        ("an {'ok': 1} object", 0, "not acceptable"),  # not a report at all
        (None, 1, "exit 1"),  # exit code disagreeing with status "ok"
        ({"status": "banana"}, 0, "no usable verdict status"),
        ({"probe_abort": {"depth": 2}}, 0, "reported a probe abort"),
        ({"artifact_exported": False}, 0, "artifact_exported: true"),
        ({"ppl_quant_native": None}, 0, "end-of-run quantized perplexity"),
        ({"solver": "admm"}, 0, "provenance solver"),
    ],
)
def test_smoke_contract_flags_every_broken_shape(payload, rc, expected):  # noqa: ANN001
    text = (
        _healthy_report()
        if payload is None
        else _healthy_report(**payload)
        if isinstance(payload, dict)
        else payload
    )
    findings = report_checks.smoke_findings(text, rc, SMOKE_EXPECT)
    assert any(expected in finding for finding in findings), (expected, findings)


def test_smoke_and_abort_assertions_reject_non_finite_json_constants():
    """A report carrying NaN/Infinity is not a report this repo accepts."""
    for text in (
        '{"status": "failed", "ppl_ratio": NaN}',
        '{"status": "ok", "ppl_baseline_bf16": Infinity}',
    ):
        assert any(
            "strict parse" in finding
            for finding in report_checks.smoke_findings(text, 0, SMOKE_EXPECT)
        )
        assert any(
            "strict parse" in finding for finding in report_checks.abort_findings(text, 1, 10.0)
        )


def _abort_with(**changes) -> dict:  # noqa: ANN003
    """The recorded abort's ``probe_abort`` block, mutated one field at a time."""
    abort = {
        "block": "layers.1",
        "depth": 2,
        "prefix_ppl": 2906604.684589879,
        "fp_prefix_ppl": 83.08454639156159,
        "ratio": 34983.69806211141,
        "line": 40.0,
    }
    abort.update(changes)
    return abort


def test_abort_contract_holds_on_the_recorded_collapse():
    assert report_checks.abort_findings(_aborted_report(), 1, 69.6) == []


@pytest.mark.parametrize(
    ("overrides", "rc", "elapsed", "expected"),
    [
        ({"status": "ok"}, 0, 69.6, "must report status 'failed'"),
        ({}, 0, 69.6, "exited 0"),
        ({}, 2, 69.6, "!= verdict_exit_code"),
        ({"artifact_exported": True}, 1, 69.6, "artifact_exported must be false"),
        ({"probe_abort": None}, 1, 69.6, "probe_abort is missing"),
        ({"probe_abort": _abort_with(depth=3)}, 1, 69.6, "before block 3"),
        ({"probe_abort": _abort_with(line=41.0)}, 1, 69.6, "shared COLLAPSE_LINE"),
        ({"probe_abort": _abort_with(ratio=1.0)}, 1, 69.6, "not a finite crossing"),
        ({"probe": {"interval": 1, "batches": 1, "history": []}}, 1, 69.6, "no probe history"),
        ({"ppl_quant_native": 12.0}, 1, 69.6, "no end-of-run numerics"),
        ({}, 1, 901.0, "minutes, not tens"),
        ({"wall_seconds": 901.0}, 1, 69.6, "wall_seconds"),
    ],
)
def test_abort_contract_flags_every_broken_shape(overrides, rc, elapsed, expected):  # noqa: ANN001
    findings = report_checks.abort_findings(_aborted_report(**overrides), rc, elapsed)
    assert any(expected in finding for finding in findings), (expected, findings)


def test_abort_leg_cache_check_fails_closed(tmp_path):  # noqa: ANN001
    """A cold or partial cache is a finding, never a silent download."""
    findings = abort_leg.cache_findings(tmp_path)
    assert len(findings) == len(abort_leg.PINNED_REVISIONS)
    assert all("never downloads" in finding for finding in findings)

    for cache_dir, revision in abort_leg.PINNED_REVISIONS.items():
        snapshot = tmp_path / cache_dir / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (tmp_path / cache_dir / "refs").mkdir(parents=True, exist_ok=True)
        (tmp_path / cache_dir / "refs" / "main").write_text(revision)
        for relative in abort_leg.REQUIRED_FILES[cache_dir]:
            path = snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    assert abort_leg.cache_findings(tmp_path) == []

    (
        tmp_path
        / "models--Qwen--Qwen3-0.6B"
        / "snapshots"
        / abort_leg.PINNED_REVISIONS["models--Qwen--Qwen3-0.6B"]
        / "model.safetensors"
    ).unlink()
    assert any("cache incomplete" in finding for finding in abort_leg.cache_findings(tmp_path))


# ---------------------------------------------------------------------------
# The workflow's own structure
# ---------------------------------------------------------------------------


def _job_block(text: str, job: str) -> str:
    """The YAML block of one job, by its 2-space-indented key."""
    match = re.search(rf"^  {re.escape(job)}:\n(.*?)(?=^  [a-z0-9-]+:\n|\Z)", text, re.M | re.S)
    assert match, f"job {job!r} not found in {WORKFLOW}"
    return match.group(1)


def _uncommented(text: str) -> str:
    """The workflow with YAML comment lines dropped (comments explain, they do not run)."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def test_workflow_is_supply_chain_hardened():
    text = _uncommented(WORKFLOW.read_text())
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert len(uses) >= 8, uses  # checkout/setup-uv in every job, cache, toolchain
    for reference in uses:
        assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference), (
            f"{reference} is not pinned by a full commit SHA"
        )
    assert "pull_request_target" not in text
    assert "secrets." not in text
    assert "continue-on-error" not in text
    assert "|| true" not in text
    assert "set +e" not in text
    assert "contents: read" in text
    assert "bash -euo pipefail {0}" in text
    for trigger in ("  pull_request:\n", "  workflow_dispatch:\n", "    branches: [main]\n"):
        assert trigger in text, trigger


def test_workflow_smoke_is_plumbing_only_and_the_abort_leg_is_offline_and_fail_closed():
    text = WORKFLOW.read_text()
    smoke = _job_block(text, "driver-smoke")
    assert "plumbing only" in smoke.lower()
    assert "tools/ci/driver_smoke.py" in smoke
    assert "HF_HUB_OFFLINE" in smoke

    abort = _job_block(text, "abort-leg")
    assert "tools/ci/abort_leg.py" in abort
    assert "fail-on-cache-miss: true" in abort
    assert "HF_HUB_OFFLINE" in abort
    assert "cache/restore@" in abort

    seed = _job_block(text, "seed-weights")
    assert "hf download" in seed
    assert "cache/save@" in seed
    assert "lookup-only: true" in seed


# ---------------------------------------------------------------------------
# The self-seeding weights job (qk-ci-referee delta, 2026-10-27)
#
# The seed job used to be `workflow_dispatch`-only: the no-PR-content property
# was bought with a job-level `if`, and the price was that nobody ever ran it —
# so the abort leg redded fail-closed on every run until a human dispatched a
# seed. The delta moves that property to STEP level: the job is never
# skippable, the abort leg `needs:` it, and only the network steps are gated on
# the event. That is what makes the leg execute for real on every run.
#
# These pins PARSE the workflow (`yaml.safe_load`) and assert on the resulting
# structure instead of matching its text. Measured reason: a text matcher for a
# step's condition has to enumerate YAML's spellings of a key AND its block
# boundaries by hand, and each enumeration leaked — a blank line inside a step
# block ended the scan before a later `if:` (legal YAML, and what a formatter
# emits), a quoted `'if':` key never matched `startswith("if:")`, and `if :`
# with a spaced colon matched neither. A parser closes all three by
# construction: every spelling resolves to the same dict key, blank lines and
# comments are its problem rather than ours, and a condition is compared as a
# VALUE (exactly equal to the gate) rather than matched as text.
# ---------------------------------------------------------------------------

_SEED_FETCH_STEP = "Fetch the pinned snapshots (live network here and only here)"
_SEED_SAVE_STEP = "Save the weights cache"
_SEED_LOOKUP_STEP = "Is this key already seeded?"
_SEED_GATE = "steps.seeded.outputs.cache-hit != 'true' && github.event_name != 'pull_request'"


def _workflow_yaml(text: str) -> dict:
    """The workflow as parsed YAML (comments and blank lines are its problem)."""
    return yaml.safe_load(text)


def _triggers(workflow: dict) -> dict:
    """The `on:` block.

    YAML 1.1 resolves the bare key `on` to the boolean `True`, NOT the string
    `"on"` — read both spellings or the trigger set silently reads empty.
    """
    for key in (True, "on"):
        if key in workflow:
            return workflow[key] or {}
    return {}


def _named_steps(job: dict) -> dict[str, dict]:
    """One job's steps, by name."""
    return {
        step["name"]: step
        for step in job.get("steps") or []
        if isinstance(step, dict) and isinstance(step.get("name"), str)
    }


def _seed_shape_findings(text: str) -> list[str]:
    """Findings for the seed job's never-skippable, step-gated shape.

    The four properties the delta froze, read off the PARSED structure: no
    job-level `if`; a lookup restore with no condition whose `with.lookup-only`
    is `True`; the event+cache gate as the exact `if` VALUE on both network
    steps; and the abort leg's `needs: seed-weights`. A workflow that does not
    parse is itself a finding — an unreadable referee certifies nothing.
    """
    findings: list[str] = []
    try:
        workflow = yaml.safe_load(text)
    except yaml.YAMLError as error:
        return [
            f"the workflow does not parse as YAML, so nothing about it can be "
            f"trusted (that is a finding, never a traceback): {error}"
        ]
    if not isinstance(workflow, dict):
        return ["the workflow does not parse to a mapping"]
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return ["the workflow has no `jobs` mapping"]
    seed = jobs.get("seed-weights")
    if not isinstance(seed, dict):
        return ["the workflow has no `seed-weights` job"]
    if "if" in seed:
        findings.append(
            "the seed job carries a job-level `if:` — a condition-skip there "
            "propagates through the abort leg's `needs:` and turns a real run "
            "into a green no-run; the no-PR-content property belongs on the "
            "fetch/save STEPS, not on the job"
        )
    steps = _named_steps(seed)
    lookup = steps.get(_SEED_LOOKUP_STEP)
    if lookup is None:
        findings.append(
            f"the seed job has no {_SEED_LOOKUP_STEP!r} step — without the "
            f"lookup a hit is not a no-op and every run downloads the snapshots"
        )
    else:
        if "if" in lookup:
            findings.append(
                f"the lookup-only restore step carries a condition "
                f"({lookup['if']!r}) — it must run unconditionally on all three "
                f"triggers, or a hit stops being a no-op and the save reds "
                f"'cache already exists' on every push (the chore-red the "
                f"idempotency gate exists to prevent)"
            )
        lookup_with = lookup.get("with")
        if not isinstance(lookup_with, dict) or lookup_with.get("lookup-only") is not True:
            findings.append(
                f"the lookup step must set `lookup-only: true` in its `with` "
                f"block (got {lookup_with!r}) — without it the restore downloads "
                f"the whole cache on every run before the fetch decision is made"
            )
    for name in (_SEED_FETCH_STEP, _SEED_SAVE_STEP):
        step = steps.get(name)
        if step is None:
            findings.append(f"the seed job has no {name!r} step")
        elif step.get("if") != _SEED_GATE:
            findings.append(
                f"the {name!r} step's `if` is {step.get('if')!r}, not exactly "
                f"{_SEED_GATE!r} — a cache-miss gate alone lets a PR run "
                f"download weights, and the event guard alone re-saves on every hit"
            )
    abort = jobs.get("abort-leg")
    if not isinstance(abort, dict):
        findings.append("the workflow has no `abort-leg` job to check `needs:` on")
    elif abort.get("needs") != "seed-weights":
        findings.append(
            f"the abort leg's `needs` is {abort.get('needs')!r}, not "
            f"'seed-weights' — without the dependency the leg can start before "
            f"the seed's one download and red fail-closed on a cold cache"
        )
    return findings


def test_seed_job_is_never_skippable_and_the_network_is_gated_at_step_level():
    """The delta's binding shape, asserted on the real workflow's parsed structure.

    Positive: the trigger set names `pull_request`, `workflow_dispatch` and
    push-to-main — asserted as an exact key set, so an added trigger (say the
    privileged `pull_request_target`) is a red rather than an unnoticed extra.
    Negative: no job-level condition can skip the seed — and through the leg's
    `needs:` skip the leg too — because the PR guard lives on the fetch and save
    steps, where it cannot propagate.
    """
    text = WORKFLOW.read_text()
    findings = _seed_shape_findings(text)
    assert findings == [], findings
    workflow = _workflow_yaml(text)
    triggers = _triggers(workflow)
    assert set(triggers) == {"pull_request", "push", "workflow_dispatch"}, triggers
    assert triggers["push"]["branches"] == ["main"], triggers["push"]
    assert "pull_request_target" not in _uncommented(text)
    # Re-assert each property directly, so this test names them itself instead
    # of only trusting the findings function's silence.
    jobs = workflow["jobs"]
    seed = jobs["seed-weights"]
    assert "if" not in seed
    steps = _named_steps(seed)
    assert "if" not in steps[_SEED_LOOKUP_STEP]
    assert steps[_SEED_LOOKUP_STEP]["with"]["lookup-only"] is True
    assert steps[_SEED_FETCH_STEP]["if"] == _SEED_GATE
    assert steps[_SEED_SAVE_STEP]["if"] == _SEED_GATE
    assert jobs["abort-leg"]["needs"] == "seed-weights"


def _reinstate_the_job_level_if(text: str) -> str:
    """Bypass: buy the no-PR property back with a job-level skip."""
    return text.replace(
        "  seed-weights:\n",
        "  seed-weights:\n    if: github.event_name != 'pull_request'\n",
        1,
    )


def _drop_the_pr_guard_from_the_fetch(text: str) -> str:
    """Bypass: the fetch goes back to cache-miss-only — a PR run downloads."""
    return text.replace(" && github.event_name != 'pull_request'", "", 1)


def _drop_the_needs_from_the_abort_leg(text: str) -> str:
    """Bypass: the leg can start before the seed and red on a cold cache."""
    return text.replace("    needs: seed-weights\n", "", 1)


def _condition_the_lookup_before_its_uses(text: str) -> str:
    """Bypass: the condition placed first in the lookup step's key list."""
    return text.replace(
        f"      - name: {_SEED_LOOKUP_STEP}\n",
        f"      - name: {_SEED_LOOKUP_STEP}\n        if: github.event_name != 'pull_request'\n",
        1,
    )


def _condition_the_lookup_after_its_lookup_only(text: str) -> str:
    """Bypass: the SAME condition placed last in the step's key list."""
    return text.replace(
        "          lookup-only: true\n",
        "          lookup-only: true\n        if: github.event_name != 'pull_request'\n",
        1,
    )


def _blank_line_then_condition_the_lookup(text: str) -> str:
    """Bypass: a blank line inside the step, then the condition.

    Legal YAML (a blank line between a mapping's keys) and exactly what a
    formatter or a readability edit emits. It terminated the old text
    matcher's hand-rolled scan before it ever reached the `if:`.
    """
    return text.replace(
        f"      - name: {_SEED_LOOKUP_STEP}\n        id: seeded\n",
        f"      - name: {_SEED_LOOKUP_STEP}\n        id: seeded\n\n"
        f"        if: github.event_name != 'pull_request'\n",
        1,
    )


def _quote_the_lookup_condition_key(text: str) -> str:
    """Bypass: `'if':` — valid YAML that no `startswith("if:")` can match."""
    return text.replace(
        f"      - name: {_SEED_LOOKUP_STEP}\n",
        f"      - name: {_SEED_LOOKUP_STEP}\n        'if': github.event_name != 'pull_request'\n",
        1,
    )


def _space_the_lookup_condition_colon(text: str) -> str:
    """Bypass: `if :` — valid YAML (PyYAML reads a plain `if` key), one char wider."""
    return text.replace(
        f"      - name: {_SEED_LOOKUP_STEP}\n",
        f"      - name: {_SEED_LOOKUP_STEP}\n        if : github.event_name != 'pull_request'\n",
        1,
    )


def _drop_the_lookup_only_flag(text: str) -> str:
    """Bypass: `lookup-only: false` — every run downloads before deciding."""
    return text.replace("          lookup-only: true\n", "          lookup-only: false\n", 1)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_reinstate_the_job_level_if, id="job-level-if-reinstated"),
        pytest.param(_drop_the_pr_guard_from_the_fetch, id="pr-guard-dropped-from-fetch"),
        pytest.param(_drop_the_needs_from_the_abort_leg, id="needs-dropped"),
        pytest.param(_condition_the_lookup_before_its_uses, id="lookup-conditional-before-uses"),
        pytest.param(
            _condition_the_lookup_after_its_lookup_only,
            id="lookup-conditional-after-lookup-only",
        ),
        pytest.param(
            _blank_line_then_condition_the_lookup, id="lookup-conditional-after-blank-line"
        ),
        pytest.param(_quote_the_lookup_condition_key, id="lookup-conditional-quoted-if-key"),
        pytest.param(_space_the_lookup_condition_colon, id="lookup-conditional-spaced-colon"),
        pytest.param(_drop_the_lookup_only_flag, id="lookup-only-flipped-to-false"),
    ],
)
def test_seed_shape_pin_rejects_every_bypass(mutate):  # noqa: ANN001
    """Every bypass against the REAL workflow, each required to red.

    Three assertions per row, in this order: the mutation is not a no-op (so a
    drifted file fails the battery loudly instead of going quietly green); the
    poison still PARSES as YAML (a syntax error would red the parser, not the
    pin, and would prove nothing about the checks); and the pin now reports at
    least one finding.
    """
    text = WORKFLOW.read_text()
    mutated = mutate(text)
    assert mutated != text, "the mutation is a no-op: the workflow drifted out from under it"
    try:
        yaml.safe_load(mutated)
    except yaml.YAMLError as error:
        pytest.fail(f"the poison is a YAML syntax error, so a red would prove nothing: {error}")
    assert _seed_shape_findings(mutated), mutated


def test_workflow_cache_key_names_the_revisions_the_abort_tooling_asserts():
    """The key and the asserted revisions cannot drift apart silently."""
    text = WORKFLOW.read_text()
    key = re.search(r"HF_CACHE_KEY:\s*(\S+)", text)
    assert key, "the weights cache key must be defined once, in workflow env"
    for repository, revision in abort_leg.PINNED_REVISIONS.items():
        assert revision in key.group(1), (repository, revision)


def _ref_restore_writes(seed: str) -> list[str]:
    """Every ``printf '%s' "$var" > "<path>/refs/main"`` line in the seed job.

    The branch-ref restore is pinned by SHAPE, not by substring: an ``echo``
    that merely *mentions* ``refs/main`` keeps a substring assertion alive while
    the actual ref write is deleted — the seed then ships a cache that cannot
    resolve a repo id offline, which is the failure this test exists to prevent.
    """
    return re.findall(
        r"""^[ \t]*printf '%s' "\$(\w+)"[ \t]*>[ \t]*"[^"\n]*refs/main"[ \t]*$""",
        seed,
        re.M,
    )


def test_workflow_fetches_weights_by_pinned_revision_and_restores_the_branch_ref():
    """SF-1: the FETCH contract equals the KEY contract, and refs/main is restored.

    Both halves were measured, not assumed: an unpinned fetch couples the
    referee's re-warm to Qwen's and Salesforce's default branches, and a
    revision-pinned fetch writes `snapshots/<sha>/` but NO `refs/` — while the
    driver resolves `--model Qwen/Qwen3-0.6B` offline THROUGH `refs/main`. Pinning
    one without the other breaks the abort leg on a cache that looks complete.
    """
    seed = _job_block(_uncommented(WORKFLOW.read_text()), "seed-weights")
    assert seed.count("hf download") == 2
    revisions = re.findall(r"--revision ([0-9a-f]{40})", seed)
    assert sorted(revisions) == sorted(abort_leg.PINNED_REVISIONS.values()), revisions
    # The restore itself: a WRITE of the pinned sha into <repo>/refs/main, as a
    # shape — and the variable it writes must be the one bound from the pinned
    # pair, so a literal cannot drift away from the fetched revision.
    assert _ref_restore_writes(seed) == ["revision"], (
        "the seed must WRITE refs/main from the pinned revision (bare 40-char "
        f"sha, no newline) — an echo mentioning refs/main is not the write: {seed!r}"
    )
    assert 'revision="${pair##*:}"' in seed
    # Idempotent re-warm: an already-seeded key is a green no-op, and the save is
    # gated on the lookup so a routine re-dispatch cannot fail red as "noise".
    assert "lookup-only: true" in seed
    assert "steps.seeded.outputs.cache-hit == 'true'" in seed
    assert "steps.seeded.outputs.cache-hit != 'true'" in seed


def _drop_the_ref_write(seed: str) -> str:
    """The demonstrated bypass: delete the printf, keep the echo."""
    return re.sub(r"^.*printf '%s' \"\$revision\".*refs/main.*\n", "", seed, flags=re.M)


def _echo_instead_of_printf(seed: str) -> str:
    """`echo` appends a newline; a branch fetch writes a bare sha with none."""
    return seed.replace("printf '%s' \"$revision\"", 'echo "$revision"', 1)


def _redirect_somewhere_else(seed: str) -> str:
    """The echo still says refs/main; the write goes to a different ref."""
    return seed.replace('"hf-cache/$repo_dir/refs/main"', '"hf-cache/$repo_dir/refs/head"', 1)


def _write_a_literal_sha(seed: str) -> str:
    """A hardcoded sha cannot follow the pins when they are bumped."""
    return seed.replace(
        "printf '%s' \"$revision\"",
        "printf '%s' \"c1899de289a04d12100db370d81485cdf75e47ca\"",
        1,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_drop_the_ref_write, id="delete-the-printf-keep-the-echo"),
        pytest.param(_echo_instead_of_printf, id="echo-instead-of-printf-adds-a-newline"),
        pytest.param(_redirect_somewhere_else, id="write-a-different-ref"),
        pytest.param(_write_a_literal_sha, id="literal-sha-instead-of-the-pinned-revision"),
    ],
)
def test_ref_restore_pin_rejects_every_bypass(mutate):  # noqa: ANN001
    """The pin is a shape assertion, so these four must all be rejected.

    Each mutation keeps the literal string "refs/main" somewhere in the job — a
    substring check passes all four — while breaking the actual restore.
    """
    seed = _job_block(_uncommented(WORKFLOW.read_text()), "seed-weights")
    mutated = mutate(seed)
    assert mutated != seed, "the mutation is a no-op: the workflow drifted out from under it"
    assert "refs/main" in mutated, "the mutation must keep the substring alive to be a bypass"
    assert _ref_restore_writes(mutated) != ["revision"], mutated


def test_workflow_never_resolves_dependencies_unlocked():
    """SF-2: all lockfiles are committed, so every sync/build must use them.

    A PR editing `pyproject.toml`/`Cargo.toml` would otherwise re-resolve
    silently and test different dependencies than main under the same green job
    names. The lean leg is the deliberate exception — it floats on purpose.
    """
    text = _uncommented(WORKFLOW.read_text())
    syncs = re.findall(r"^\s*run: .*\buv sync\b.*$", text, re.M)
    assert len(syncs) == 3, syncs
    for line in syncs:
        assert "--locked" in line, line
    cargo_gates = re.findall(r"^\s*run: .*\bcargo (?:test|clippy)\b.*$", text, re.M)
    assert len(cargo_gates) == 2, cargo_gates
    for line in cargo_gates:
        assert "--locked" in line, line
    assert "uv run --locked maturin build" in text

    lean = _job_block(text, "lean-python")
    assert "uv pip install -e . pytest" in lean
    # The referee test file imports yaml and PyYAML is NOT in the product floor,
    # so the lean recipe must install it explicitly: an uninstalled harness
    # dependency would fail the whole lean leg at collection.
    assert "pyyaml" in lean, "the lean recipe must install the referee's harness dependency"
    assert "--locked" not in lean, "the lean leg floats on the dependency floor by design"


def _rust_gate_lines(text: str) -> dict[str, str]:
    """The rust job's `cargo test|clippy|fmt` run lines, keyed by gate.

    Comments are dropped first: the job body explains why the server crate is in
    scope, and an explanation must not be readable as an executed command.
    """
    block = _job_block(_uncommented(text), "rust-kernels")
    gates: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        for gate in ("cargo test", "cargo clippy", "cargo fmt"):
            if stripped.startswith(f"run: {gate}"):
                gates[gate] = stripped
    return gates


def _package_selection_findings(text: str) -> list[str]:
    """What is wrong with the rust job's package selection (empty = correct).

    `crates/quantibbler-server` carries the two purity guards —
    `binary_purity.rs` (no libpython in the served binary) and `cubek_purity.rs`
    (cubek-attention reachable under `gpu`, `cpu-reference` never enabled). A
    `--manifest-path` selection of the kernels crate executes neither, so both
    guards were decoration: the gates must select the packages by name.
    """
    findings: list[str] = []
    gates = _rust_gate_lines(text)
    for gate in ("cargo test", "cargo clippy"):
        line = gates.get(gate)
        if line is None:
            findings.append(f"{gate}: gate missing from the rust job")
            continue
        for package in ("quantibbler-kernels", "quantibbler-server"):
            if f"-p {package}" not in line:
                findings.append(f"{gate} does not select -p {package}: {line}")
        # Selection can also be narrowed while keeping the name in the text:
        # `--manifest-path <kernels>` ignores the rest of the workspace, and
        # `--exclude quantibbler-server` keeps the literal name alive.
        for narrowing in ("--manifest-path", "--exclude"):
            if narrowing in line:
                findings.append(f"{gate} narrows its selection with {narrowing}: {line}")
    fmt = gates.get("cargo fmt", "")
    if "--all" not in fmt or "--check" not in fmt:
        findings.append(f"cargo fmt is not a workspace-wide --check: {fmt}")
    return findings


def test_rust_job_executes_the_crates_carrying_the_purity_guards():
    """A guard that exists but no job runs is decoration.

    Measured before this pin existed: a deliberately failing test appended to
    `crates/quantibbler-server/tests/cubek_purity.rs` was **invisible** to the
    old `--manifest-path`-scoped job (0 mentions in its output) and red (rc=101,
    `test result: FAILED`) under the current selection.
    """
    findings = _package_selection_findings(WORKFLOW.read_text())
    assert findings == [], findings


def _revert_test_gate_to_manifest_path(text: str) -> str:
    return text.replace(
        "run: cargo test --locked -p quantibbler-kernels -p quantibbler-server --all-targets",
        "run: cargo test --locked --manifest-path"
        ' "crates/quantibbler-kernels/Cargo.toml" --all-targets',
        1,
    )


def _drop_server_from_the_test_gate(text: str) -> str:
    return text.replace(
        "run: cargo test --locked -p quantibbler-kernels -p quantibbler-server --all-targets",
        "run: cargo test --locked -p quantibbler-kernels --all-targets",
        1,
    )


def _drop_server_from_the_clippy_gate(text: str) -> str:
    return text.replace(
        "run: cargo clippy --locked -p quantibbler-kernels -p quantibbler-server "
        "--all-targets -- -D warnings",
        "run: cargo clippy --locked -p quantibbler-kernels --all-targets -- -D warnings",
        1,
    )


def _exclude_server_while_naming_it(text: str) -> str:
    """The lexical dodge: the package name stays in the line, excluded anyway."""
    return text.replace(
        "run: cargo test --locked -p quantibbler-kernels -p quantibbler-server --all-targets",
        "run: cargo test --locked --workspace --exclude quantibbler-server --all-targets",
        1,
    )


def _drop_workspace_fmt_check(text: str) -> str:
    return text.replace(
        'cargo fmt --manifest-path "$MANIFEST" --all -- --check', "cargo fmt --check", 1
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_revert_test_gate_to_manifest_path, id="test-gate-reverted-to-manifest-path"),
        pytest.param(_drop_server_from_the_test_gate, id="server-dropped-from-the-test-gate"),
        pytest.param(_drop_server_from_the_clippy_gate, id="server-dropped-from-the-clippy-gate"),
        pytest.param(_exclude_server_while_naming_it, id="server-excluded-while-still-named"),
        pytest.param(_drop_workspace_fmt_check, id="fmt-no-longer-workspace-wide"),
    ],
)
def test_rust_job_selection_pin_rejects_every_bypass(mutate):  # noqa: ANN001
    """Each mutation narrows the selection while staying plausible YAML.

    The first four keep the string `quantibbler-server` alive in the workflow
    (the clippy line, or an `--exclude` argument), so a naive substring check
    passes them all — the pin must read the gate lines themselves.
    """
    text = WORKFLOW.read_text()
    mutated = mutate(text)
    assert mutated != text, "the mutation is a no-op: the workflow drifted out from under it"
    assert "quantibbler-server" in mutated, (
        "the mutation must keep the package name alive to be a bypass"
    )
    assert _package_selection_findings(mutated) != [], (
        "the pin accepted a mutation that narrows the selection"
    )
