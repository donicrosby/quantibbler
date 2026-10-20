#!/usr/bin/env python3
"""Parse pytest's own summary line and assert the tier's expected skip count.

CI must never believe a count it did not parse, and must never believe a count
from an interpreter whose absence assertion it did not run. The lean job's
``import torch, transformers, tokenizers`` step comes first in the workflow; this
script is the second half of that discipline — the numbers are read off pytest's
summary line deterministically, never from a ``grep`` of coloured output and
never by eyeballing a log.

One script, both python jobs (the workflow shell stays thin):

    python tools/ci/pytest_summary.py --log pytest-lean.log --pytest-rc "$rc" \
        --expect-skipped 16
    python tools/ci/pytest_summary.py --log pytest-full.log --pytest-rc "$rc" \
        --expect-skipped 0

Exit 0 only when pytest's own rc is 0, the parsed summary shows zero failures and
zero errors, at least one test passed, and the skip count equals
``--expect-skipped`` exactly. Anything unparseable (a truncated log, a killed
run, a non-pytest file) is an error, not a pass: the job fails closed instead of
reporting a green it cannot substantiate.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_COUNT_KINDS = (
    "passed",
    "failed",
    "error",
    "errors",
    "skipped",
    "xfailed",
    "xpassed",
    "deselected",
    "warning",
    "warnings",
    "rerun",
    "reruns",
)
_COUNT = rf"\d+ (?:{'|'.join(_COUNT_KINDS)})"
_TIME_TAIL = r"in \d+(?:\.\d+)?s(?:\s*\(\d+:\d{2}:\d{2}\))?"
# pytest's last line, with or without `-q` decoration (`==== 80 passed ... ====`).
_SUMMARY_RE = re.compile(rf"^\s*=*\s*{_COUNT}(?:, {_COUNT})* {_TIME_TAIL}\s*=*\s*$")
_NO_TESTS_RE = re.compile(rf"^\s*=*\s*no tests ran {_TIME_TAIL}\s*=*\s*$")
_COUNT_RE = re.compile(rf"(\d+) ({'|'.join(_COUNT_KINDS)})\b")
_KNOWN = ("passed", "failed", "errors", "skipped")


class SummaryParseError(ValueError):
    """The log carries no pytest summary line pytest itself would have written."""


@dataclass(frozen=True)
class Summary:
    """The counts from pytest's summary line."""

    line: str
    passed: int
    failed: int
    errors: int
    skipped: int


def parse_summary(text: str) -> Summary:
    """The last pytest summary line in ``text``, or raise ``SummaryParseError``.

    Scanning from the end (and accepting the decorated `-q` form) keeps this
    robust against the ``-rs`` skip-detail lines that precede the summary.
    """
    for raw in reversed(_ANSI.sub("", text).splitlines()):
        line = raw.strip()
        if not line:
            continue
        if _SUMMARY_RE.match(line) or _NO_TESTS_RE.match(line):
            counts = dict.fromkeys(_KNOWN, 0)
            for number, kind in _COUNT_RE.findall(line):
                if kind in counts:
                    counts[kind] += int(number)
                elif kind in ("error",):
                    counts["errors"] += int(number)
            return Summary(line=line, **counts)
    raise SummaryParseError(
        "no pytest summary line found in the log (truncated output, a killed run, "
        "or a file that is not pytest's output)"
    )


def findings(summary: Summary, pytest_rc: int, expect_skipped: int) -> list[str]:
    """Everything wrong with this tier's run. Empty list == the assertions held."""
    problems: list[str] = []
    if pytest_rc != 0:
        problems.append(f"pytest exited {pytest_rc} — the suite did not pass")
    if summary.failed or summary.errors:
        problems.append(f"summary reports {summary.failed} failed and {summary.errors} errors")
    if summary.passed == 0:
        problems.append("no test passed: an empty suite is a failed referee, not a green one")
    if summary.skipped != expect_skipped:
        problems.append(
            f"skipped {summary.skipped}, expected exactly {expect_skipped} "
            f"(tier skips are documented values, not ignorable noise)"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, required=True, help="captured pytest output")
    parser.add_argument("--pytest-rc", type=int, required=True, help="pytest's own exit code")
    parser.add_argument(
        "--expect-skipped",
        type=int,
        required=True,
        help="the tier's documented skip count (0 for the full-extras job)",
    )
    args = parser.parse_args(argv)

    try:
        text = args.log.read_text()
    except OSError as exc:
        print(f"::error::cannot read the pytest log {args.log}: {exc}")
        return 2
    try:
        summary = parse_summary(text)
    except SummaryParseError as exc:
        print(f"::error::{exc}")
        return 2

    print(
        f"pytest summary: {summary.line}  "
        f"(passed={summary.passed} failed={summary.failed} errors={summary.errors} "
        f"skipped={summary.skipped} rc={args.pytest_rc} expected_skipped={args.expect_skipped})"
    )
    problems = findings(summary, args.pytest_rc, args.expect_skipped)
    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        return 1
    print("skip discipline holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
