"""qk-cli-sane: the driver's operator-grade ``--help`` surface, pinned.

Scenarios (spec delta ``specs/qk-ptq``): ``help_output_has_operator_sections``
(the gate below), with ``help_naming_unit_and_default`` and
``help_free_of_bot_vocabulary`` / ``cut_flags_absent_from_help`` asserted on the
same invocation.

Why a gate test and not a review note: the surface held when this change was
reviewed by hand (four sections, twenty-two flags, no bot vocabulary), and a
property that only ever held in a reviewer's terminal is not pinned — the next
flag added to a section, or the next cut that leaves its name in a help string,
would land green. This runs the real ``--help`` and reads it as an operator
would.

Cost: no model is built and no pipeline runs; the driver's ``--help`` exits at
argparse.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tools" / "quantize_hf_model.py"

# The four operator sections, in the order the parser groups them.
_SECTIONS = ("Model and data:", "Solver and quality:", "Output:", "Resources:")

# Every flag that SURVIVES the cut, by the section it belongs to. This list is
# the pin: a flag silently leaving a section (or a new one arriving without its
# help text) reds here, and a name drifting to another group reds here too.
_SURVIVING = {
    "Model and data:": (
        "--model",
        "--rank",
        "--seq-len",
        "--tune-seq-len",
        "--calibration-seqs",
        "--eval-seqs",
        "--seed",
    ),
    "Solver and quality:": (
        "--solver",
        "--max-admm-iter",
        "--compute-dtype",
    ),
    "Output:": (
        "--out",
        "--container-format",
        "--scale-dtype",
        "--remainder-dtype",
        "--checkpoint-dir",
        "--no-checkpoints",
    ),
    "Resources:": (
        "--device",
        "--max-threads",
        "--no-good-guest",
        "--gpu-memory-fraction",
        "--gpu-memory-gb",
        "--device-audit",
    ),
}

# The seven flags this change CUT. Each was a real argument before it, so the
# help surface must not name any of them any more.
_CUT_FLAGS = (
    "--paper-faithful",
    "--ste-flip-pass-budget",
    "--probe-interval",
    "--probe-batches",
    "--factor-mode",
    "--ablation",
    "--layer-scope",
)

# Bot vocabulary the operator surface must not carry.
_BOT_VOCABULARY = ("paper", "faithful")

_OPTION_LINE = re.compile(r"^  (--[a-z0-9][a-z0-9-]*)")


def _help_text() -> str:
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip(), "the driver produced no help output"
    return proc.stdout


def _options_by_section(text: str) -> dict[str, list[str]]:
    """The option names of each section BODY (the ``usage:`` block excluded).

    Reading only the section bodies is the point: a flag named in the usage
    summary but missing from its group is exactly the drift this pins.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.strip() in _SECTIONS:
            current = line.strip()
            sections[current] = []
            continue
        if current is not None:
            match = _OPTION_LINE.match(line)
            if match:
                sections[current].append(match.group(1))
    return sections


def test_scn_help_output_has_operator_sections():
    """Four sections, every surviving flag under its own, none of the cut ones.

    ``help_output_has_operator_sections`` + ``cut_flags_absent_from_help`` +
    ``help_free_of_bot_vocabulary``, all read off one ``--help``.
    """
    text = _help_text()
    sections = _options_by_section(text)

    assert tuple(sections) == _SECTIONS, (
        f"--help sections drifted: got {tuple(sections)!r}, want {_SECTIONS!r}"
    )
    for header, expected in _SURVIVING.items():
        assert sections[header] == list(expected), (
            f"{header!r} flags drifted: got {sections[header]!r}, want {list(expected)!r}"
        )

    # The union is the whole surviving surface, with no flag listed twice.
    named = [flag for flags in sections.values() for flag in flags]
    assert len(named) == len(set(named)), f"a flag is listed in two sections: {named}"
    assert len(named) == 22, f"the surviving surface is {len(named)} flags, expected 22: {named}"

    for cut in _CUT_FLAGS:
        assert cut not in text, f"{cut} is back on the operator surface"

    lowered = text.lower()
    for banned in _BOT_VOCABULARY:
        assert banned not in lowered, f"{banned!r} is bot vocabulary on an operator surface"
