"""Incident narratives live in the change dirs, not in product source.

Source states the rule and the measurement; the run ledger (run names,
version tags, post-mortem essays) lives in openspec/changes/. This gate
bans the incident-ledger token class from src/ and tools/ so the prose
diet cannot quietly regrow. Legitimate derivation comments carry no run
tags and pass untouched.
"""

import pathlib
import re

PATTERN = re.compile(
    r"r137|run #\d|post-mortem|postmortem|user-directed|lane history|autopsy",
    re.IGNORECASE,
)
ROOTS = (pathlib.Path("src"), pathlib.Path("tools"))


def test_no_incident_narratives_in_product_source():
    hits = []
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if PATTERN.search(line):
                    hits.append(f"{path}:{lineno}: {line.strip()[:90]}")
    assert hits == [], (
        "incident-ledger narration in product source — state the rule and "
        "the measurement, point at the change dir:\n" + "\n".join(hits)
    )
