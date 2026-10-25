# Proposal: qk-code-hygiene

## Why

The user's deletion law: if we deviate from something, we delete it — this is
what version control is for. And the prose law: incident narratives do not
live in product source. Both were applied once (2ac0b1b, f4e5d37) and must
be structural from now on, enforced by tests and audit, not by memory.

## What Changes

- Dead code SHALL be deleted: evidence-based cuts (vulture + grep-verified
  callers), deletion-only commits, both lanes green before landing.
- Incident narrative SHALL live in the openspec change dirs; product source
  keeps the durable rule and its measured numbers in 1-3 comment lines.

## Impact

Developer workflow and review gates only. No production behavior changes.
