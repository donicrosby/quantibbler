# Code hygiene delta

## ADDED Requirements

### Requirement: dead_code_shall_be_deleted

Deviated-from code SHALL be deleted, never kept around — this is what version control is for. A deletion SHALL be evidence-based: every candidate verified by static callers (vulture) and grep across `src/`, `tools/`, and `tests/` before the cut, false positives (pytest hooks, framework overrides, stub attributes) left alone, and the commit SHALL be deletion-only with both the default and integration lanes green.

#### Scenario: deletion_is_evidence_based

- **WHEN** a code path is deleted
- **THEN** the commit records the caller census that justified the cut and lands as a deletion-only change with both lanes green
- **AND** a sweep that deletes without the census, or keeps fossilized code because it "might be revived", is refused at review — revival is a git operation, not a file operation

### Requirement: incident_narrative_lives_in_change_dirs

Incident narration SHALL live in the openspec change dirs, not in product source. Comments in `src/` and `tools/` SHALL keep the durable rule and its measured numbers in 1-3 lines and point at the change dir that houses the post-mortem; the story itself — run names, version tags, autopsy framing — is exiled there. The hot-lane fence in `tests/test_prose_discipline.py` SHALL be the mechanical enforcement.

#### Scenario: narrative_exile_keeps_the_evidence

- **WHEN** an incident narrative is written into product source
- **THEN** the durable rule stays as a 1-3 line comment with its measured numbers (e.g. "108,691 negative s1 entries" justifies the sign-fold) and the narration moves to the change dir
- **AND** the fence test failing is the fix's trigger, not a reason to widen the fence's token class
