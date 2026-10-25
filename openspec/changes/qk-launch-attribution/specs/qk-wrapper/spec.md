# qk-launch-attribution — Spec Delta (qk-wrapper)

## ADDED Requirements

### Requirement: launch_ownership_required

The wrapper SHALL refuse to launch a fire that carries no owner stamp —
`QKB_LAUNCH_OWNER` (who is launching) plus `QKB_LAUNCH_CARD` (the owning kanban
card), the label spellings accepted as the same evidence — before it takes the
container name. An attributed launch SHALL stamp `qkb_launch_owner` and
`qkb_launch_card` beside `qkb_launch_id`, mirror both into the container env,
and name the owner in the announce, `launch.env`, and the trail's run header.

#### Scenario: wrapper_refuses_unattributed_launch

- **WHEN** the wrapper runs with none of the accepted owner spellings in its environment
- **THEN** it exits 2 with an error naming `QKB_LAUNCH_OWNER` and `QKB_LAUNCH_CARD`, and resolves no argv
- **AND** this is the 2026-10-25 `qk-pertarget-v4` incident's defect: an ownerless fire took the production container name, and no record said whose it was

#### Scenario: launch_ownership_label_spellings_accepted

- **WHEN** only the label spellings `qkb_launch_owner` / `qkb_launch_card` are present
- **THEN** the launch proceeds and stamps the contract labels — the incident ruling named the container labels, which a container can only receive as env, so those spellings are one piece of evidence under a second name

#### Scenario: attributed_launch_stamps_owner_labels

- **WHEN** the wrapper runs with an owner and an owning card id
- **THEN** the argv carries `--label qkb_launch_owner=…` and `--label qkb_launch_card=…` beside `--label qkb_launch_id=…`, the container env mirrors all three, and the announce and `launch.env` name the owner and the card
- **AND** the breadcrumb run header carries `owner=… card=…` — the owner-line announcement is machine-written at fire time, not remembered
