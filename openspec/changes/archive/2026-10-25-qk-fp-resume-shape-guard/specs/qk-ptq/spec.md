# qk-ptq delta: qk-fp-resume-shape-guard

## ADDED Requirements

### Requirement: Checkpoint restore SHALL refuse a mis-shaped state before writing any layer

Checkpoint restore SHALL validate each stored state's arrays against the live
linear's weight and bias shapes before writing any layer. A state whose
Ub/Vb/s1/s2/bias cannot factorize to the live `(dout, din)` weight SHALL raise
naming the block, layer, and stored shapes, leaving the block untouched.

#### Scenario: restore_refuses_mis_shaped_state

- **WHEN** a resumed run applies a checkpointed state whose arrays cannot
  produce the live linear's weight `(dout, din)` — Ub not `(dout, r)`, Vb not
  `(din, r)` with a shared r, s1 not `(dout,)`, s2 not `(din,)`, or a stored
  bias not `(dout,)`
- **THEN** `apply_layer_states` raises a RuntimeError naming the block, the
  layer, the stored shapes, and the live weight shape, refusing to write the
  state

#### Scenario: mis_shaped_refusal_leaves_block_untouched

- **WHEN** a right-key/wrong-shape archive is applied to a multi-layer block
- **THEN** the raise fires before the first `copy_`, and every linear's weight
  in the block is bit-identical to its value before the restore
