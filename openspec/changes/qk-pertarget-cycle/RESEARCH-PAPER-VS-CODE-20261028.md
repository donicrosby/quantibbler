# Paper-vs-code research arc — formula crosswalk, deviation ledger, ranked falsifiable PPL hypotheses

- **Date:** 2026-10-28 (UTC). **Card:** `t_17d59a6e` (planner). Parent sanity verdict: `t_6ae320a6` → NOT-SANE (v7 bf16+GPU, ratio 18.04×).
- **Object of study:** the in-repo pinned paper `docs/papers/nanoquant-2602.06694v3.md` ([NQ]) — prose, Algorithm 1, Figure 3, Appendix A/B/C/D formulas, the carried Figure 2/3 captions — against this repo's implementing code under `src/quantibbler/ptq/` and `tools/quantize_hf_model.py`.
- **Clean-room hard line (held):** SamsungLabs NanoQuant *source code* is forbidden and was not consulted. Every paper citation below is a `file:line` in the pinned markdown; every code citation is a `file:line` in this repo, verified at trunk `d197a2e`. Nothing in this document is obtainable only from the external source.
- **Run evidence (both legs):** `artifacts/qwen3-0.6B-r512-v7-bf16-gpu-pertarget/quantize_report.json` (sha256 `7f330b5e…`), `artifacts/qwen3-0.6B-r512-admm-paperfaithful-v6b-pertarget/quantize_report.json`, the v7 `solver_probe.jsonl` (196 rows), the v7 `stages[]` (282 events), the v7 per-block checkpoints, and the v7 artifact `layers/**/*.qpk1` (196 streams, parsed for scale magnitudes).
- **Operator audit (2026-10-28, delivered on this card):** a measured paper-vs-code audit was handed down mid-run. Its findings are folded in below and attributed as "operator audit"; where its numbers diverge from the recorded instrument, the divergence is flagged in §8 rather than silently resolved.

---

## 1. Operating point vs the paper's own table (context for every claim below)

| quantity | value | source |
|---|---|---|
| paper Table 2, Qwen3-0.6B, 0.80-bit NanoQuant | **33.79** WT2 PPL | `docs/papers/nanoquant-2602.06694v3.md:1079` (Q3-0.6 column) |
| paper Table 2, Qwen3-0.6B, 16-bit baseline | **12.66** | `:881` |
| paper implied ratio | 33.79 / 12.66 = **2.67** | computed from the two rows |
| our r512 runs, `effective_bpw` **0.804** | v6b **285.78** (14.1×), v7 **364.55** (18.0×) | v6b/v7 `quantize_report.json` `ppl_ratio` |
| our recorded baseline | 20.21 (bf16 pedestal) | v7 `ppl_baseline_bf16` |

The 0.80-bit operating point is **proven reachable** by the pinned table at 33.79, while we ship 364.55 at the same 0.80-bit budget — ~10.8× absolute / ~6.8× ratio off (v7 vs paper's 33.79/2.67; v6b comparator is ~8.5×/5.3×). This is implementation damage, not geometry: rank 512 / 0.80-bit is squarely inside the paper's demonstrated regime. (Operator audit, item 1.)

---

## 2. The paper's pipeline, in its own words (Algorithm 1 anchor)

Algorithm 1 (`:674-825`) is the skeleton every stage below hangs off. Its file lines (verified):

| Alg. line | content | file line |
|---|---|---|
| 2–5 | Phase 1 global calibration: collect `(z_in, z_out)` per linear, `RobustDiag(·; τ, γ)` | `:695,700,705,710` |
| 8 | `for block b = 1..B` | `:725` |
| 9–10 | `X_b ← M̂_<b(X_cal)` (compressed-prefix input); `Y_b ← B^FP_b(X_b)` (teacher output) | `:730,735` |
| 12 | `TuneFP(b, X_b, Y_b; T_pre)` — Step 1 | `:745` |
| 14–17 | `for linear ℓ ∈ b`: `W~ ← D_out W D_in`, `LB-ADMM(W~; r,K,ρ,λ,ε)` — Step 2 | `:755,760,765,770` |
| 19 | `TuneLatentSTE(b, X_b, Y_b; T_post)` — Step 3 | `:780` |
| 20–23 | `sign(U)`, `sign(V)`, `PackBinary` | `:785,790,795,800` |
| 26–27 | packed binaries frozen; `TuneScalesKD(M̂, M, X_cal; T_glob)` — Phase 3 | `:815,820` |

The three-step block loop is §3.2 (`:516-654`); the prose that defines each step's *purpose* is §3.2 Step 1/2/3 (`:522-526`, `:528-531`, `:639-654`). Figure 3's carried caption is `:577`; the Figure 3 *image* (`docs/papers/assets/block_recon.png`, legend: yellow=FP16 weight, green=factorized, flame=tune, scissors=factorize) is the placement tie-breaker used below.

---

## 3. Formula-by-formula crosswalk

Legend per row: **MATCH** (paper form implemented as written), **DEVIATION** (implemented differently; declared or not), **UNVERIFIABLE** (paper names the call but writes no formula), **INFORMED** (paper names the class; the code fills a blank the corpus leaves open).

### 3.1 ADMM objective and penalty/rho handling

| paper | code | verdict |
|---|---|---|
| Eq. 4/27: `min ½‖W−UVᵀ‖²_F + λ/2(‖U‖²+‖V‖²)` s.t. `U=Z_U, V=Z_V` (`:566-574`, `:2785-2796`) | `lb_admm.py:4` (module docstring) states Eq. 4; the augmented-Lagrangian continuous subproblem of Eq. 31 (`:2861`) is what the loop minimizes | **MATCH** on the objective's *data + ridge* structure |
| Eq. 5/32/33 normal equation `(VᵀV+(ρ+λ)I)Uᵀ = VᵀWᵀ + ρ(Z_U−Λ_U)ᵀ` (`:584`, `:2869`, `:2877`) | `lb_admm.py:546-558` solves `(VnᵀVn + shift·I) Yᵀ = VnᵀW_aᵀ + ρ((Z_U·D_s)−(Λ_U·D_s))ᵀ`, then `u = Y/D_s` | **DEVIATION** (see 3.1a) |
| stabilized Cholesky, `O(r³/3)` (`:588`) | `lb_admm.py:140-186` `_cholesky_factor/_cholesky_solve` with relative jitter (`_JITTER=1e-10`, `:68`) | **MATCH** (stabilized-Cholesky; jitter is an implementation detail) |
| `rho` linear penalty scheduler, 400 steps (`App. C :3008`) | `lb_admm.py:532-534` `rho = rho_start + (rho_end−rho_start)·k/denom`; `config.py:223-224` `rho_start=0.0, rho_end=1.0`; `admm_max_iter=400` | **MATCH** on shape; **INFORMED** on endpoints (App. C pins the *linear schedule*, not the endpoints — `config.py:171-181`) |
| early stop at `epsilon` (`App. C`; Algorithm-1 input `ε`) | `lb_admm.py:591-593` relative consensus residual `< admm_epsilon=1e-7` | **MATCH** (relative form, `config.py:226`) |
| dual update Eq. 36 `Λ ← Λ + U − Z` (`:2904`) | `lb_admm.py:584-586` `lu = lu + u − zu`, `lv = lv + v − zv` | **MATCH** |

**3.1a — the normalized-design, scale-relative ridge (DEVIATION, not in the v7 stage_map).** The paper's continuous update regresses `U` against the raw design `V` and shifts the Gram by the *absolute* `(ρ+λ)I` (Eq. 32/33, App. B.3/B.4). The code regresses against the column-normalized design `Vn = V / ‖V‖_{:,cols}` with unknown `Y = U·D_s`, and sets the shift to `rho · mean(diag(VnᵀVn)) + admm_reg` (`lb_admm.py:418-431`, `:546-558`). Effects: (i) the ridge is *scale-relative* (a unit-diagonal design pins `mean(diag)` at 1, so one `rho` schedule serves all layers — the rationale is recorded at `:418-431`); (ii) the effective stabilizer floor is `admm_reg = 3e-2` (`config.py:222`), not the paper's `λ = 1e-6` (`admm_lam`, `config.py:225`). The SPD structure of App. B.4 Lemma 2 is preserved, but the *conditioning* bound (Corollary 2, below) is evaluated on `Vn`, not `V`. This is a deliberate, in-code-labeled deviation, but it is **not** carried in the v7 report's `runtime_deviations=[]` or the `lb_admm` stage_map cell (which mentions only polish) — it belongs on the ledger (§4).

**3.1b — target-norm normalization (DEVIATION, unlisted).** The target is divided by its own Frobenius norm at entry and the factors are re-scaled by `√(target_scale)` each on exit (`lb_admm.py:479-483`, `:607-615`). The comment's own justification (`:467-478`) is that the absolute `rho` schedule was calibrated in the O(1) scale class; this is an informed-implementation choice with no paper line behind it.

**3.1c — SVD warm start (INFORMED).** The paper does not state the U/V initialization (SVID is only the *proxy* update, Eq. 6/35). The code warm-starts U, V from a truncated rank-r SVD of the preconditioned target (`lb_admm.py:298-342`, label A-6). Matches nothing in the paper; recorded as informed.

### 3.2 Hessian-aware preconditioning (Eq. 2–3)

| paper | code | verdict |
|---|---|---|
| Eq. 2 `L(Ŵ) ≈ ‖D~_out(W−Ŵ)D~_in‖²_F`, D from activation + gradient statistics (`:540-544`) | `calibration.py:161-184` `d_out = E[g²]` from a real CE backward (`cotangent_grad_sq_sum`, App. C :3011), `d_in = E[x²]`; per-token 99.9th-pct norm clip before squaring (`calibration.py:115-141`, cited to App. B.1) | **MATCH** (K-FAC diagonal reading) |
| Eq. 3 shrinkage `D~ ← (1−γ)D + γ·mean(D)` (`:551`) | `calibration.py:428-478` `robust_diag`: clip to `tau` (tail-gated q95 default) *before* shrinkage, then `(1−γ)·clip + γ·mean(clip)`; `gamma=0.2` | **MATCH** (γ=0.2 is the paper's Qwen/Llama value, `:556`; clip-before-shrink is App. B.2 "clipping and shrinkage" `:2803`, A-1) |
| preconditioned target `W~ = D~_out W D~_in` (Eq. 26 `:2775`; Algorithm 1 line 15 `:760`) | `_block_stage.py:410-412` folds `sqrt(D)` on each side: `w_tilde = (√d_out·w)·√d_in` | **DEVIATION (adjudicated non-cause)** — see 3.2a |

**3.2a — the sqrt fold.** The code's own comment cites "[NQ] B.1 Eq. 26 folds the preconditioner as a square root" (`_block_stage.py:402-409`), but Eq. 26 (`:2775`) writes the *raw* fold `D~_out W D~_in`. The code folds `√D` on each side and un-folds with `√D` in `magnitude_balance` (`lb_admm.py:674-675`), so factorization and balance stay internally consistent. The operator audit adjudicated the sqrt fold a **verified non-cause** ("preconditioner fold (sqrt, B.1 Eq. 26 ✓)"), consistent with the fact that v6b (fp64) and v7 (bf16) share the fold and show the same damage trajectory. It stays on the ledger as a *textual* deviation from Eq. 26's literal form, but is explicitly **not** a damage hypothesis (§4, non-cause class).

### 3.3 SVD / factorization initialization

| paper | code | verdict |
|---|---|---|
| (no init formula; the factorization is produced by LB-ADMM) | `lb_admm.py:298-342` `_warm_start` (truncated SVD, A-6); `FactorizedLinear.from_weight` (`_factorized.py:108-164`) for the fp/measurement path; `_block_stage.py:421-425` the `solver='svd'` ablation arm | **INFORMED** (paper silent on init; the SVD arm is a measured ablation, not the paper's mechanism) |

### 3.4 Latent magnitude balancing and scale extraction (Eq. 7–9, App. A)

| paper | code | verdict |
|---|---|---|
| Eq. 7 `η = √(‖V̂‖_F / ‖Û‖_F)` (`:609`) | `lb_admm.py:674-683` `u_hat = p_u/du`, `v_hat = p_v/di`, `eta = √(n_v/n_u)` | **MATCH** |
| Eq. 8 `[s1]_i = mean(|η û_i|)`, `[s2]_j = mean(|η⁻¹ v̂_j|)` (`:617`) | `lb_admm.py:685-689` `s1 = mean(|u|, axis=1)`, `s2 = mean(|v|, axis=1)` on the balanced `u = η·u_hat`, `v = v_hat/η` | **MATCH** |
| Eq. 9 `U := ηÛ`, `V := η⁻¹V̂` (`:626-634`) | `lb_admm.py:682-683` | **MATCH** |
| sign(0)=+1 (App. B.5 / container convention) | `lb_admm.py:685-686` `where(u>=0, +1, −1)`; `_factorized.py:306-307`; `check_layer_magnitude` `:365-368` | **MATCH** (the +1-for-zero convention is the repo-wide one) |
| *(nothing)* — Eq. 8 has no global scalar refit | `lb_admm.py:691-707` least-squares `alpha = <W, M>/‖M‖²` folded into `s1` | **DEVIATION (this is F2, the top-ranked scale-shrink suspect)** — see 3.4a |

**3.4a — the alpha refit (a.k.a. "global scale shrink").** The paper's Eq. 8 scale extraction stops at per-row mean-abs of the balanced latents; it never fits a global scalar. The code adds a scalar refit of the *binary* product against the original weight (`lb_admm.py:644-649` documents it as "informed implementation choice; Phase 3 recalibrates afterwards"). The scalar is the orthogonal projection coefficient, so **by construction** the shipped reconstruction's norm is `‖Wq‖/‖W‖ = cos(W, M)` where `M = (s1⊙Ub)(s2⊙Vb)ᵀ` — a *systematic* ~0.72× attenuation when cos ≈ 0.72. The operator audit's item 3 measured exactly this: "normratio ≈ cos on every layer" (per-linear cos medians q .79 / k .84 / v .81 / o .76 / gate .72 / up .69 / down .68). The attenuation is an *identity*, not a coincidence: a 1-D scalar refit onto `span(M)` always yields `‖Wq‖ = ‖W‖·cos(W,M)`. With the downstream gain-correcting stages inert (F1, §7 H1), nothing restores the lost magnitude.

### 3.5 SVID proxy update (Eq. 6 / Eq. 35)

| paper | code | verdict |
|---|---|---|
| Eq. 6 `Z_U ← SVID(P_U)` ("optimal rank-1 approximation that preserves the sign structure", `:592`; App. B.5 `:2984-2990`) | `lb_admm.py:243-295` `svid_proxy`: with `S=sign(P)` the objective `min‖P − S⊙(ghᵀ)‖_F == min‖|P| − ghᵀ‖_F`, solved as the top rank-1 of `|P|` via power iteration (`_top_rank1`, `:189-240`), output `Z = S⊙(g hᵀ)` | **MATCH** (the paper writes no SVID formula; the code's two-sided optimum is the standard sign-value rank-1 reading, and the derivation is written out at `:247-269`; the 100-step cap + `1e-14` tolerance are implementation bounds, `:189-240`) |

### 3.6 TuneFP (Step 1) — placement and objective

**Placement (MATCH after the Figure-3 ruling).** Algorithm 1 places `TuneFP` once per block before the factorization loop (`:745`, schematic line 12). The §3.2 Step 1 prose (`:524-526`) and Figure 3 decide otherwise: the figure's left panel shows the FP remainder (`O,G,U,D` yellow) under flame icons (tuning) while `Q,K,V` are already factorized (green), the middle panel shows the target `O` under scissors (factorization), the right panel shows `O` under flame (refinement) — a *per-target-linear* cycle, not a once-per-block pass. The shipped code implements this: `_block_stage.py:354-365` fires TuneFP when `idx >= 1`, tuning `quantizable_linears[idx:]` (the still-FP remainder incl. the target), and target 1 is factorized first with no preceding TuneFP (the vacuous pass the figure's placement rules out). The per-target-cycle ruling and its detector pins are the subject of this change dir's `design.md` §1.

**Objective (UNVERIFIABLE / INFORMED — the corpus names the form, not the weights).** The paper writes, for Step 1, only the purpose ("tune the full-precision weights … to minimize the error introduced by the quantization of preceding blocks, as well as previously factorized layers in the current block", `:524-526`) and, for the whole block-reconstruction phase, "a weighted MSE function" (`App. C :3012`). It never writes the TuneFP loss. §5 below derives the objective from the corpus and states what is and is not pin-able; the code's shipped objective is `loss = (w[None,:]·(out − y_b[idx])²).mean()` (`_tuning.py:310`), weight `w` = the terminal linear's (`down_proj`) robust-diagonal `d_out`, unit-mean-normalized (`_stats.py:167-186`, `:217-236`).

### 3.7 TuneLatentSTE (Step 3) — objective, gradient form, scope

| paper | code | verdict |
|---|---|---|
| Eq. 10 `min ‖B(X_in) − B̂(X_in; sign(U), sign(V), s1, s2)‖²_F` (`:649`) | `_ste.py:226-234` block-output weighted MSE against `y_b` (same weight construction as TuneFP); STE passthrough `sign(u) + u − u.detach()` under `torch.is_grad_enabled()`, literal `sign` under no_grad (`_factorized.py:50-66`) | **MATCH** on the objective form and STE gradient (d sign(u)/du ≡ 1 under the passthrough, `_factorized.py:37-48`); **DEVIATION** on scope (3.7a) and on the optimizer (3.7b) |
| tune `U, V, s1, s2` (`:643`, `:649`) | `_ste.py:72` params = `(u_latent, v_latent, s1, s2)` for all `facs` | **MATCH** on the parameter set |
| `TuneLatentSTE(b, X_b, Y_b; T_post)` once per block (Alg. line 19 `:780`); §3.2 Step 3 prose "the current target linear layer" (`:641`) | one *joint* pass over the whole factorized set, ONE optimizer, ONE block-output loss (`_block_stage.py:585-613`, `_ste.py:42-51`) | **DEVIATION (declared; this is F3)** — see 3.7a |

**3.7a — joint-block scope vs per-target-linear (declared in the v7 stage_map).** The code runs one joint TuneLatentSTE over the block's whole factorized set, consuming the App. C factorized-stage budget (8 epochs × n/bs, `config.py:228`) **once** for the block (`_block_stage.py:299-310`, `:585-599`), whereas Algorithm 1 line 19's prose and the Figure 3 cycle name a per-target-linear refinement, and the App. C budget is stated per-stage without per-layer scoping (`:3007`). The stage_map's `ste` cell declares this. The operator audit's item 5 ranks it F3 ("refinement budget collapse … ~7× tuning-compute cut vs the paper's cycle").

**3.7b — per-tensor median-relative lr (DEVIATION, unlisted; this is F1).** App. C prescribes *absolute* lrs for the factorized stage: `1e-5` (`:3005`). The code instead gives each of the four parameter tensors its own group with `lr = stage_latent.lr · median(|p|)` (`_ste.py:119-130`). The same per-tensor scheme appears in Phase-3 KD (`scale_calib.py:148-157`). This is the polish-era pathological-scale fix (`_ste.py:119-129` comment), but in the *balanced* class of the v7 artifact (`|s| ≈ 2e-2..1.2e-1`, §6) it shrinks effective steps by `median(|p|)` = 20–100× below the App. C values the paper calibrated. Combined with bf16 storage (§6), it makes the scale parameters structurally incapable of moving (measured: `s1_rel_move == s2_rel_move == 0.0` exactly on all 196 rows; §7 H1).

### 3.8 Sign extraction and packing (Algorithm 1 lines 20–23)

| paper | code | verdict |
|---|---|---|
| `U_±1 ← sign(U)`, `V_±1 ← sign(V)` (`:790`); pack into integers (Figure 2 caption: bits `−1→0, +1→1`, `:514`) | `extract_and_pack` (`_factorized.py:274-318`): `sign` with sign(0)=+1, `pack_bits`, then round-trip `unpack_bits` parity assert; s1/s2/bias copied in `_block_stage.py:676-680` | **MATCH** (single latent-to-bit site, D-5; parity-gated; the exact byte layout is container-specific, not a paper claim) |

### 3.9 TuneScalesKD (Phase 3, Eq. 11)

| paper | code | verdict |
|---|---|---|
| Eq. 11 `min KL(softmax(z_M/T) ‖ softmax(z_M̂/T))` (`:664`); binaries frozen (`:659-672`, Algorithm 1 line 26 `:815`) | `scale_calib.py:65-70` `_kl_with_logits` token-averaged KL in nats; `:210-239` the frozen `Ub/Vb` are built once and applied through a patched forward, gradients reach only `s1/s2` (`:146-147`) | **MATCH** (T is `config.py:204` `temperature=1.0`, A-7 — the paper states the objective, not T) |
| App. C global lr `1e-6`, bs 1, 8 epochs, cosine (`:3006-3007`) | `scale_calib.py:245-279` bs=1, 8 epochs, cosine; but lr is per-tensor `stage_global.lr · median(|p|)` (`:148-157`) | **DEVIATION** on lr form (F1, same as 3.7b) |
| *(nothing)* | trust-region clip of each scale to `[0.5×, 2×]` its pre-stage value, plus non-finite restore (`scale_calib.py:284-315`) | **DEVIATION (unlisted; armor)** — an extra-paper safety mechanism, likely benign but worth ledger-ing |
| *(nothing)* | disk-backed teacher-logit cache (`scale_calib.py:159-198`) | harness detail, not a method deviation |

### 3.10 Eval protocol

| paper | code | verdict |
|---|---|---|
| WT2 PPL via `lm_eval=0.4.9` (`App. C :3010`), calibration 128 × seq 2048, seed 0 (`:3009`) | local fixed-window blockwise WT2 evaluator: `tools/quantize_hf_model.py:318-340` (per-row CE, mean-of-means → exp), `:367-397` (seeded block shuffle, `picked_index`; `selection_fingerprint` at `:1478`) | **DEVIATION (documented, non-table-comparable)** — the v7 report's own `evaluation` block sets `paper_table_comparable: false` with the coverage caveat; this is a same-window baseline/quantized ratio, not a paper-table row. Both baseline and quantized use the identical window/fingerprint (`29edf9820bb649b3`), so the *ratio* is honest. |

---

## 4. Deviation ledger

Convention: every row cites the paper location that pins the *un*deviated form, the code location that deviates, whether the v7 report declares it, and a damage-plausibility note ("could THIS yield 18× at bf16 while parity cells stay clean?").

### 4.1 Declared in the v7 report's `stage_map` (verbatim from the run)

| # | stage | deviation | paper cite | code cite | damage plausibility |
|---|---|---|---|---|---|
| D1 | `tune_fp` | placement read from Figure 3 as a per-target-linear cycle; internal loss formula an informed assumption | Fig. 3 (`:577` + asset), Step 1 prose `:524-526`, App. C `:3012` | `_block_stage.py:354-365`; `_tuning.py:310` | placement: **exonerated** (it is the *corrected* reading; the once-per-block reading was the falsified one). objective: **live** — see H4 |
| D2 | `lb_admm` | greedy bit-flip polish is extra-paper; round count 0 on this ablation | (paper has no polish) | `_block_stage.py:282,461-470`; `bit_polish.py` | **exonerated** (0 rounds in v7; polish off, `fidelity_conditions.polish_off=true`) |
| D3 | `ste` | one joint pass per block vs Algorithm 1 line 19's per-target prose; App. C budget consumed once per block | Alg. 1 line 19 `:780`; §3.2 Step 3 `:641`; App. C `:3007` | `_block_stage.py:585-613`; `_ste.py:42-51` | **live — F3** (budget cut ~7×; joint scope trades one linear's residual against another under a bf16 loss landscape) |
| D4 | `tune_scales_kd` | packed binaries frozen during scale tuning (verified semantics) | Algorithm 1 line 26 `:815`; §3.3 `:659-672` | `scale_calib.py:210-239` | **exonerated** (freezing is *correct*; the suspect is that the scale updates themselves are inert — H1/H6) |

### 4.2 Newly discovered (not in the v7 `runtime_deviations` or stage cells)

| # | deviation | paper cite | code cite | damage plausibility |
|---|---|---|---|---|
| N1 | **per-tensor median-relative lr** in STE (and KD) vs App. C's absolute lrs | App. C `:3005-3006` | `_ste.py:119-130`; `scale_calib.py:148-157` | **live — F1, highest-ranked.** At bf16 storage the effective step `1e-5·median(|p|)` is 20–100× under App. C and *rounds to zero* on every scale entry (measured `s1/s2_rel_move == 0.0`, §6) |
| N2 | **global scale shrink**: LS scalar refit `alpha` folded into `s1` | Eq. 8 `:617` (no global scalar) | `lb_admm.py:691-707` | **live — F2.** `‖Wq‖/‖W‖ ≡ cos` by identity → ~0.72× network-wide attenuation; inert KD (N1) never restores it |
| N3 | **normalized-design + scale-relative ridge** (ridge = `rho·mean(diag(VnᵀVn)) + admm_reg`, not `(ρ+λ)I`) | Eq. 32/33 `:2869,2877`; Cor. 2 `:2945` | `lb_admm.py:418-431,546-570` | **moderate** — changes the conditioning regime (κ ~1.7e4 at ρ=0, §6); `converged=False` 196/196 is consistent but per-linear cos 0.81 is healthy, so not the 18× driver alone |
| N4 | **target-norm normalization** (`W~/‖W~‖`, exit `√scale` rescale) | (no paper line) | `lb_admm.py:479-483,607-615` | low (scale-equivariance argument; internal consistency holds) |
| N5 | **SVD warm start** of U/V | (paper silent) | `lb_admm.py:298-342` | low |
| N6 | **KD trust-region clip** `[0.5×, 2×]` + non-finite restore | (no paper line) | `scale_calib.py:284-315` | low (armor; but note it *caps* any KD scale correction at 2× — relevant only if KD were otherwise live) |
| N7 | **sqrt-fold of the preconditioner** vs Eq. 26's raw `D_out W D_in` | Eq. 26 `:2775`; Algorithm 1 line 15 `:760` | `_block_stage.py:410-412`; `lb_admm.py:674-675` | **adjudicated non-cause** (operator audit; same fold in both fp64 and bf16 legs, same trajectory) |

### 4.3 Verified NON-causes (operator audit — recorded so nobody re-chases them)

dtype regime (v7 bf16 ≈ v6b fp64, same trajectory shape); preconditioner fold (N7); statistics collection (99.9th-pct token-norm clip + CE-backward cotangent + Eq. 3 shrinkage, all literal); calibration hyperparameters (128×2048 seed 0, ADMM 400 linear ρ, App. C stage budgets — literal); eval parity (same windows both legs); 196/196 zero biases (attention_bias=false, legit zero).

---

## 5. TuneFP objective — derivation from the corpus (resolving the "informed assumption")

**What the corpus pins (verbatim, cited):**

1. Step 1's purpose: "We tune the full-precision weights of the current block to minimize the error introduced by the quantization of preceding blocks, as well as previously factorized layers in the current block." (`:524-526`).
2. The data available to the call: `TuneFP(b, X_b, Y_b; T_pre)` with `X_b ← M̂_<b(X_cal)` and `Y_b ← B^FP_b(X_b)` (Algorithm 1 lines 9–12, `:730,735,745`). The error from *preceding blocks* is carried in `X_b`; the error from *factorized in-block layers* is whatever makes the current block's output differ from `Y_b`.
3. The loss *form* for the block-reconstruction phase: "a weighted MSE function" (`App. C :3012`). No weight formula is written anywhere in the paper.
4. The only per-feature weight vectors the paper constructs are `D~_in`/`D~_out` of Eq. 2 (activation- and gradient-statistic diagonals, `:540-544`).

**Derivation (each step tagged pin vs construction):**

- **Step A (pin):** TuneFP minimizes a block-output reconstruction error against the pinned teacher `Y_b`. The block output is `B̂_b(X_b)` — the current block applied to the compressed-prefix input. This is forced by (1)+(2): the error to absorb lives in the difference between the current block's output and the pristine teacher's output on the same input. (The same quantity Eq. 10 uses for Step 3, `:649`.)
- **Step B (pin):** the loss is a *weighted* MSE (`:3012`), i.e. `Σ_f w_f · (B̂_b(X_b) − Y_b)²_f` for per-output-feature weights `w_f` — *some* `w`. The paper does not say which.
- **Step C (construction, the only gap):** the weight vector. Eq. 2's `D_out` is the gradient-statistic diagonal of a linear layer; the block's *terminal* linear (`down_proj`) is the one whose `D_out` has the same feature space as the block output, so `w = D_out(down_proj)` is the unique corpus-consistent per-output-feature weight. This is what `_stats.py:167-186` resolves, and it is labeled an informed assumption end-to-end (the `TUNE_FP_LOSS_MARKER`, `_tuning.py:38`, declares "weight source: informed_assumption").
- **Step D (construction, numerical):** the raw `D_out = E[g²]` sits at ~1e-11..1e-13 (measured, §6), squashing the loss ~1e10×; unit-mean normalization (`_stats.py:217-236`, `LOSS_WEIGHT_MEAN_NORMALIZATION`) rescales it to the residual's scale, preserving per-feature structure. Uniform scaling of `w` is a re-gauging; the *structure* is the content.

**What this does and does not conclude.** The paper's math implies a weighted block-output MSE against `Y_b`; it does **not** imply *which* weight. The shipped weight (terminal `D_out`, unit-mean) is the most defensible corpus-consistent choice, but it is a choice. The two candidate objectives that must be empirically discriminated — uniform MSE vs terminal-`D_out`-weighted MSE — differ in exactly one free vector, so a one-block gradient comparison settles whether the shipped weight changes the fixed point at all (probe H4, §7). The residual fact that *matters* for the ledger is narrower: the paper names "weighted MSE" and Eq. 10 writes an *unweighted* Frobenius norm for Step 3, so the paper itself is not internally committed to the terminal-weighting of the tuning losses — the weighting is ours.

---

## 6. bf16-specific numerics audit

**Precision constants.** bf16 mantissa = 7 bits + implicit = 8 significant bits; relative resolution `2^-8 = 3.906e-3`, `torch.finfo(bfloat16).eps = 7.8125e-3`; ULP of a value `x` ≈ `x·2^-8`. Adam `eps = 1e-8`. The resolution-derived guard floor is `optimizer_eps · finfo(bfloat16).eps = 1e-8 · 2^-7 = 7.8125e-11` (`_stats.py:261-265`; the record's `floor_dtype_rel_resolution=0.0078125` confirms the `2^-7` reading), matching the v7 record's `gradient_floor = 7.8125e-11` on every row.

**Where the paper's math silently assumes more precision than bf16 delivers, per stage:**

| stage | paper's implicit assumption | bf16 reality | measured consequence (v7) |
|---|---|---|---|
| Phase 1 stats | D from E[g²] needs a real CE backward (App. C :3011) | accumulated at fp32 (`calibration.py:107-108` names bf16 sums as "the mantissa loss the accumulation contract exists to prevent") | none — correct by construction |
| LB-ADMM | stable Cholesky, κ bounded by `1 + ‖V‖²/(ρ+λ)` (Cor. 2, `:2945`) | iterates at fp32 (`adt`), target rounded to bf16 at entry (`lb_admm.py:459,493`), latents/proxies rounded to bf16 at the return boundary (`:610-615`) | `converged=False` 196/196; `min_eigs ≈ 0.16`; κ(H) at ρ=0 is `1 + 512/0.03 ≈ 1.7e4`, at ρ=1 `1 + 512/1.03 ≈ 5e2` (bounds from Cor. 2 with `σ²_max ≤ ‖Vn‖²_F = r = 512`) — a ~4-digit solve at the start, ~2.5 digits at the end, *but* per-linear cos_raw 0.81, so the solve is weak-but-usable, not the 18× driver |
| magnitude balance | Eq. 7/8 on f64-equivalent proxies | balances on bf16-rounded proxies at fp32 (`lb_admm.py:662-671`) | scales are balanced in the v7 artifact (median `|s1|/|s2| = 0.896`, range 0.48–2.02, §6 data) — balance is fine |
| TuneFP / STE | Adam with App. C absolute lr `1e-4`/`1e-5` (`:3004-3005`) | params stored bf16 (`floor_param_dtype = bfloat16` on 168/28 rows); Adam moments also bf16; step `lr·unit` must exceed `ulp(x) ≈ x·2^-8` to move `x` | TuneFP (lr 1e-4) moves weights 2.3–16% (live). STE (lr 1e-5, per-tensor shrunk) moves latents `u_rel_move ≈ 1.4e-5` (sub-resolution → declared inert 196/196) and scales **exactly 0.0** on all 196 rows |
| Phase 3 KD | Eq. 11 with App. C lr `1e-6` | per-tensor lr `1e-6·median(|p|)` | operator audit: effective step ~1e-8 → inert |

**The decisive bf16 arithmetic (why Stage 3 / Phase 3 scales are structurally dead at bf16).** A scale parameter of magnitude `m` (v7 balanced class: `m ∈ [2e-2, 1.2e-1]`, measured from the artifact) has bf16 ULP `≈ m·2^-8 ≈ 1e-4`-class. An Adam step moves it by `~ lr · (unit ≈ 1)` absolute. Under the **paper's own** absolute lr `1e-5`, the step `1e-5` is ~20× *below* the ULP `≈ 2e-4`: even the paper's calibrated lr cannot move a bf16-stored scale in a single step. Under the **shipped per-tensor** lr `1e-5 · m ≈ 5e-7`, the step is ~400× below the ULP. Either way, at bf16 *storage* the scale parameters round every step back to zero — and the v7 record confirms it: `s1_rel_move == s2_rel_move == 0.0` on all 196 rows, with nonzero first-step gradients (`6.6e-2..224`, above the floor, so the stage *runs* and *fails to move anything*). The paper never declares the storage dtype of its factorized parameters (App. C gives torch 2.6.0 but no dtype), so this is a paper-side omission colliding with this repo's bf16-storage ruling — the single sharpest interaction between "bf16 compute" and "the paper's stages" in the v7 run.

**Loss-weight scale (TuneFP/STE input).** The raw terminal weight `D_out = E[g²]` measured `7.1e-14 .. 2.7e-10` (mean `2.9e-11`) — a mean-*squared*-gradient statistic at hidden-state scale. Un-normalized, the weighted loss sits ~1e10× under the residual, and the step-0 gradient lands below Adam's absolute `eps` (the v5 inert regime, `design.md` §23). The unit-mean normalization (`_stats.py:217-236`) fixed this for v7: TuneFP now moves (weight_delta_rel median 5.4%), but the same scale then makes the *STE* scale updates sub-ULP (previous paragraph). The two fixes interact: normalization made TuneFP live and left STE's scale path dead against bf16 storage.

---

## 7. Ranked falsifiable hypotheses + probe designs

Ranking key: (evidence-now) × (damage-plausibility at bf16) × (probe cost). Probes are single-stage / one-block / small-fixture and CPU-fine unless noted. Each names the measurement that would **indict** vs **exonerate**, per the gradient-probe rule (absence provable by measurement, never flag-assertion).

### H1 (F1) — per-tensor median-relative lr × bf16 storage ⇒ Stage 3 / Phase 3 scales (and most latents) cannot move. **[highest-ranked]**

- **Mechanism.** `lr = stage_latent.lr · median(|p|)` (`_ste.py:119-130`, `scale_calib.py:148-157`) shrinks the App. C absolute lrs (`1e-5`/`1e-6`, `:3005-3006`) by the parameter magnitude, and the bf16 ULP makes the resulting step round to zero (§6 arithmetic). The record already shows the consequence: `s1_rel_move == s2_rel_move == 0.0` exactly on all 196 STE rows, `u_rel_move ≈ 1.4e-5` (below the bf16 resolution floor → declared inert), KD effective step ~1e-8.
- **Probe (one block, r512, both bf16 and fp32 arms; CPU-fine).** On the *same* reconstructed block state (post-balance latents + scales, one block, one calibration batch), run TuneLatentSTE for 128 steps in four arms: {per-tensor lr, absolute lr} × {bf16 params, fp32 params}. Record `s1_rel_move, s2_rel_move, u_rel_move, v_rel_move, flips_u, flips_v, residual_after/before`.
- **Pass/fail (indict):** absolute-lr@bf16 or per-tensor@fp32 produce `s1_rel_move > 0` while per-tensor@bf16 yields exactly 0 — then the interaction is proven by the *contrast*, not the single zero. **Exonerate:** per-tensor@bf16 shows `s1_rel_move ≥ floor_rel` (scales move) *or* absolute-lr@bf16 also yields 0 (then bf16 storage alone kills the scales even at the paper's lr, and the fix is parameter dtype, not the lr form — a different, equally actionable verdict).
- **Cost:** minutes; one block; no export; uses the existing stage instrument (the four fields are already recorded).

### H2 (F2) — the magnitude-balance alpha refit enforces `‖Wq‖/‖W‖ = cos`, a network-wide ~0.72× attenuation nothing corrects. **[2nd]**

- **Mechanism.** `alpha = <W, M>/‖M‖²` folded into `s1` (`lb_admm.py:691-707`) projects the reconstruction onto `span(M)`, so `‖Wq‖/‖W‖ ≡ cos(W, M)` by identity. With the gain-correcting stages inert (H1), the ~0.72× attenuation ships.
- **Probe (one block; CPU-fine).** For one block, run `magnitude_balance` twice on identical post-ADMM proxies: refit-on (shipped) vs refit-off (Eq. 8 as-written, no global scalar). Report `‖Wq‖/‖W‖`, per-linear cos vs pristine W, and (cheap) the block-output residual of the two factorizations on `X_b`.
- **Pass/fail (indict):** refit-on yields `‖Wq‖/‖W‖ ≈ cos < 1` (the identity), refit-off yields `‖Wq‖/‖W‖` closer to 1 with cos degraded only slightly, and refit-off's block-output residual is lower. **Exonerate:** refit-off is *worse* on cos/residual (the refit earns its keep and the attenuation is a symptom, not a cause).
- **Cost:** minutes; no optimizer; the two arms differ in one scalar.

### H3 (F3) — joint-block STE (vs Algorithm 1 line 19's per-target-linear refinement) cuts the refinement budget ~7× and trades one linear against another. **[3rd]**

- **Mechanism.** `_block_stage.py:585-613` runs one joint pass; the App. C factorized-stage budget (`:3007`) is consumed once for 7 linears, and the objective is block-output, so one linear's residual can cancel another's under a shared bf16 loss. The paper's line-19 prose + Figure 3 name a per-target refinement.
- **Probe (one block; two arms; CPU-fine).** Same reconstructed block: (a) shipped joint pass (1024 steps over all `facs`), (b) per-target passes (each linear refined right after its factorization, its own `epochs × n/bs` budget). Compare post-STE per-linear cos vs pristine W and block-output residual.
- **Pass/fail (indict):** per-target arm improves per-linear cos/residual over the joint arm by a margin > noise (say ≥0.01 cos). **Exonerate:** joint ≥ per-target (the trade-across-linears claim is real and the budget cut is harmless).
- **Cost:** ~2× the H1 cost; one block; no export.

### H4 — TuneFP's weighting (terminal `D_out` vs uniform) changes the fixed point of Step 1. **[required probe; 4th]**

- **Mechanism.** The paper names "weighted MSE" (`:3012`) but not the weight (§5); the shipped weight is terminal-`D_out` unit-mean (`_tuning.py:310`). If the two candidate objectives pick different updates, the compensation is objective-dependent and the *wrong* one could inject error into the tuned weights the solver then factorizes.
- **Probe (one block after the first factorization; CPU-fine).** On one per-target state, compute the step-0 gradient of (i) uniform MSE and (ii) terminal-`D_out`-weighted MSE (both unit-mean-normalized) w.r.t. the same tunable parameters; report `cos(g1, g2)` and run 16 steps of each arm, comparing residual reduction.
- **Pass/fail (indict):** `cos(g1, g2) < 0.9` and the two arms move the residual differently — the weight is behaviorally load-bearing. **Exonerate:** `cos(g1, g2) > 0.99` (unit-mean normalization made the weight effectively uniform; the objective is not the damage source).
- **Cost:** one block, minutes; gradient-only, no full T_pre budget needed to read the *direction*.

### H5 — ADMM at bf16: the normalized-design ridge (N3) puts κ(H) ≈ 1.7e4 at ρ=0, and the bf16-rounded target/latents degrade the solve. **[required audit; 5th]**

- **Mechanism.** Cor. 2 (`:2945`) with the code's normalized design gives `κ(H) ≤ 1 + r/(ρ + admm_reg)` = `1 + 512/0.03 ≈ 1.7e4` at ρ=0 (reg=3e-2, `config.py:222`), falling to ~5e2 at ρ=1; the target is rounded to bf16 at entry (`lb_admm.py:459`) and the final latents to bf16 at return (`:610-615`).
- **Probe (one block, two arms; CPU-fine — the NumPy fp64 oracle path exists).** Re-run LB-ADMM on one layer's preconditioned target: (a) shipped torch/bf16-entry/fp32-accumulate, (b) NumPy fp64 oracle (same tree, `solver_backend='numpy'`, `compute_dtype='float64'`). Compare `cos_raw`, `n_iter`, `converged`, and the residual trajectory over 400 iterations.
- **Pass/fail (indict):** arm (b) converges (`converged=True`) or reaches materially higher `cos_raw` (>1e-2 gap) — bf16 entry/latent rounding is degrading the solve. **Exonerate:** arms agree within ~1e-3 (the bf16 boundary rounding is sub-noise; `converged=False` is a feature of the ρ schedule, not a defect).
- **Cost:** one layer × 400 iterations × 2 arms; minutes (the oracle arm is the existing fixture path).

### H6 — Phase-3 KD scale updates are inert at bf16 (per-tensor lr + trust region), so the designated global-gain corrector never runs. **[required probe; 6th]**

- **Mechanism.** KD uses the same per-tensor lr (`scale_calib.py:148-157`) as STE and clips any move to `[0.5×, 2×]` (`:310-315`); at bf16 storage the effective step ~1e-8 never clears the ULP.
- **Probe (tiny fixture; CPU-fine).** On the existing `_SevenLinearBlock`-class fixture at bf16, run `calibrate_scales` for the full budget and record per-layer `s1`/`s2` before/after; also assert the frozen-binary invariant (Ub/Vb byte-identical across the stage) with a poison arm (flip one packed bit → the gate must catch it, per the KD frozen-binary semantics already in `scale_calib.py`).
- **Pass/fail (indict):** `‖s_after − s_before‖ == 0.0` for all layers (KD inert) while binaries are provably frozen. **Exonerate:** scales move measurably (the designated corrector is live, and the 18× must live elsewhere).
- **Cost:** tiny fixture, seconds-to-minutes; no export.

### H7 (flagged, not ranked as a cause) — the `flips_u`/`flips_v` discrepancy

The operator audit reports "flips_u ≈ 2–3 per linear vs flips_v ≈ 579–749 — sign refinement is V-side-only." The recorded instrument disagrees: v7 `flips_u` median **65** (mean 111.8, p25 32, p75 179), `flips_v` median **17** (mean 88); v6b `flips_u` median 69, `flips_v` median 20; per-role it is *mixed* (gate_proj `u=225 v=5`, but down_proj `u=16 v=344`). This is **not** a silent disagreement to resolve in the dark — it is recorded for the reviewer/operator to reconcile: the recorded `flips_u`/`flips_v` are *cumulative* sign crossings (a latent can cross and cross back), while "ships at ADMM-init signs" would require a *net* entry-vs-exit sign comparison the report does not store. The ranked hypotheses do not depend on the split: H1's decisive evidence is the exact-zero `s1/s2_rel_move` and the sub-resolution `u_rel_move`, both of which are unambiguous.

---

## 8. What is not covered / open

- The v7 artifact's scale *imbalance* is absent (§6: median ratio 0.896) — the earlier 13–17-decade imbalance (`design.md` §17) is gone in this code state; "scale imbalance as the bridge" is retired for v7.
- `converged=False` on 196/196 is consistent with the normalized-design ridge + linear-ρ-from-zero schedule, but the paper's own D.4 (`:3200-3213`) shows the linear schedule *is* the quality choice, so non-convergence is not itself evidence of breakage — H5's oracle comparison is the arbiter.
- Whether the paper's tuning stages were themselves run in fp16/bf16/fp32 is unstated in the corpus (`App. C` gives torch 2.6.0 only) — the bf16-audit's central collision (paper lr vs bf16 ULP) is therefore a *paper omission*, not a paper claim we contradict.

---

## 9. Evidence index

| artifact | sha / count |
|---|---|
| `nanoquant-2602.06694v3.md` (pinned paper) | the sole paper object; all `:line` citations above |
| v7 `quantize_report.json` | sha256 `7f330b5e6baf8e663091c05216d4eb0ba3bcb6a168f39c55e24e7d5faf9ce306` |
| v7 `solver_probe.jsonl` | 196 rows |
| v7 `stages[]` | 282 events (168 tune_fp + 28 ste + 28 lb_admm + 1 kd + …) |
| v7 artifact `layers/**/*.qpk1` | 196 streams; scale magnitudes parsed (median `|s1|/|s2|` 0.896, range 0.48–2.02) |
| v6b `quantize_report.json` | comparator leg (ratio 14.14×) |

*Analysis only — no code changed, no refire authorized. The ranked list above feeds a separate decision on what to probe next.*
