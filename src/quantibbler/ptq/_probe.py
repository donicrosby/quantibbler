"""Recording seam: the per-linear solver-probe JSONL rows and the crash-safe
per-block breadcrumb.

Split out of ``block_recon``; same code, same call graph.
``SOLVER_PROBE_ENV`` stays imported from ``.lb_admm``: the same flag gates the
solver's per-iteration eigenvalue probe, and this writer must never be able to
name a different variable than the gate reads.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .lb_admm import SOLVER_PROBE_ENV
from .solver_backend import solver_backend_provenance, to_host_np

# `SOLVER_PROBE_ENV` is imported from ``lb_admm``: the same flag gates the
# solver's per-iteration eigenvalue probe, and this writer must never be able to
# name a different variable than the gate reads.
SOLVER_PROBE_PATH_ENV = "QK_SOLVER_PROBE_PATH"


SOLVER_PROBE_DEFAULT_PATH = "/out/solver_probe.jsonl"


def _record_solver_probe(name, w, bal, res, cfg) -> None:
    """Append one JSONL row per solved linear: the solver's own numbers.

    Diagnostic, read-only, OFF by default -- with QK_SOLVER_PROBE unset this
    returns before touching an array, so a default run is unchanged.

    Why it exists: the attribution probe needs per-linear cos and the solve's
    iteration count, and NOTHING in this tree records either. Every ``cos`` in
    the solver path is the cosine LR schedule; ``n_iter``/``converged`` live on
    ADMMResult and are dropped; the PTQ stage prints nothing per block. And
    stage records only reach disk in the end-of-run report, so a per-layer
    probe that is stopped early -- which a per-layer probe must be -- leaves no
    numbers at all. Hence JSONL appended as each linear solves.

    Both cos numbers are recorded deliberately, because they answer different
    questions and the gap between them is measured history: the SVD arm split
    raw 0.84 / pack 0.45, and a single number would have hidden that the pack
    stage, not the solve, is the bottleneck.

      cos_raw  = cos(W, U V^T)                              [NQ] Eq. 9 state
      cos_pack = cos(W, s1 o (sign(U) sign(V)^T) o s2^T)    [NQ] Eq. 10 packed

    Both against the ORIGINAL (un-preconditioned) FP weight W, never W~, per
    the stage-boundary pin. The effective rho_end and its source ride along, so
    a probe row can never be mislabelled about which arm produced it.

    ``rel_raw`` measures ``||W - U V^T||_F / ||W||_F`` on the CONTINUOUS
    product, which is deliberately NOT fit to W: the alpha refit fits the
    *packed* product (see lb_admm.magnitude_balance -- it corrects
    ``s1 o Ub Vb^T o s2^T`` and never rescales the latents). Nothing consumes
    the continuous product either: every site that turns latents into a weight
    goes through ``_factor_terms``, whose binary mode applies ``sign``, i.e.
    the packed class -- the module forward, the live STE forward
    (``_quantized_forward_block``) and the ``_set_linear`` write-back alike.

    So ``rel_raw > 1`` is EXPECTED on the ADMM arm and is not a defect signal.
    It reads as magnitude drift of the continuous latents, which is the
    consensus shaping that makes the packed product good in the first place
    (measured: ADMM packs at -0.133, SVD at -0.389 on the same real weight).
    Driving that drift toward 1.0 would trade away the pack
    fidelity the solver is chosen for. Do not close it as a balance-stage bug;
    the useful reading is that raw fit is a weak predictor of the served
    artifact, which is the same lesson the SVD arm taught from the other side.
    """
    if not os.environ.get(SOLVER_PROBE_ENV):
        return

    from .lb_admm import admm_reg_provenance, rho_end_provenance
    from .solver_backend import resolve_device

    u = to_host_np(bal.U)
    v = to_host_np(bal.V)
    s1 = to_host_np(bal.s1)
    s2 = to_host_np(bal.s2)
    wf = to_host_np(w)

    def _cos(recon: np.ndarray) -> float:
        num = float(np.sum(wf * recon))
        den = float(np.linalg.norm(wf) * np.linalg.norm(recon))
        return num / den if den > 0.0 else float("nan")

    recon_raw = u @ v.T
    recon_pack = (s1[:, None] * (np.sign(u) @ np.sign(v).T)) * s2[None, :]
    nw = float(np.linalg.norm(wf))

    def _rel(recon: np.ndarray) -> float:
        return float(np.linalg.norm(wf - recon)) / nw if nw > 0.0 else float("nan")

    value, source = rho_end_provenance(cfg.rho_end)
    # Regime identity, resolved exactly as the solver resolves it. A row
    # without these cannot say which regime produced its cos, which is the whole
    # point of the regime label (old-regime rows are inadmissible for the claim).
    reg_value, reg_source = admm_reg_provenance(cfg.admm_reg)
    # Backend arm + the device it actually executed on. "torch" alone does not
    # identify a row: cpu and cuda are different numerics (reduction order), so
    # an unlabelled device is the same mislabelling class as an unlabelled rho.
    backend_value, backend_source = solver_backend_provenance(cfg.solver_backend)
    solver_device = resolve_device(backend_value, device=os.environ.get("QUANTIBBLER_DEVICE"))
    row = {
        "name": name,
        "solver": cfg.solver,
        # The dtype the solve actually ran in. Two arms whose rows differ only
        # in precision are otherwise indistinguishable -- the same provenance
        # gap as an unlabelled rho_end, and worse here, because an fp32 row's
        # cos is not comparable to an fp64 row's.
        "compute_dtype": cfg.compute_dtype,
        # Image identity, carried in from the wrapper via the container env: the
        # driver cannot inspect its own image, and a tag is mutable, so the
        # digest is what makes "did both arms run the same binary" answerable
        # from the artifact. Absent outside a container, and stated as such.
        "image_digest": os.environ.get("QUANTIBBLER_IMAGE_DIGEST", "unknown"),
        # Which device the torch stages ran on. An fp32 row on a GPU and an fp32
        # row on a CPU are different measurements of the same thing.
        "device": os.environ.get("QUANTIBBLER_DEVICE", "unknown"),
        "rho_end": value,
        "rho_end_source": source,
        "rho_start": cfg.rho_start,
        "rho_start_source": "config",
        "admm_reg": reg_value,
        "admm_reg_source": reg_source,
        "solver_backend": backend_value,
        "solver_backend_source": backend_source,
        "solver_device": solver_device,
        "n_iter": getattr(res, "n_iter", None),
        "converged": getattr(res, "converged", None),
        "w_fro": nw,
        "wq_raw_fro": float(np.linalg.norm(recon_raw)),
        "wq_pack_fro": float(np.linalg.norm(recon_pack)),
        "cos_raw": _cos(recon_raw),
        "cos_pack": _cos(recon_pack),
        "rel_raw": _rel(recon_raw),
        "rel_pack": _rel(recon_pack),
        "rank": int(u.shape[1]),
        "ts": time.time(),
    }
    path = os.environ.get(SOLVER_PROBE_PATH_ENV) or SOLVER_PROBE_DEFAULT_PATH
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:
        # The numbers ARE the deliverable of a probe run: a probe that cannot
        # write them has measured nothing, so fail loudly rather than return.
        raise RuntimeError(f"solver probe enabled but cannot write {path}: {exc}") from exc
    print(
        f"[solver_probe:{name}] solver={row['solver']} rank={row['rank']} "
        f"n_iter={row['n_iter']} converged={row['converged']} "
        f"cos_raw={row['cos_raw']:.6f} cos_pack={row['cos_pack']:.6f} "
        f"rel_raw={row['rel_raw']:.6f} rel_pack={row['rel_pack']:.6f}",
        flush=True,
    )


def _note_block_progress(idx: int, name: str | None, x_running=None) -> None:
    """Append a crash-safe breadcrumb before each block reconstruction.

    The real-model driver runs under a hard cgroup cap; if the kernel
    SIGKILLs us mid-solve, stdout dies unflushed and the only surviving
    record of WHERE we died is this file. No-op on any error so the
    solver never depends on it.
    """
    path = os.getenv("QUANTIBBLER_BREADCRUMBS")
    if not path:
        return
    try:
        with open("/sys/fs/cgroup/memory.current") as fh:
            mem = int(fh.read()) / 2**30
        line = f"block[{idx}] {name or '(stem)'} cgroup={mem:.2f}G"
        if x_running is not None:
            line += f" x_running={tuple(x_running.shape)}@{x_running.dtype}"
        with open(path, "a") as fh:
            fh.write(f"{time.time():.0f} :: {line}\n")
    except Exception:
        pass
