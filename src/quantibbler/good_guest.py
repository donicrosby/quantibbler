"""Good-guest resource discipline: bounds derived from observed host state.

Defaults are considerate, never system-killing. A fire sizes its CPU thread
pools to the cores it can actually use and backs off under load, caps its GPU
memory, and spawns no unbounded workers; ``--no-good-guest`` is the explicit
opt-out for benchmarking.

Stdlib-only at import time, deliberately: OMP/MKL/OPENBLAS read their thread
counts when the BLAS library LOADS, so the sizing must run before ``import
torch``. A module that pulled torch in to size a thread pool could not satisfy
its own contract.

The anti-pattern this exists to prevent: the v6 fire's hardcoded
``OMP_NUM_THREADS=16`` sat on a 48-core host that several lanes share, so a
BLAS-heavy ADMM solve left 0 free cores and the host's 1-minute load average
spiked to 47-70, starving co-tenant agents. Thread count is therefore a function
of ``sched_getaffinity`` (cpuset-respecting) and ``/proc/loadavg``, not a
constant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "PRE_IMPORT_ENV",
    "GoodGuestPlan",
    "apply_gpu_memory_cap",
    "apply_thread_env",
    "apply_torch_threads",
    "available_cores",
    "compute_thread_count",
    "is_unified_memory_device",
    "loadavg_1m",
    "plan_from_argv",
    "resolve_memory_fraction",
]

#: The env variables a BLAS library reads at LOAD time. Setting any of them
#: after the library is loaded changes nothing.
PRE_IMPORT_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

#: Default per-process GPU memory cap. 0.85 leaves headroom for the driver,
#: the tokenizer's host allocations and any co-tenant sharing the device.
DEFAULT_GPU_FRACTION = 0.85


@dataclass(frozen=True)
class GoodGuestPlan:
    """The sizing decision and the inputs it was derived from."""

    enabled: bool
    n_cores: int
    loadavg: float
    n_threads: int
    reason: str
    max_threads: int | None = None
    gpu_memory_fraction: float = DEFAULT_GPU_FRACTION
    gpu_memory_gb: float | None = None


def available_cores() -> int:
    """Cores this process may use — the affinity set, not the machine's count.

    A container with ``--cpus 2`` must size to 2, so ``os.cpu_count()`` (which
    reports the HOST's CPUs) is only the fallback where ``sched_getaffinity``
    does not exist (non-Linux).
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        n = os.cpu_count() or 1
    return max(1, n)


def loadavg_1m() -> float:
    """The 1-minute load average, or 0.0 where the host has none to report."""
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):  # pragma: no cover - non-POSIX
        return 0.0


def compute_thread_count(n_cores: int, load: float) -> tuple[int, str]:
    """Thread count and the branch that produced it, from host state.

    Light (< half the cores busy): leave two cores. Moderate (at or above half,
    not above the core count): leave four — an over-subscribed box needs more
    headroom than an idle one. Heavy (load above the core count): a quarter of
    the cores, floored at one, so a fire that must run on a busy host still
    makes progress without owning it.
    """
    if load > n_cores:
        return max(1, n_cores // 4), "heavy"
    if load >= n_cores * 0.5:
        return max(2, n_cores - 4), "moderate"
    return max(2, n_cores - 2), "light"


def plan_from_argv(argv: list[str]) -> GoodGuestPlan:
    """Plan the bounds from the command line, before anything is imported.

    A minimal pre-parse, because argparse cannot run before the import it must
    precede: only the three resource flags are recognised here, and anything
    unrecognised is left for the real parser.
    """
    enabled = "--no-good-guest" not in argv
    max_threads = _flag_int(argv, "--max-threads")
    fraction = _flag_float(argv, "--gpu-memory-fraction")
    gb = _flag_float(argv, "--gpu-memory-gb")

    n_cores = available_cores()
    load = loadavg_1m()
    if not enabled:
        return GoodGuestPlan(
            enabled=False,
            n_cores=n_cores,
            loadavg=load,
            n_threads=n_cores,
            reason="no_good_guest",
            max_threads=max_threads,
            gpu_memory_fraction=DEFAULT_GPU_FRACTION if fraction is None else fraction,
            gpu_memory_gb=gb,
        )
    n_threads, reason = compute_thread_count(n_cores, load)
    if max_threads is not None:
        n_threads = min(n_threads, max_threads)
        reason = f"{reason}+max_threads"
    return GoodGuestPlan(
        enabled=True,
        n_cores=n_cores,
        loadavg=load,
        n_threads=n_threads,
        reason=reason,
        max_threads=max_threads,
        gpu_memory_fraction=DEFAULT_GPU_FRACTION if fraction is None else fraction,
        gpu_memory_gb=gb,
    )


def apply_thread_env(plan: GoodGuestPlan, environ=None) -> None:
    """Set the BLAS thread variables. MUST run before the BLAS library loads.

    ``--no-good-guest`` writes nothing: an opt-out that still pinned threads
    would be a bound wearing an opt-out's name.
    """
    env = os.environ if environ is None else environ
    if not plan.enabled:
        return
    for name in PRE_IMPORT_ENV:
        env[name] = str(plan.n_threads)


def apply_torch_threads(plan: GoodGuestPlan, torch_module=None) -> int:
    """Apply the same count to torch's own pool; returns ``get_num_threads()``.

    torch does not read any thread environment variable, so the env trio alone
    leaves its intra-op pool at the default. Called after ``import torch`` and
    the return value is the startup assertion's subject.
    """
    if not plan.enabled:
        return -1
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415
    torch_module.set_num_threads(plan.n_threads)
    return int(torch_module.get_num_threads())


def resolve_memory_fraction(
    fraction: float = DEFAULT_GPU_FRACTION,
    gb: float | None = None,
    total_bytes: int | None = None,
) -> float:
    """The fraction to hand ``set_per_process_memory_fraction``.

    An absolute-GB cap (``--gpu-memory-gb``) is what bounds a UNIFIED-memory
    device: on an APU the device memory IS host RAM, so a bare 0.85-of-device
    fraction can still drain the host beyond the container's own fence. When
    both are given the tighter bound wins; without a total to divide by the GB
    cap cannot be expressed as a fraction and the fraction stands.
    """
    if gb is None or total_bytes is None or total_bytes <= 0:
        return fraction
    return min(fraction, float(gb) * (1024**3) / float(total_bytes))


def apply_gpu_memory_cap(plan: GoodGuestPlan, torch_module=None) -> float | None:
    """Cap this process's GPU memory at startup; the applied fraction or None.

    Returns None when no accelerator is present (there is nothing to bound) or
    under ``--no-good-guest``, so a caller can record the absence rather than
    infer it.
    """
    if not plan.enabled:
        return None
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415

    if not torch_module.cuda.is_available():
        return None
    total = int(torch_module.cuda.get_device_properties(0).total_memory)
    fraction = resolve_memory_fraction(plan.gpu_memory_fraction, plan.gpu_memory_gb, total)
    torch_module.cuda.set_per_process_memory_fraction(fraction, device=0)
    return fraction


def is_unified_memory_device(torch_module=None) -> bool:
    """Whether device memory IS host RAM (an APU, e.g. Strix Halo/gfx1151).

    On such a device the per-process FRACTION alone can still drain the host,
    because the device allocates out of the same pool the OS uses; the absolute
    ``--gpu-memory-gb`` bound is what actually protects co-tenants. Detected from
    the HIP arch name rather than assumed from the vendor.
    """
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415

    if not torch_module.cuda.is_available():
        return False
    if getattr(torch_module.version, "hip", None) is None:
        return False  # CUDA devices carry their own VRAM
    try:
        arch = str(torch_module.cuda.get_device_properties(0).gcnArchName)
    except AttributeError:  # pragma: no cover - older torch builds
        return False
    return arch.startswith("gfx11")


def _flag_int(argv: list[str], name: str) -> int | None:
    value = _flag_value(argv, name)
    return None if value is None else int(value)


def _flag_float(argv: list[str], name: str) -> float | None:
    value = _flag_value(argv, name)
    return None if value is None else float(value)


def _flag_value(argv: list[str], name: str) -> str | None:
    """``--name V`` or ``--name=V``; the last occurrence wins, as argparse does."""
    found: str | None = None
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            found = argv[i + 1]
        elif token.startswith(f"{name}="):
            found = token.split("=", 1)[1]
    return found
