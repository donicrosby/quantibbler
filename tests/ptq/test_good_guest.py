"""Good-guest resource discipline (qk-ptq: "The CPU path SHALL size thread pools
to available cores with load-aware backoff" and "GPU memory allocation SHALL be
capped at a default fraction with a user override").

Scenarios: thread_sizing_light_load, thread_sizing_heavy_load,
thread_sizing_moderate_load, thread_sizing_small_affinity, max_threads_cap,
no_good_guest_disables, gpu_memory_fraction_override.

The sizing is a pure function of the affinity set and the 1-minute load average,
so every branch is constructible here without a loaded host.
"""

import pytest

from quantibbler.good_guest import (
    PRE_IMPORT_ENV,
    apply_thread_env,
    compute_thread_count,
    plan_from_argv,
    resolve_memory_fraction,
)


def test_scn_thread_sizing_light_load():
    """Below half the cores busy: leave two cores."""
    assert compute_thread_count(48, 1.0) == (46, "light")
    assert compute_thread_count(8, 3.9) == (6, "light")


def test_scn_thread_sizing_moderate_load():
    """At or above half the cores, not above the core count: leave four."""
    assert compute_thread_count(8, 4.0) == (4, "moderate")
    assert compute_thread_count(48, 47.9) == (44, "moderate")


def test_scn_thread_sizing_heavy_load():
    """Above the core count: a quarter of the cores, floored at one."""
    assert compute_thread_count(48, 70.0) == (12, "heavy")
    assert compute_thread_count(48, 48.1) == (12, "heavy")


def test_scn_thread_sizing_small_affinity():
    """A 2-core cpuset still produces a usable count on every branch."""
    assert compute_thread_count(2, 0.0) == (2, "light")
    assert compute_thread_count(2, 9.0) == (1, "heavy")


def test_scn_max_threads_cap():
    """--max-threads only ever tightens the computed bound."""
    plan = plan_from_argv(["--max-threads", "1"])
    assert plan.n_threads == min(compute_thread_count(plan.n_cores, plan.loadavg)[0], 1)
    assert plan.n_threads >= 1
    assert plan.max_threads == 1


def test_scn_no_good_guest_disables():
    """--no-good-guest writes NO thread variable at all."""
    plan = plan_from_argv(["--no-good-guest"])
    assert plan.enabled is False
    env = {}
    apply_thread_env(plan, environ=env)
    assert env == {}
    assert all(name not in env for name in PRE_IMPORT_ENV)


def test_good_guest_sets_all_three_blas_variables():
    """OMP alone does not size the pool numpy actually uses.

    numpy's scipy-openblas is PTHREADS-backed, so OMP/MKL pins leave its pool at
    the visible-CPU count (measured 235.5s vs 3.98s). All three move together.
    """
    plan = plan_from_argv(["--max-threads", "3"])
    env = {}
    apply_thread_env(plan, environ=env)
    assert env == {name: "3" for name in PRE_IMPORT_ENV}


def test_scn_gpu_memory_fraction_override():
    """An explicit --gpu-memory-fraction reaches the plan verbatim."""
    assert plan_from_argv(["--gpu-memory-fraction", "0.5"]).gpu_memory_fraction == 0.5
    # Default when unset.
    assert plan_from_argv([]).gpu_memory_fraction == pytest.approx(0.85)


def test_gpu_memory_gb_cap_wins_when_tighter():
    """A unified-memory APU needs the absolute bound, not the fraction.

    Device memory IS host RAM there, so 0.85 of a large device can still drain
    the host past the container's fence; the GB cap is resolved into the same
    fraction and the tighter of the two is what gets applied.
    """
    # 8 GiB of a 64 GiB device is tighter than 0.85.
    assert resolve_memory_fraction(0.85, 8.0, 64 * 1024**3) == pytest.approx(0.125)
    # A GB cap looser than the fraction leaves the fraction in place.
    assert resolve_memory_fraction(0.5, 60.0, 64 * 1024**3) == pytest.approx(0.5)
    # Without a total to divide by the GB cap cannot be expressed: fraction stands.
    assert resolve_memory_fraction(0.85, 8.0, None) == pytest.approx(0.85)


def test_plan_records_the_host_state_it_used():
    """The plan carries its inputs, so a run report can state the reason."""
    plan = plan_from_argv(["--max-threads", "2"])
    assert plan.n_cores >= 1
    assert plan.reason.endswith("+max_threads")
    assert plan.loadavg >= 0.0
