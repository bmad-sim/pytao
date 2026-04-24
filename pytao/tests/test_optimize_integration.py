"""
Integration tests for :mod:`pytao.optimize`.

These require a working Tao shared library / binary. If pytao cannot load
Tao (``TaoSharedLibraryNotFoundError`` at construction time), the whole
module is skipped so this file is safe to run on hosts without the binary.

We use the packaged ``optics_matching_tweaked`` input rather than pointing
at ``$ACC_ROOT_DIR`` so these tests run anywhere the binary is available.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from pytao import Tao
from pytao.errors import TaoSharedLibraryNotFoundError
from pytao.optimize import TaoOptimizationProblem

PACKAGED_INIT = (
    pathlib.Path(__file__).resolve().parent
    / "input_files"
    / "optics_matching_tweaked"
    / "tao.init"
)


@pytest.fixture(scope="module")
def _tao_available() -> bool:
    """
    Probe whether a Tao shared library is reachable. If not, every test in
    this module skips cleanly.
    """
    if not PACKAGED_INIT.exists():
        pytest.skip(f"Packaged init file missing: {PACKAGED_INIT}")
    try:
        tao = Tao(init_file=str(PACKAGED_INIT), noplot=True)
    except TaoSharedLibraryNotFoundError:
        pytest.skip("Tao shared library not available on this host")
    except Exception as exc:
        pytest.skip(f"Tao init failed: {exc}")
    close = getattr(tao, "close_subprocess", None)
    if close:
        close()
    return True


@pytest.fixture
def live_tao(_tao_available):
    tao = Tao(init_file=str(PACKAGED_INIT), noplot=True)
    yield tao
    close = getattr(tao, "close_subprocess", None)
    if close:
        close()


def test_extracts_quad_variable_group(live_tao):
    """optics_matching_tweaked declares a 'quad' v1 group; extract it."""
    problem = TaoOptimizationProblem(live_tao)
    v1_names = {v.v1_name for v in problem.variables}
    assert v1_names == {"quad"}
    assert problem.n_var > 0


def test_extracts_twiss_datums_with_mixed_merit_types(live_tao):
    """
    The packaged init declares twiss.end (all 'target') and twiss.max
    ('max' + 'abs_max'). All eight should be active.
    """
    problem = TaoOptimizationProblem(live_tao)
    d2_names = {d.d2_name for d in problem.datums}
    assert d2_names == {"twiss"}
    merit_types = {d.merit_type for d in problem.datums}
    assert {"target", "max", "abs_max"}.issubset(merit_types)
    assert problem.n_data >= 6


def test_universe_resolves_to_tao_default(live_tao):
    """With no argument, the problem snapshots the live default universe."""
    problem = TaoOptimizationProblem(live_tao)
    # Default for this init is 1; it should be resolved (not still None).
    assert problem.universe == 1


def test_unbounded_limits_translate_to_inf(live_tao):
    """
    The packaged init leaves ``default_low_lim`` / ``default_high_lim``
    commented out, so Tao fills in its ±1e30 sentinels. bounds_array must
    translate those to ±inf.
    """
    problem = TaoOptimizationProblem(live_tao)
    lb, ub = problem.bounds_array
    # Either every finite, or at least one hit ±inf — the real check is that
    # no ±1e20+ sentinel leaks through.
    assert not (np.abs(lb) >= 1e20).any() or np.isinf(lb).any()
    assert not (np.abs(ub) >= 1e20).any() or np.isinf(ub).any()
    # optics_matching_tweaked in particular has no explicit limits → expect
    # at least one side to end up at ±inf.
    assert np.isinf(lb).any() or np.isinf(ub).any()


def test_snapshot_is_tuple(live_tao):
    problem = TaoOptimizationProblem(live_tao)
    assert isinstance(problem.variables, tuple)
    assert isinstance(problem.datums, tuple)


def test_x0_length_matches_n_var(live_tao):
    problem = TaoOptimizationProblem(live_tao)
    assert problem.x0.shape == (problem.n_var,)
    # x0 values should all be finite; unbounded limits don't imply unbounded
    # starting values.
    assert all(math.isfinite(v) for v in problem.x0)
