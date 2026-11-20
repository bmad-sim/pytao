"""
Integration tests for :mod:`pytao.optimize`.

These run against a real Tao binary via the packaged ``optics_matching_tweaked``
init file. We use the existing ``get_packaged_example`` + ``run_context`` pattern
so the tests cooperate with pytao's ``TAO_REUSE_SUBPROCESS=1`` CI setting and
with conftest's live Tao fixture machinery.
"""

from __future__ import annotations

import math

import numpy as np

from pytao.optimize import TaoOptimizationProblem

from .conftest import get_packaged_example


def test_extracts_quad_variable_group(use_subprocess: bool):
    """optics_matching_tweaked declares a 'quad' v1 group; extract it."""
    startup = get_packaged_example("optics_matching_tweaked")
    with startup.run_context(use_subprocess=use_subprocess) as tao:
        problem = TaoOptimizationProblem.from_tao(tao)
        v1_names = {v.v1_name for v in problem.variables}
        assert v1_names == {"quad"}
        assert problem.n_var > 0


def test_extracts_twiss_datums_with_mixed_merit_types(use_subprocess: bool):
    """
    The packaged init declares twiss.end (all 'target') and twiss.max
    ('max' + 'abs_max'). All should be active.
    """
    startup = get_packaged_example("optics_matching_tweaked")
    with startup.run_context(use_subprocess=use_subprocess) as tao:
        problem = TaoOptimizationProblem.from_tao(tao)
        d2_names = {d.d2_name for d in problem.datums}
        assert d2_names == {"twiss"}
        merit_types = {d.merit_type for d in problem.datums}
        assert {"target", "max", "abs_max"}.issubset(merit_types)
        assert problem.n_data >= 6


def test_universe_resolves_to_tao_default(use_subprocess: bool):
    """With no argument, from_tao() snapshots Tao's live default_universe.

    We assert it resolved to *an* int rather than a specific value — the
    reusable-subprocess fixture can leave Tao's default_universe in a
    non-1 state if earlier tests mutated it.
    """
    startup = get_packaged_example("optics_matching_tweaked")
    with startup.run_context(use_subprocess=use_subprocess) as tao:
        problem = TaoOptimizationProblem.from_tao(tao)
        assert isinstance(problem.universe, int)
        assert problem.universe >= 1


def test_unbounded_limits_translate_to_inf(use_subprocess: bool):
    """
    The packaged init leaves ``default_low_lim`` / ``default_high_lim``
    commented out, so Tao fills in its ±1e30 sentinels. bounds_array must
    translate those to ±inf.
    """
    startup = get_packaged_example("optics_matching_tweaked")
    with startup.run_context(use_subprocess=use_subprocess) as tao:
        problem = TaoOptimizationProblem.from_tao(tao)
        lb, ub = problem.bounds_array
        # No ±1e20+ sentinel should leak through.
        assert not ((np.abs(lb) >= 1e20) & ~np.isinf(lb)).any()
        assert not ((np.abs(ub) >= 1e20) & ~np.isinf(ub)).any()
        # Expect at least one side to end up at ±inf for this init file.
        assert np.isinf(lb).any() or np.isinf(ub).any()


def test_snapshot_is_tuple(use_subprocess: bool):
    startup = get_packaged_example("optics_matching_tweaked")
    with startup.run_context(use_subprocess=use_subprocess) as tao:
        problem = TaoOptimizationProblem.from_tao(tao)
        assert isinstance(problem.variables, tuple)
        assert isinstance(problem.datums, tuple)


def test_x0_length_matches_n_var(use_subprocess: bool):
    startup = get_packaged_example("optics_matching_tweaked")
    with startup.run_context(use_subprocess=use_subprocess) as tao:
        problem = TaoOptimizationProblem.from_tao(tao)
        assert problem.x0.shape == (problem.n_var,)
        assert all(math.isfinite(v) for v in problem.x0)
