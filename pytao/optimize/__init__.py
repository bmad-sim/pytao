"""
Inspect a live Tao instance's optimization setup from Python.

This subpackage exposes the variables, datums, bounds, and weights that Tao
itself would use for optimization, as structured Python objects. That makes it
straightforward to feed them into external optimizer libraries in follow-up
work — but nothing in this first slice mutates Tao state or evaluates merit.

Start with :class:`TaoOptimizationProblem`: construct one from a live
:class:`~pytao.Tao`, then inspect :attr:`~TaoOptimizationProblem.variables`,
:attr:`~TaoOptimizationProblem.datums`, :attr:`~TaoOptimizationProblem.x0`,
:attr:`~TaoOptimizationProblem.bounds`, and :attr:`~TaoOptimizationProblem.weights`.
"""

from __future__ import annotations

from .problem import (
    DatumInfo,
    TaoOptimizationProblem,
    VariableInfo,
)

__all__ = [
    "DatumInfo",
    "TaoOptimizationProblem",
    "VariableInfo",
]
