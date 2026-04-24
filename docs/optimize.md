# Inspecting a Tao optimization setup from Python

The `pytao.optimize` subpackage exposes the variables, datums, bounds, and
weights that Tao itself would use for optimization as structured Python
objects. That makes it straightforward to inspect a `tao.init` configuration
interactively, or to plug Tao into external optimizer libraries in follow-up
work.

This first release is intentionally read-only. It does not mutate Tao's
state or evaluate the merit function. Both of those capabilities will arrive
in subsequent releases; the data model here is the foundation they'll build on.

## Quickstart

```python
from pytao import Tao
from pytao.optimize import TaoOptimizationProblem

tao = Tao(init_file="tao.init", noplot=True)
problem = TaoOptimizationProblem(tao)

print(f"{problem.n_var} active variables, {problem.n_data} active datums")

for v in problem.variables:
    print(f"  {v.name} ({v.ele_name}.{v.attrib_name}): "
          f"initial={v.initial_value:.3g}  "
          f"bounds=[{v.low_lim:.3g}, {v.high_lim:.3g}]")

for d in problem.datums:
    print(f"  {d.name} ({d.data_type}, {d.merit_type}): "
          f"meas={d.meas_value:.3g}  weight={d.weight:.3g}")
```

## What gets picked up from the init file

`TaoOptimizationProblem` reads the live Tao state and collects:

- **Variables** with `useit_opt = True` (i.e., `good_user & good_opt`).
  Bounds come from `low_lim` / `high_lim`; values at or past ±10²⁰ are
  treated as "no limit" and translated to `±inf`. Steps, weights, merit
  types, and element/attribute names are preserved.
- **Datums** with `useit_opt = True`, with their `meas_value`, `weight`,
  and `merit_type`. Target-style datums (`merit_type = 'target'`) will
  become standard residuals when the evaluation layer lands; limit-style
  datums (`'min'`, `'max'`, `'abs_min'`, `'abs_max'`) are preserved so
  downstream code can reproduce Tao's soft-constraint semantics.

Negative weights are rejected at construction time — they're nonsensical in
Tao's merit function (`sum(w * delta**2)` requires `w >= 0`) and catching
them at the boundary avoids silent NaN production downstream.

## Properties you can read

```python
problem.n_var               # number of active optimization variables
problem.n_data              # number of active datums
problem.x0                  # initial variable vector (np.ndarray)
problem.variable_names      # ["v1[1]", "v1[2]", ...]
problem.bounds              # [(low, high), ...] — ±inf allowed
problem.bounds_array        # (lb, ub) ndarrays (scipy.least_squares shape)
problem.weights             # per-datum merit weights (np.ndarray)
problem.variables           # list[VariableInfo]
problem.datums              # list[DatumInfo]
```

`problem.variables` and `problem.datums` are tuples of frozen dataclasses, so
the snapshot is immutable — no accidental `append`/`pop` can desynchronise
it from the `x0`/`bounds`/`weights` arrays.

## Multi-universe sessions

By default, `TaoOptimizationProblem` reads Tao's live
`s%global%default_universe` and snapshots against it — the same behaviour as
Tao's own `pipe data_*` commands. That means in a single-universe session
you don't need to think about universes at all, and in a multi-universe
session the problem follows whatever universe was active when you built it.

To target a specific universe explicitly:

```python
problem = TaoOptimizationProblem(tao, universe=2)
```

Useful when you want to inspect several universes side-by-side, or when a
routine mutates Tao's default universe and you want the snapshot pinned to
a known one.

## What's next

Follow-up releases will layer on:

- `set_variables(x)` and `evaluate_merit(x)` for round-tripping variable
  updates through Tao.
- `evaluate_residuals(x)` and `jacobian(x)` mirroring Tao's merit function
  structure (including limit-type variable contributions).
- Adapters such as `run_scipy_minimize` and `run_scipy_least_squares` that
  let you drive Tao optimization from the SciPy ecosystem.

Each will ship as its own PR so reviewers can evaluate them independently.

## See also

- [API reference](api/optimize.md)
- [Tao manual — optimization chapter][tao-opt]

[tao-opt]: https://www.classe.cornell.edu/bmad/tao_manual.pdf
