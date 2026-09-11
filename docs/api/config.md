# TaoConfig

`TaoConfig` captures the complete state of a running Tao session:
startup parameters, Bmad common settings, space charge settings,
beam initialization, beam tracking parameters, global settings,
and per-element overrides.

It can generate `set` commands (all or only changed), apply them back to
a Tao instance, and write reproducible bash loader scripts for archiving.

::: pytao.model.TaoConfig

## TaylorMap

::: pytao.model.TaylorMap

## Settings Sub-Models

These are the `TaoSettableModel` subclasses that `TaoConfig` aggregates.
Each corresponds to a Tao settings group and supports `from_tao()`,
`set_commands`, `get_set_commands(tao=...)`, and `set(tao)`.

### BmadCom

::: pytao.model.BmadCom

### SpaceChargeCom

::: pytao.model.SpaceChargeCom

### BeamInit

::: pytao.model.BeamInit

### Beam

::: pytao.model.Beam

### TaoGlobal

::: pytao.model.TaoGlobal
