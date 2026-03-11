# Element and Lattice

## Element

An `Element` aggregates all data for a single lattice element: head metadata,
general attributes, Twiss parameters, orbit, transfer matrix, floor
coordinates, multipoles, wake fields, chamber walls, and more.

Data is loaded on-demand via `from_tao()` or `fill()`, controlled by
`defaults` and per-field boolean flags. The class-level `DEFAULTS` set
determines which fields are loaded when `defaults=True`.

::: pytao.model.Element

## Lattice

A `Lattice` is a tuple of `Element` objects with lookup dictionaries
by name, key (element type), and index.

::: pytao.model.Lattice

## ElementID

Parses and represents Tao's element identifier syntax, including universe,
branch, key, name, match number, and offset.

::: pytao.model.ElementID

## GeneralAttributes

Dynamic container for element-type-specific attributes (e.g., `L`, `K1`,
`angle`). Supports case-insensitive access. Each attribute is an `Attr`
with name, data, units, type, and a settable flag.

::: pytao.model.GeneralAttributes

## Element Sub-Models

These are the data classes that populate the fields of `Element`.

### ElementHead

::: pytao.model.ElementHead

### ElementTwiss

::: pytao.model.ElementTwiss

### ElementOrbit

::: pytao.model.ElementOrbit

### ElementMat6

::: pytao.model.ElementMat6

### ElementFloorAll

::: pytao.model.ElementFloorAll

### ElementFloor

::: pytao.model.ElementFloor

### ElementFloorPosition

::: pytao.model.ElementFloorPosition

### ElementBunchParams

::: pytao.model.ElementBunchParams

### ElementMultipoles

::: pytao.model.ElementMultipoles

### ElementPhoton

::: pytao.model.ElementPhoton

### ElementWake

::: pytao.model.ElementWake

### ElementWall3D

::: pytao.model.ElementWall3D

### ElementChamberWall

::: pytao.model.ElementChamberWall

### ElementLordSlave

::: pytao.model.ElementLordSlave

### ElementGridField

::: pytao.model.ElementGridField

## Comb

Cumulative bunch moment data across the lattice.

::: pytao.model.Comb

## Helper Types

### ElementRange

::: pytao.model.ElementRange

### ElementList

::: pytao.model.ElementList

### ElementIntersection

::: pytao.model.ElementIntersection
