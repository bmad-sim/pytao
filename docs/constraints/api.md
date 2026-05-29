
# Class Structure

The folowing inheritance diagrams are provided to aid developers in understanding the constraints codebase.

## Base Classes

### Observations and Observables

The principle object in the constraints tool is an `Observation`.
This abstract class represents the stored information from a measurement (from the lattice or from a literal).
These measurements are defined by `Observables` which have all of the information needed to produce the `Observation` from a loaded Tao lattice (in the case of a `LatticeObservable`) or from scratch (for a `LiteralObservable`).

An `Observable` is a hashable type allowing the map `obs_map: dict[Observable, Observation]` to be the context needed for constraint checking.
This abstracts the checks allowing collection to take place in a consolidated step that avoids loading lattices multiple times.
Constraints are designed to maximally tolerate and report missing data allowing all checks to be run even when some observations and lattices fail.
It also means that the `obs_map` may be saved to disk and loaded later for regression tests.

```mermaid
classDiagram
    class Observation
    class Observable~ObsT~
    class LatticeObservable~ObsT~ {
        +str lattice_id
        +__call__(tao) ObsT
    }
    class LiteralObservable~ObsT~ {
        +__call__() ObsT
    }
    Observable <|-- LatticeObservable
    Observable <|-- LiteralObservable

    LatticeObservable ..> Observation : creates
    LiteralObservable ..> Observation : creates
```

### Operators and Results

Comparisons are defined between two `Observation` objects of the same type in the form of operators.
Each operator produces a typed result containing per-field `CheckResult` entries.

```mermaid
classDiagram
    class Comparison
    class IsClose~ObsT~ {
        +__call__(a, b) IsCloseResult
    }
    class IsLess~ObsT~ {
        +__call__(a, b) IsLessResult
    }
    Comparison <|-- IsClose
    Comparison <|-- IsLess

    class CheckResult {
        +bool passed
        +str detail
    }
    class IsCloseResult {
        +bool is_satisfied
    }
    class IsLessResult {
        +bool is_satisfied
    }

    IsClose ..> IsCloseResult : produces
    IsLess ..> IsLessResult : produces
    IsCloseResult *-- CheckResult
    IsLessResult *-- CheckResult
```

### Constraint Hierarchy

`Constraint` is the abstract base for all checks.
`ComparisonConstraint` subclasses compare two live observations against each other.
`RegressionConstraint` subclasses compare a current observation against a previously saved reference.

```mermaid
classDiagram
    class Constraint {
        <<abstract>>
        +str description
        +str comment
        +required_observables() frozenset
        +error_result(error)
    }
    class ComparisonConstraint {
        <<abstract>>
        +is_satisfied(observations)
    }
    class IsCloseConstraint {
        +IsClose comparison
    }
    class IsLessConstraint {
        +IsLess comparison
    }
    class RegressionConstraint {
        <<abstract>>
        +IsClose comparison
        +evaluate(current, reference)
    }
    class IsCloseResult {
        +bool is_satisfied
    }
    class IsLessResult {
        +bool is_satisfied
    }

    Constraint <|-- ComparisonConstraint
    Constraint <|-- RegressionConstraint
    ComparisonConstraint <|-- IsCloseConstraint
    ComparisonConstraint <|-- IsLessConstraint
    IsCloseConstraint ..> IsCloseResult : produces
    IsLessConstraint ..> IsLessResult : produces
    RegressionConstraint ..> IsCloseResult : produces
```

## Concrete Classes

The following notes document the concrete classes used in the constraints tool organized by `Observation` type.

### Datum

A `DatumObservation` stores the output of a tao datum.
These can be defined and evaluated on the fly using a `DatumObservable`.

#### Observations, Observables, and Operators

```mermaid
flowchart TD
    Observable([Observable]) --> LatticeObservable([LatticeObservable])
    Observable --> LiteralObservable([LiteralObservable])
    LatticeObservable --> DatumObservable[DatumObservable]
    LiteralObservable --> DatumLiteral[DatumLiteral]
    Comparison([Comparison]) --> IsClose([IsClose])
    Comparison --> IsLess([IsLess])
    IsClose --> DatumIsClose[DatumIsClose]
    IsLess --> DatumLessThan[DatumLessThan]
    DatumObservable -. creates .-> DatumObservation[DatumObservation]
    DatumLiteral -. creates .-> DatumObservation
    DatumIsClose -. creates .-> DatumIsCloseResult[DatumIsCloseResult]
    DatumLessThan -. creates .-> DatumLessThanResult[DatumLessThanResult]
```

#### Constraints, and Results

```mermaid
flowchart TD
    Constraint([Constraint]) --> ComparisonConstraint([ComparisonConstraint])
    Constraint --> RegressionConstraint([RegressionConstraint])
    ComparisonConstraint --> IsCloseConstraint([IsCloseConstraint])
    ComparisonConstraint --> IsLessConstraint([IsLessConstraint])
    IsCloseConstraint --> DatumIsCloseConstraint[DatumIsCloseConstraint]
    IsLessConstraint --> DatumLessThanConstraint[DatumLessThanConstraint]
    RegressionConstraint --> DatumRegressionConstraint[DatumRegressionConstraint]
    DatumIsCloseConstraint -. creates .-> DatumIsCloseResult[DatumIsCloseResult]
    DatumLessThanConstraint -. creates .-> DatumLessThanResult[DatumLessThanResult]
    DatumRegressionConstraint -. creates .-> DatumIsCloseResult
```

### Element

An `EleObservation` contains the output of a `tao.ele(...)` call (ie Twiss parameters, reference energy, floor positions, etc.).
The observation may be evaluted from a single element in a lattice with `EleObservable`.
The min and max of the values in the element can be evaluated using `EleMinObservable` and `EleMaxObservable`.

#### Observations, Observables, and Operators

```mermaid
flowchart TD
    Observable([Observable]) --> LatticeObservable([LatticeObservable])
    Observable --> LiteralObservable([LiteralObservable])
    LatticeObservable --> EleObservable[EleObservable]
    LatticeObservable --> EleMaxObservable[EleMaxObservable]
    LatticeObservable --> EleMinObservable[EleMinObservable]
    LiteralObservable --> EleLiteral[EleLiteral]
    Comparison([Comparison]) --> IsClose([IsClose])
    Comparison --> IsLess([IsLess])
    IsClose --> EleIsClose[EleIsClose]
    IsLess --> EleLessThan[EleLessThan]
    EleObservable -. creates .-> EleObservation[EleObservation]
    EleMaxObservable -. creates .-> EleObservation
    EleMinObservable -. creates .-> EleObservation
    EleLiteral -. creates .-> EleObservation
    EleIsClose -. creates .-> EleIsCloseResult[EleIsCloseResult]
    EleLessThan -. creates .-> EleLessThanResult[EleLessThanResult]
```

#### Constraints, and Results

```mermaid
flowchart TD
    Constraint([Constraint]) --> ComparisonConstraint([ComparisonConstraint])
    Constraint --> RegressionConstraint([RegressionConstraint])
    ComparisonConstraint --> IsCloseConstraint([IsCloseConstraint])
    ComparisonConstraint --> IsLessConstraint([IsLessConstraint])
    IsCloseConstraint --> EleIsCloseConstraint[EleIsCloseConstraint]
    IsLessConstraint --> EleLessThanConstraint[EleLessThanConstraint]
    RegressionConstraint --> EleRegressionConstraint[EleRegressionConstraint]
    EleIsCloseConstraint -. creates .-> EleIsCloseResult[EleIsCloseResult]
    EleLessThanConstraint -. creates .-> EleLessThanResult[EleLessThanResult]
    EleRegressionConstraint -. creates .-> EleIsCloseResult
```
