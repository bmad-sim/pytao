# Bmad Lattice Constraint Checker Tool

Pytao includes a CLI / configuration-file-based tool to define and then check constraints among a set of Bmad lattices.
The values computed from the lattices may be saved and then loaded as a reference for further runs for regression testing.

## Class Structure

### Base Classes

#### Observations and Observables

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

#### Operators, Constraints, and Results

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

    class Constraint {
        <<abstract>>
        +str description
        +str comment
        +required_observables() frozenset
        +is_satisfied(observations)
        +error_result(error)
    }
    class EqualityConstraint {
        +IsClose comparison
    }
    class IsLessConstraint {
        +IsLess comparison
    }
    Constraint <|-- EqualityConstraint
    Constraint <|-- IsLessConstraint

    class CheckResult {
        +bool passed
        +str detail
    }
    class IsCloseResult {
        +bool is_close
    }
    class IsLessResult {
        +bool is_less
    }

    EqualityConstraint ..> IsCloseResult : produces
    IsLessConstraint ..> IsLessResult : produces
    IsCloseResult *-- CheckResult
    IsLessResult *-- CheckResult
```

### Concrete Classes

#### Datum

##### Observations, Observables, and Operators

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

##### Constraints, and Results

```mermaid
flowchart TD
    Constraint([Constraint]) --> EqualityConstraint([EqualityConstraint])
    Constraint --> IsLessConstraint([IsLessConstraint])
    EqualityConstraint --> DatumIsCloseConstraint[DatumIsCloseConstraint]
    IsLessConstraint --> DatumLessThanConstraint[DatumLessThanConstraint]
    DatumIsCloseConstraint -. creates .-> DatumIsCloseResult[DatumIsCloseResult]
    DatumLessThanConstraint -. creates .-> DatumLessThanResult[DatumLessThanResult]
```

#### Ele

##### Observations, Observables, and Operators

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

##### Constraints, and Results

```mermaid
flowchart TD
    Constraint([Constraint]) --> EqualityConstraint([EqualityConstraint])
    Constraint --> IsLessConstraint([IsLessConstraint])
    EqualityConstraint --> EleIsCloseConstraint[EleIsCloseConstraint]
    IsLessConstraint --> EleLessThanConstraint[EleLessThanConstraint]
    EleIsCloseConstraint -. creates .-> EleIsCloseResult[EleIsCloseResult]
    EleLessThanConstraint -. creates .-> EleLessThanResult[EleLessThanResult]
```
