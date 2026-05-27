# Bmad Lattice Constraint Checker Tool

Pytao includes a CLI / configuration-file-based tool to define and then check constraints among a set of Bmad lattices.
The values computed from the lattices may be saved and then loaded as a reference for further runs for regression testing.

## Class Structure

### Base Classes

#### Observations, Observables, and Operators

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

    class Comparison
    class IsClose~ObsT~ {
        +__call__(a, b) IsCloseResult
    }
    class IsLess~ObsT~ {
        +__call__(a, b) IsLessResult
    }
    Comparison <|-- IsClose
    Comparison <|-- IsLess

    LatticeObservable ..> Observation : creates
    LiteralObservable ..> Observation : creates
    IsClose ..> Observation : operates on
    IsLess ..> Observation : operates on
```

#### Constraints and Results

```mermaid
classDiagram
    class Constraint {
        <<abstract>>
        +str description
        +str comment
        +required_observables() frozenset
        +is_satisfied(observations) ComparisonResult
        +error_result(error) ComparisonResult
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
    class ComparisonResult {
        <<abstract>>
        +str error
    }
    class IsCloseResult {
        +bool is_close
    }
    class IsLessResult {
        +bool is_less
    }
    ComparisonResult <|-- IsCloseResult
    ComparisonResult <|-- IsLessResult

    EqualityConstraint ..> IsCloseResult : produces
    IsLessConstraint ..> IsLessResult : produces
    IsCloseResult *-- CheckResult
    IsLessResult *-- CheckResult
```

### Concrete Classes

#### Datum

```mermaid
classDiagram
    class Observation
    class LatticeObservable~ObsT~
    class LiteralObservable~ObsT~
    class IsClose~ObsT~
    class IsLess~ObsT~

    class DatumObservation {
        +float model_value
        +float design_value
    }
    class DatumObservable {
        +str data_type
        +str ele_name
        +str data_source
    }
    class DatumLiteral {
        +float model_value
        +float design_value
    }
    class DatumIsClose {
        +TolComparison model_value_test
        +TolComparison design_value_test
    }
    class DatumLessThan {
        +bool model_value
        +bool design_value
    }

    Observation <|-- DatumObservation
    LatticeObservable <|-- DatumObservable
    LiteralObservable <|-- DatumLiteral
    IsClose <|-- DatumIsClose
    IsLess <|-- DatumLessThan
```

```mermaid
classDiagram
    class EqualityConstraint
    class IsLessConstraint

    class DatumIsCloseConstraint {
        +DatumObservable obs_a
        +DatumObservable obs_b
        +DatumIsClose comparison
    }
    class DatumLessThanConstraint {
        +DatumObservable obs_a
        +DatumObservable obs_b
        +DatumLessThan comparison
    }
    class DatumIsCloseResult {
        +CheckResult model_value
        +CheckResult design_value
    }
    class DatumLessThanResult {
        +CheckResult model_value
        +CheckResult design_value
    }

    EqualityConstraint <|-- DatumIsCloseConstraint
    IsLessConstraint <|-- DatumLessThanConstraint
    DatumIsCloseConstraint ..> DatumIsCloseResult : creates
    DatumLessThanConstraint ..> DatumLessThanResult : creates
```

#### Ele

```mermaid
classDiagram
    class Observation
    class LatticeObservable~ObsT~
    class LiteralObservable~ObsT~
    class IsClose~ObsT~
    class IsLess~ObsT~

    class EleObservation {
        +Element element
    }
    class EleObservable {
        +str ele_id
    }
    class EleMaxObservable
    class EleMinObservable
    class EleLiteral {
        +float beta_a
        +float alpha_a
        +...
    }
    class EleIsClose {
        +TwissComparisonMethod twiss_a_test
        +TwissComparisonMethod twiss_b_test
        +TolComparison eta_x_test
        +...
    }
    class EleLessThan {
        +bool beta_a
        +bool alpha_a
        +...
    }

    Observation <|-- EleObservation
    LatticeObservable <|-- EleObservable
    LatticeObservable <|-- EleMaxObservable
    LatticeObservable <|-- EleMinObservable
    LiteralObservable <|-- EleLiteral
    IsClose <|-- EleIsClose
    IsLess <|-- EleLessThan
```

```mermaid
classDiagram
    class EqualityConstraint
    class IsLessConstraint

    class EleIsCloseConstraint {
        +EleObservable obs_a
        +EleObservable obs_b
        +EleIsClose comparison
    }
    class EleLessThanConstraint {
        +EleObservable obs_a
        +EleObservable obs_b
        +EleLessThan comparison
    }
    class EleIsCloseResult {
        +CheckResult twiss_a
        +CheckResult twiss_b
        +CheckResult eta_x
        +...
    }
    class EleLessThanResult {
        +CheckResult beta_a
        +CheckResult alpha_a
        +...
    }

    EqualityConstraint <|-- EleIsCloseConstraint
    IsLessConstraint <|-- EleLessThanConstraint
    EleIsCloseConstraint ..> EleIsCloseResult : creates
    EleLessThanConstraint ..> EleLessThanResult : creates
```
