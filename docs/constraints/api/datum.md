# Datum Constraints

A `DatumObservation` stores the output of a tao datum.
These can be defined and evaluated on the fly using a `DatumObservable`.

## Observation Classes

```mermaid
flowchart TD
    LatticeObservable([LatticeObservable]) --> Observable([Observable])
    LiteralObservable([LiteralObservable]) --> Observable
    DatumObservable[DatumObservable] --> LatticeObservable
    DatumLiteral[DatumLiteral] --> LiteralObservable
    IsClose([IsClose]) --> Comparison([Comparison])
    IsLess([IsLess]) --> Comparison
    DatumIsClose[DatumIsClose] --> IsClose
    DatumLessThan[DatumLessThan] --> IsLess
    DatumObservable -. creates .-> DatumObservation[DatumObservation]
    DatumLiteral -. creates .-> DatumObservation
    DatumIsClose -. creates .-> DatumIsCloseResult[DatumIsCloseResult]
    DatumLessThan -. creates .-> DatumLessThanResult[DatumLessThanResult]
```


#### ::: pytao.constraints.observables.DatumObservation

### Observables

#### ::: pytao.constraints.observables.DatumObservable
#### ::: pytao.constraints.observables.DatumLiteral

### Operators and Results

#### ::: pytao.constraints.observables.DatumIsClose
#### ::: pytao.constraints.observables.DatumIsCloseResult
#### ::: pytao.constraints.observables.DatumLessThan
#### ::: pytao.constraints.observables.DatumLessThanResult

## Constraints Classes

```mermaid
flowchart TD
    ComparisonConstraint([ComparisonConstraint]) --> Constraint([Constraint])
    RegressionConstraint([RegressionConstraint]) --> Constraint
    IsCloseConstraint([IsCloseConstraint]) --> ComparisonConstraint
    IsLessConstraint([IsLessConstraint]) --> ComparisonConstraint
    DatumIsCloseConstraint[DatumIsCloseConstraint] --> IsCloseConstraint
    DatumLessThanConstraint[DatumLessThanConstraint] --> IsLessConstraint
    DatumRegressionConstraint[DatumRegressionConstraint] --> RegressionConstraint
    DatumIsCloseConstraint -. creates .-> DatumIsCloseResult[DatumIsCloseResult]
    DatumLessThanConstraint -. creates .-> DatumLessThanResult[DatumLessThanResult]
    DatumRegressionConstraint -. creates .-> DatumIsCloseResult
```

#### ::: pytao.constraints.config.DatumIsCloseConstraint
#### ::: pytao.constraints.config.DatumLessThanConstraint
#### ::: pytao.constraints.config.DatumRegressionConstraint
