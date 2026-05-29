# Datum Constraints

A `DatumObservation` stores the output of a tao datum.
These can be defined and evaluated on the fly using a `DatumObservable`.

## Observation Classes

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

#### ::: pytao.constraints.config.DatumIsCloseConstraint

#### ::: pytao.constraints.config.DatumLessThanConstraint

#### ::: pytao.constraints.config.DatumRegressionConstraint
