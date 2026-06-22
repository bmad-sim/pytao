# Element Constraints

An `EleObservation` contains the output of a `tao.ele(...)` call (ie Twiss parameters, reference energy, floor positions, etc.).
The observation may be evaluted from a single element in a lattice with `EleObservable`.
The min and max of the values in the element can be evaluated using `EleMinObservable` and `EleMaxObservable`.

## Observation Classes

```mermaid
flowchart TD
    LatticeObservable([LatticeObservable]) --> Observable([Observable])
    LiteralObservable([LiteralObservable]) --> Observable
    EleObservable[EleObservable] --> LatticeObservable
    EleMaxObservable[EleMaxObservable] --> LatticeObservable
    EleMinObservable[EleMinObservable] --> LatticeObservable
    EleLiteral[EleLiteral] --> LiteralObservable
    IsClose([IsClose]) --> Comparison([Comparison])
    IsLess([IsLess]) --> Comparison
    EleIsClose[EleIsClose] --> IsClose
    EleLessThan[EleLessThan] --> IsLess
    EleObservable -. creates .-> EleObservation[EleObservation]
    EleMaxObservable -. creates .-> EleObservation
    EleMinObservable -. creates .-> EleObservation
    EleLiteral -. creates .-> EleObservation
    EleIsClose -. creates .-> EleIsCloseResult[EleIsCloseResult]
    EleLessThan -. creates .-> EleLessThanResult[EleLessThanResult]
```

#### ::: pytao.constraints.observables.EleObservation

### Observables

#### ::: pytao.constraints.observables.EleObservable
#### ::: pytao.constraints.observables.EleMinObservable
#### ::: pytao.constraints.observables.EleMaxObservable
#### ::: pytao.constraints.observables.EleLiteral

### Operators and Results

#### ::: pytao.constraints.observables.EleIsClose
#### ::: pytao.constraints.observables.EleLessThanResult
#### ::: pytao.constraints.observables.EleLessThan
#### ::: pytao.constraints.observables.EleIsCloseResult

### Operator Helper Classes

#### ::: pytao.constraints.observables.TolComparison
#### ::: pytao.constraints.observables.BmagTwissComparison

## Constraints, and Results

```mermaid
flowchart TD
    ComparisonConstraint([ComparisonConstraint]) --> Constraint([Constraint])
    RegressionConstraint([RegressionConstraint]) --> Constraint
    IsCloseConstraint([IsCloseConstraint]) --> ComparisonConstraint
    IsLessConstraint([IsLessConstraint]) --> ComparisonConstraint
    EleIsCloseConstraint[EleIsCloseConstraint] --> IsCloseConstraint
    EleLessThanConstraint[EleLessThanConstraint] --> IsLessConstraint
    EleRegressionConstraint[EleRegressionConstraint] --> RegressionConstraint
    EleIsCloseConstraint -. creates .-> EleIsCloseResult[EleIsCloseResult]
    EleLessThanConstraint -. creates .-> EleLessThanResult[EleLessThanResult]
    EleRegressionConstraint -. creates .-> EleIsCloseResult
```

#### ::: pytao.constraints.config.EleIsCloseConstraint
#### ::: pytao.constraints.config.EleLessThanConstraint
#### ::: pytao.constraints.config.EleRegressionConstraint
