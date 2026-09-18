# Configuration File

On this page, we document the configuration file used in the CLI tool.

## General Structure
The constraints file consists of two main sections and one optional section.
A `lattices` block used to define the Bmad lattices probed by the tool
Each lattice has a `lattice_id` associated with it which is how it will be referenced by the constraints later on.
It is helpful to pick descriptive names in the same vein as the naming of variables.
The lattice is configured by the fields in  `TaoStartup` which are listed below the `lattice_id`.
These are passed to `Tao` when the lattice is loaded.

```yaml
lattices:
  lat_a:
    lattice_file: ...
    init_file: ...
```

Note: paths in the lattice section are resolved relative to the config file.

The second important section in the config file is the `constraints` section.
There are two ways that constraints may be defined.
For smaller projects, a simple list of the constraints will suffice.
```yaml
constraints:
  - type: ...
    description: ...
    comment: ...
```

To enable organization of constraints when the number is large, they may also be grouped together. 
When the constraints tool is run, these groups will be propagated to the results and output will also be grouped.
```yaml
constraints:
  Treaty-Points:
    - type: ...
      description: ...
      comment: ...
  Sanity-Checks:
    - type: ...
      description: ...
      comment: ...
```

The optional `comparisons` section holds reusable comparison operator settings which constraints may reference by name.
It is described in [Shared Comparisons](#shared-comparisons) below.

Every polymorphic object in the config file (constraints, observables, comparison operators, and Twiss comparison methods) is selected with the same `type` key.

## Constraints

Constraints are distinguished by the `type` field and a complete listing of all constraint types can be found in the [API documentation](api/index.md).
All constraints allow the fields `description` and `comment` which allow for human-readable documentation of the constraints.
A `description` is a short string of text which is included in the constraint tool's output whenever the constraint is referenced.
The `comment` is meant to be a longer description or explanation of what the constraint does and is output on violation of the constraint (as well as being included in saved artifacts).

We will quickly explore the element approximate equality constraint (`EleIsCloseConstraint`) as a prototypical example of defining constraints.
After writing the `type` (in this case `ele_eq` from [API docs](api/ele.md#pytao.constraints.config.EleIsCloseConstraint)), we also define the two observables it acts on and (optionally) any configuration for the comparison operator.
Constraints are organized by `Observation` type and they accept any observable that outputs the right object.

For an `EleIsCloseConstraint` we have a choice of several observables: `ele`, which evaluates the properties of a single element; `ele_literal`, which is a user-defined value; and `ele_reduce`, which reduces each property over the tracking elements of a single lattice.
The reduction is chosen with the `operator` field, which accepts `min`, `max`, or `avg`, and may be restricted to a range of elements with the optional `begin_ele` and `end_ele` fields.
Observables are distinguished by `type` and the correct values can be found in the [API docs](api/ele.md#observables).
We also define the other fields in the chosen observable. 
For example, for the `EleObservable` we must reference the lattice and element to perform the measurement on.
In the `ele_literal` observable, we list the values of the properties we are defining.

Finally, the comparison operator may be configured. 
Tolerances can be set for the approximate comparison operator.
Checks can also be disabled or enabled as needed (for example, enabling floor comparisons or disabling checks against literal fields that are not defined).

```yaml
# Fields common to all constraints
type: ele_eq
description: Short description, used in labels
comment: Longer comment included on error and in reports.

# First observable, a measurement on element `END` in lattice `lat_a`
obs_a:
    type: ele
    lattice_id: lat_a
    ele_id: END

# Second observable, a literal value
obs_b:
    type: ele_literal
    beta_a: 16
    beta_b: 32
    floor_z: 42

# Configuration for the comparison operator: EleIsClose
comparison:
    twiss_a:
        type: bmag
        max_bmag: 1.05
    twiss_b: null
    floor_z:
        rtol: 1e-3
        atol: 1e-6
    ...
```

The result of any comparison is a `ComparisonResult`.
It holds an optional `error` string (set when an observation was missing or Tao failed) and a `checks` mapping from field name to a per-field `CheckResult` with `passed` and `detail` entries.
A constraint is satisfied when there is no error and every check passed.

## Shared Comparisons

When many constraints use the same comparison settings, the settings may be defined once in the top-level `comparisons` section and referenced by name from the `comparison` field of a constraint.
Each entry is a comparison operator selected by `type` (for example `ele_is_close`, `ele_is_less`, `datum_is_close`, or `datum_is_less`) and must match the operator expected by the constraint that references it.

```yaml
comparisons:
  beta_only:
    type: ele_is_less
    beta_a: true
    beta_b: true

constraints:
  - type: ele_lt
    description: lat_a beta sanity check
    obs_a:
      type: ele_reduce
      operator: max
      lattice_id: lat_a
    obs_b:
      type: ele_literal
      beta_a: 200
      beta_b: 200
    comparison: beta_only
```

## Regression Tests

The constraints tool enables regression tests through the saving of all observations made during the checks and then loading them in subsequent runs.
These will show up in a separate section of the output and results artifact.
The approximate equality constraints (such as `EleIsClose` and `DatumIsClose`) inherently define regression tests for their observables as they include a equality operator letting us compare to past results.
Implicit regression tests can be turned off with `regression_check: false` in the constraint.
Users may optionally define their own regression checks at additional points using the `RegressionConstraint` objects.
These are constraints in their own right and are defined in the constraints section, but only have one observable and don't get run unless a comparison file is provided.

The following is an example of an element regression test.
```yaml
type: ele_reg
description: End of lat_a
comment: Check for changes at the end of lat_a
obs:
    type: ele
    lattice_id: lat_a
    ele_id: END
```

## Full Example

Finally, all of this comes together in a full config file.
We give an example below.
```yaml title="config.yml"
lattices:
  lat_a:
    lattice_file: lattices/lat_a.lat.bmad
  lat_b:
    lattice_file: lattices/lat_b.lat.bmad
  lat_c:
    lattice_file: lattices/lat_c.lat.bmad

constraints:
  - type: ele_eq
    description: Short description, lat_a, lat_b treaty point
    comment: Longer comment included on error and in reports. eg Please rerun `tune_matcher.tao` in event of failure.
    obs_a:
      type: ele
      lattice_id: lat_a
      ele_id: END
    obs_b:
      type: ele
      lattice_id: lat_b
      ele_id: BEGINNING

  - type: ele_lt
    description: lat_a beta sanity check
    obs_a:
      type: ele_reduce
      operator: max
      lattice_id: lat_a
    obs_b:
      type: ele_literal
      beta_a: 200
      beta_b: 200
    comparison:
      beta_a: true
      beta_b: true

  - type: datum_eq
    description: Datum and literal example
    obs_a:
      type: datum
      lattice_id: lat_a
      data_type: r56_compaction
      ele_name: END
    obs_b:
      type: datum_literal
      model_value: 1.04448e-08
      design_value: 0.0

```
