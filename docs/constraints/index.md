# Bmad Lattice Constraint Checker Tool

Pytao includes a CLI / configuration-file-based tool to define and then check constraints among a set of Bmad lattices.
The values computed from the lattices may be saved and then loaded as a reference for further runs for regression testing.

## Quickstart

Constraints and regression tests are defined through a [YAML](https://en.wikipedia.org/wiki/YAML) configuration file.
Your Bmad lattice files are defined at the top followed by an (optionally grouped) list of constraints.
Each of the constraints has a type which determines the check which will run and some observables (things to measure from the lattice or a literal value).
In this configuration file, we define an approximate equality constraint (`ele_eq`) between two elements in two lattices (`ele` observable).

```yaml title="config.yml"
lattices:
  lat_a:
    lattice_file: lat_a.lat.bmad
  lat_b:
    lattice_file: lat_b.lat.bmad

constraints:
  - constraint_type: ele_eq
    description: My first equality constraint!
    comment: Longer text describing the constraint for documentation purposes
    obs_a:
      obs_type: ele
      lattice_id: lat_a
      ele_id: element_a
    obs_b:
      obs_type: ele
      lattice_id: lat_b
      ele_id: element_b
```

The constraints tool is run from the terminal with the following command.
```console
$ pytao-constraints config.yml
Lattices:
  [OK  ] lat_a  loaded in 1.53s, observables in 0.02s
  [OK  ] lat_b  loaded in 1.38s, observables in 0.02s

Constraints:
  [PASS] lat_a[element_a] == lat_b[element_b]  My first equality constraint!

1/1 constraints passed
```

## CLI Tool

Constraints checking happens through the tool `pytao-constraints`.

```console
$ pytao-constraints --help
usage: pytao-constraints [-h] [--save-observations FILE] [--save-results FILE]
                         [--compare-path FILE] [--markdown] [--log-file FILE]
                         [--log-level LEVEL]
                         config

Run pytao constraints checks against Bmad lattice files.
```

### Arguments

- `config` — Path to YAML configuration file

### Options

- `-h`, `--help` — Show the help message and exit
- `--save-observations FILE` — Write a JSON snapshot of current observations to `FILE`
- `--save-results FILE` — Write a JSON snapshot of the results to `FILE`
- `--compare-path FILE` — Path to a previously saved observations JSON for regression comparison
- `--markdown` — Emit GitHub-flavored markdown suitable for `GITHUB_STEP_SUMMARY`
- `--log-file FILE` — Write pytao/Tao log output to `FILE`
- `--log-level LEVEL` — Log level for `--log-file` (one of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`; default `INFO`)

## Example

A complete example with three lattices is available for download.
These demonstrate a variety of constraints defined over multiple lattices including failing constraints to observe what they look like in the tool's output.

**Constraint files:**

- [constraints.yaml](example/constraints.yaml) — simple list of constraints covering all constraint types
- [constraints_grouped.yaml](example/constraints_grouped.yaml) — constraints organized into named groups with a regression test

**Lattice files:**

- [lat_a.lat.bmad](example/lat_a.lat.bmad)
- [lat_b.lat.bmad](example/lat_b.lat.bmad)
- [lat_c.lat.bmad](example/lat_c.lat.bmad)

After downloading all files into the same directory, run the following for the list example.

```console
$ pytao-constraints constraints.yaml
Lattices:
  [OK  ] lat_a  loaded in 1.83s, observables in 0.03s
  [OK  ] lat_b  loaded in 1.36s, observables in 0.01s
  [OK  ] lat_c  loaded in 1.41s, observables in 0.01s

Constraints:
  [PASS] lat_a[END] == lat_b[BEGINNING]  Short description, lat_a, lat_b treaty point
  [FAIL] lat_a[END] == lat_c[BEGINNING]  Example of a failing constraint
  [PASS] lat_a[max] < literal  lat_a beta sanity check
  [PASS] lat_a[r56_compaction@END] == literal  Datum and literal example

============================================================
FAILURES
============================================================

  lat_a[END] == lat_c[BEGINNING]  Example of a failing constraint
  --------------------------------------------------------
    twiss_a     PASS
    twiss_b     FAIL  bmag=1.0862 outside [0.99, 1.01]  (beta0=4.625, alpha0=1.375)  (beta1=3.625, alpha1=1.375)
    eta_x       PASS
    etap_x      PASS
    eta_y       PASS
    etap_y      PASS
    ref_energy  FAIL  a=5e+09, b=4e+09, diff=1.000e+09
    p0c         FAIL  a=5e+09, b=4e+09, diff=1.000e+09
    orbit       PASS
    floor_x     PASS
    floor_y     PASS
    floor_z     FAIL  a=1, b=0, diff=1.000e+00

3/4 constraints passed
```

Run the following for the grouped example.
```console
$ pytao-constraints constraints_grouped.yaml 
Lattices:
  [OK  ] lat_a  loaded in 1.55s, observables in 0.03s
  [OK  ] lat_b  loaded in 1.38s, observables in 0.02s
  [OK  ] lat_c  loaded in 1.37s, observables in 0.01s

Constraints:
  [Lattice-Consistency]
    [PASS] lat_a[END] == lat_b[BEGINNING]  Short description, lat_a, lat_b treaty point
    [PASS] lat_a[END]  Trivial test
  [Sanity-Checks]
    [PASS] lat_a[max] < literal  lat_a beta sanity check
    [PASS] lat_b[max] < literal  lat_b beta sanity check
    [FAIL] lat_c[max] < literal  lat_c beta sanity check

============================================================
FAILURES
============================================================

  [Sanity-Checks] lat_c[max] < literal  lat_c beta sanity check
  This constraint is expected to fail
  --------------------------------------------------------
    beta_a  FAIL  a=6.66667 not < b=0
    beta_b  PASS

4/5 constraints passed
```