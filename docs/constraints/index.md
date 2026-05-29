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
```console title="Running the constraints tool"
pytao-user$ pytao-constraints config.yml
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
                         [--compare-path FILE] [--markdown]
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