# Unreleased

## New Features

### Read-only optimization introspection (`pytao.optimize`)

A new subpackage that reads a live `Tao` instance and reports its
optimization setup as structured Python objects — the variables and datums
Tao would use for optimization, plus their bounds, weights, and merit types.

```python
from pytao import Tao
from pytao.optimize import TaoOptimizationProblem

tao = Tao(init_file="tao.init", noplot=True)
problem = TaoOptimizationProblem(tao)

problem.n_var           # number of active optimization variables
problem.n_data          # number of active datums
problem.x0              # initial variable vector
problem.bounds          # list of (low, high) tuples; ±inf allowed
problem.weights         # per-datum merit weights
problem.variables       # list[VariableInfo]
problem.datums          # list[DatumInfo]
```

Highlights:

- Filters to `useit_opt = True` entries, i.e. what Tao itself would pass to
  its internal optimizers.
- Translates Tao's ±1e30 "no limit" sentinels to ±inf so downstream
  optimizer libraries see standard unbounded-side semantics.
- Rejects negative weights at construction — catches a class of user errors
  at the boundary rather than silently producing NaNs later.
- Multi-universe aware: with no argument, `TaoOptimizationProblem` reads
  Tao's live `s%global%default_universe`; pass `universe=N` to pin the
  snapshot to a specific universe.
- `problem.variables` and `problem.datums` are immutable tuples of frozen
  dataclasses, so the snapshot cannot drift out of sync with `x0`,
  `bounds`, and `weights`.

This is the first slice of a larger Python-driven optimization workflow.
Follow-up releases will add merit evaluation, Jacobian access, and SciPy
adapters (`run_scipy_minimize`, `run_scipy_least_squares`). See the
[optimization guide](optimize.md) for a walkthrough.

# v1.0.0

## New Features

### Pydantic data models for Tao state (`pytao.model`)

PyTao now provides Pydantic v2 models that represent the live state of Tao.
These models can be queried from Tao, modified in Python, and applied back.

- **`TaoConfig`** — captures the complete configuration of a Tao session:
  startup parameters, `BmadCom`, `SpaceChargeCom`, `BeamInit`, `Beam`,
  `TaoGlobal`, and per-element overrides.

  ```python
  config = tao.get_config()
  config.beam_init.a_emit = 1e-8
  config.set(tao)                    # apply all settings
  config.set(tao, only_changed=True) # apply only what changed
  ```

- **Settings sub-models** — `BmadCom`, `SpaceChargeCom`, `BeamInit`, `Beam`,
  and `TaoGlobal` can also be used independently:

  ```python
  from pytao.model import BeamInit
  beam_init = BeamInit.from_tao(tao)
  beam_init.a_emit = 1e-6
  with beam_init.set_context(tao):
      # temporarily applied; restored on exit
      ...
  ```

- **`Element`** — structured representation of a lattice element with head
  metadata, general attributes, Twiss parameters, orbit, transfer matrix,
  floor coordinates, multipoles, wake fields, chamber walls, and more.
  Data loading is controlled per-field.

  ```python
  ele = tao.ele("Q00W")
  ele.twiss.beta_a
  ele.attrs["k1"].data
  ele.floor.end.actual.x
  ```

- **`Lattice`** — a collection of `Element` objects with lookup by name,
  key, or index. Multiple constructors: `from_tao_tracking`,
  `from_tao_unique`, `from_tao_eles`.

- **`tao.eles()`** — query multiple elements using Tao's full element
  matching syntax (wildcards, ranges, key filters):

  ```python
  quads = tao.eles("quad::*")
  eles_1_to_20 = tao.eles("1:20")
  ```

- **Serialization** — all models support `.write()` / `.from_file()` for JSON,
  `.json.gz`, msgpack, and YAML.
  [`ormsgpack`](https://github.com/ormsgpack/ormsgpack) is used for the
  (recommended and fastest) serialization method [msgpack](https://msgpack.org),
  while `orjson` is used for faster-than-stdlib
  JSON serialization. These are both new requirements.

- **Code generation** — `TaoGlobal`, `BeamInit`, etc. are auto-generated from
  Bmad/Tao structure definitions (`scripts/structs.json`) via `model/codegen.py`.

### ParticleGroup helper

- **`tao.particles()`** — is a new shortcut to get an openPMD-BeamPhysics `ParticleGroup`
  instance for a given element.

  ```python
  P = tao.particles("quad::Q1")  # type: beamphysics.ParticleGroup
  P.plot("x", "px")
  ```

### Session archiving

- **`tao.archive(directory)`** writes a self-contained, reproducible archive:
  the current lattice (via `write bmad`), a Tao command file with all `set`
  commands, and a bash script that re-launches the session.

  ```python
  sh_file, cmd_file = tao.archive("my_archive")
  ```

- **`config.write_bash_loader_script()`** provides lower-level control over
  archive generation (custom Tao binary, prefix, optional lattice export).

### CLI argument handling rework

- `TaoStartup` now parses all Tao command-line arguments the same way Tao
  does, rather than juggling a free-form `init` string alongside individual
  attributes.
- Fixes for `pytao -command` which Tao had silently been ignoring; PyTao
  now handles it correctly.
- All arguments are documented consistently in `pytao --help`.

### Error handling improvements

- More consistent error handling between `SubprocessTao` and `Tao`:
  Tao messages from pipe command output are handled consistently between
  the regular Tao and the subprocess variant.
- Filtered messages from pipe commands are redirected to Python logging.
- Added `PYTAO_LIB_PATH` environment variable to force PyTao to use a
  specific `libtao.so`.

## Other Changes

- Autogenerated interface commands are now a mixin class
  (`_TaoAutogeneratedCommandMixin`), simplifying development.
- Python 3.14 compatibility fixes.
- Better support for `pip`-based (PyPI) installs with proper dependency
  listings.
- Import cleanups and fixes throughout the codebase.
- Type annotations for auto-generated Tao `pipe` interface routines.
- Small performance improvements in subprocess
- Subprocess tests reuse a single `SubprocessTao` instance by default
  (set `TAO_REUSE_SUBPROCESS=1`), speeding up CI.

## Backward Incompatible Changes

- **Python 3.10+ is now required.**
- **Bmad >=20260317** is also now required for all features to behave as designed.
- Removed legacy `tao_interface` and `tao_pexpect`.
- Removed `pytao-gui` entrypoint and `pytao.gui` submodule.
- Removed the largely defunct `as_dict` parameter from all interface commands.
- `TaoStartup.init` attribute was removed. The free-form init string is now
  parsed into individual attributes. It is still accepted as an initializer
  argument: `TaoStartup(init="-lat foo.lat")`.

---

# v0.5.7

## What's Changed

- Fix Derivative Parser by @electronsandstuff in <https://github.com/bmad-sim/pytao/pull/148>

**Full Changelog**: <https://github.com/bmad-sim/pytao/compare/v0.5.6...v0.5.7>

---

# v0.5.6

## What's Changed

- DOC: release process by @ken-lauer in <https://github.com/bmad-sim/pytao/pull/146>
- FIX: handle potential empty string case for pytypes by @ken-lauer in <https://github.com/bmad-sim/pytao/pull/147>

**Full Changelog**: <https://github.com/bmad-sim/pytao/compare/v0.5.5...v0.5.6>

---

# v0.5.5

## What's Changed

- Fix FODO notebook by @ChristopherMayes in <https://github.com/bmad-sim/pytao/pull/145>
- Add parsers for the new `lord_control` and `slave_control` Tao pipe commands by @ChristopherMayes in <https://github.com/bmad-sim/pytao/pull/144>

**Full Changelog**: <https://github.com/bmad-sim/pytao/compare/v0.5.4...v0.5.5>

---

# v0.5.4

## What's Changed

- ENH: add Tao.from_lattice_contents by @ken-lauer in <https://github.com/bmad-sim/pytao/pull/137>
- Added that ACC_ENABLE_SHARED_ONLY=Y is acceptable. by @DavidSagan in <https://github.com/bmad-sim/pytao/pull/142>

## **Full Changelog**: <https://github.com/bmad-sim/pytao/compare/v0.5.3...v0.5.4>

# v0.5.3

## What's Changed

- ENH: unique element ID helpers by @ken-lauer in <https://github.com/bmad-sim/pytao/pull/135>

**Full Changelog**: <https://github.com/bmad-sim/pytao/compare/v0.5.2...v0.5.3>

---

# v0.5.2

## What's Changed

- MAINT: rebuild interface for 'lat_header' by @ken-lauer in <https://github.com/bmad-sim/pytao/pull/134>
- DOC: add floor plan plot with ParticleGroup plot overlays example by @ken-lauer in <https://github.com/bmad-sim/pytao/pull/110>

**Full Changelog**: <https://github.com/bmad-sim/pytao/compare/v0.5.1...v0.5.2>

---

# v0.5.1

## What's Changed

- PERF: PyTao plotting performance has been improved.
- FIX: Overlapping plot regions should no longer affect PyTao's internal plotting mechanism.

### Pull Requests

- PERF: use array buffer methods for plot curve line/symbol data retrieval by @ken-lauer in <https://github.com/bmad-sim/pytao/pull/133>

**Full Changelog**: <https://github.com/bmad-sim/pytao/compare/v0.5.0...v0.5.1>

---

# v0.5.0

## Changes

- Bug fix: Tao refused to initialize after the very first initialization attempt failed.

- Enhanced and reworked the `pytao` command-line interface
  - If in a directory with a `tao.init` file, PyTao can be started with just running `pytao` with no arguments.
  - IPython is used if available, though the regular Python interpreter will be used as a backup.

- For the command-line interface, as in JupyterLab mode, `%tao` and `%%tao` magics allow you to send commands directly to the Tao command-line.
  - For example: `%tao show lat`
- An even shorter shortcut is the input transformer.
  - This shortcut can be customized with the `--pyprefix` argument. Every IPython line that starts with this character will turn into a `tao.cmd()` line. It defaults to the backtick character (i.e., the key typically shared by tilde `~`)

  - For example, noting the backtick:

  ```
  In [1]: ` show lat
  ```

- A preliminary Tao input mode is also available that behaves like the standard Tao command-line. Tab completion is offered for top-level commands and element names (in certain scenarios). To access it, either use `tao.shell()` or the single backtick shortcut:

  ```
  In [1]: `
  Tao> show lat
  ```

- New command-line arguments:
  - `--help` to see PyTao and Tao's command-line options
  - `--pyplot` to configure PyTao's plotting backend (`mpl` or `bokeh`)
  - `--pyscript` to run a Python script after Tao initialization
  - `--pycommand` to run a Python command after Tao initialization
  - `--pylog` set the Python log level
  - `--pyno-subprocess` use `Tao` instead of `SubprocessTao`
  - `--pyquiet` to hide the banner when starting `pytao`
  - `--pytao` to go straight to Tao mode without initializing a `Tao` object

## Issues

- `#130`
- `#129`
- `#128`
- `#101`

## Screenshots

The input transformer:
<img width="692" alt="input transformer" src="https://github.com/user-attachments/assets/db34a299-5ac8-4e03-832b-dcf79e1139be" />

`Tao>` command-line mode:

<img width="575" alt="tao command-line mode" src="https://github.com/user-attachments/assets/8315b522-c972-4e13-ac88-f067af7a56b3" />

Element name tab completion in `Tao>` command-line mode:

<img width="645" alt="tab completion" src="https://github.com/user-attachments/assets/c1a59367-52f3-4ff7-bd60-b2719befcd1a" />

## What's Changed

- ENH: `pytao` CLI rework with tab completion/magics + fixes for `TaoInitializationError` by @ken-lauer in <https://github.com/bmad-sim/pytao/pull/131>

**Full Changelog**: <https://github.com/bmad-sim/pytao/compare/v0.4.8...v0.5.0>

---

# v0.4.8

## What's Changed

- FIX: Tao.track_beam() case issue by @ken-lauer in <https://github.com/bmad-sim/pytao/pull/127>

**Full Changelog**: <https://github.com/bmad-sim/pytao/compare/v0.4.7...v0.4.8>

## Changes

- `pytao` import speed was improved when not using plotting libraries.

- In `Tao.track_beam`:
  - Bug fix: case sensitivity of track_start/track_end elements could cause `Tao.track_beam` to fail.
  - Add support for `Tao.track_beam(start, end)` to set the track start/end positions easily
  - Add support for setting `lattice_calc_on` automatically (and resetting it afterward)

## Issues

- `#126`

---

# v0.4.7

## Changes

- `pytao` now requires [Bmad 20250226](https://github.com/bmad-sim/bmad-ecosystem/releases/tag/20250226-0) or later.
- Added `Tao.track_beam()` which (by default) displays a progress bar while setting `track_type = beam`.
- Added new module `pytao.pbar` which helps with progress-bar related tools
  - For now, this is limited to beam tracking updates with track_beam_wrapper, a context manager which allows you to perform arbitrary Tao commands while it displays beam tracking status
  - Auto-detection of JupyterLab to determine which type of progress bar to use (CLI vs Jupyter widget)

- Bug fix: layout plots may raise an exception in taking the `max()` of an empty `ele_y2s` list.

- Added support for new `-quiet` flags from `Tao`.

## Issues

Closed #119
Closed #120

## Screenshots

- Using PyTao in Jupyter Lab:

<https://github.com/user-attachments/assets/ed987e51-119b-4da0-96f6-b0c20135a451>

- Using PyTao in the terminal:

<https://github.com/user-attachments/assets/667e92bf-08b0-49f7-9abb-ad396ab9a3e2>

# v0.4.8

## Changes

- `pytao` import speed was improved when not using plotting libraries.

- In `Tao.track_beam`:
  - Bug fix: case sensitivity of track_start/track_end elements could cause `Tao.track_beam` to fail.
  - Add support for `Tao.track_beam(start, end)` to set the track start/end positions easily
  - Add support for setting `lattice_calc_on` automatically (and resetting it afterward)

## Issues

Closed #126

# v0.5.0

## Changes

- Bug fix: Tao refused to initialize after the very first initialization
  attempt failed.

- Enhanced and reworked the `pytao` command-line interface
  - If in a directory with a `tao.init` file, PyTao can be started with just
    running `pytao` with no arguments.
  - IPython is used if available, though the regular Python interpreter will
    be used as a backup.

- For the command-line interface, as in JupyterLab mode, `%tao` and `%%tao`
  magics allow you to send commands directly to the Tao command-line.
  - For example: `%tao show lat`
- An even shorter shortcut is the input transformer.
  - This shortcut can be customized with the `--pyprefix` argument. Every
    IPython line that starts with this character will turn into a `tao.cmd()`
    line. It defaults to the backtick character (i.e., the key typically shared by tilde `~`)

  - For example, noting the backtick:

  ```
  In [1]: ` show lat
  ```

- A preliminary Tao input mode is also available that behaves like the
  standard Tao command-line. Tab completion is offered for top-level commands
  and element names (in certain scenarios). To access it, either use
  `tao.shell()` or the single backtick shortcut:

  ```
  In [1]: `
  Tao> show lat
  ```

- New command-line arguments:
  - `--help` to see PyTao and Tao's command-line options
  - `--pyplot` to configure PyTao's plotting backend (`mpl` or `bokeh`)
  - `--pyscript` to run a Python script after Tao initialization
  - `--pycommand` to run a Python command after Tao initialization
  - `--pylog` set the Python log level
  - `--pyno-subprocess` use `Tao` instead of `SubprocessTao`
  - `--pyquiet` to hide the banner when starting `pytao`
  - `--pytao` to go straight to Tao mode without initializing a `Tao` object

## Issues

Closed #130
Closed #129
Closed #128
Closed #101

## Screenshots

The input transformer:

<img width="692" alt="image" src="https://github.com/user-attachments/assets/db34a299-5ac8-4e03-832b-dcf79e1139be" />

`Tao>` command-line mode:

<img width="575" alt="image" src="https://github.com/user-attachments/assets/8315b522-c972-4e13-ac88-f067af7a56b3" />

Element name tab completion in `Tao>` command-line mode:

<img width="645" alt="image" src="https://github.com/user-attachments/assets/c1a59367-52f3-4ff7-bd60-b2719befcd1a" />
