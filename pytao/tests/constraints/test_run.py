import pathlib
import tempfile

from pytao.constraints.config import (
    ConstraintsConfig,
    DatumIsCloseConstraint,
    DatumLessThanConstraint,
    DatumRegressionConstraint,
    EleIsCloseConstraint,
    EleLessThanConstraint,
)
from pytao.constraints.main import run
from pytao.constraints.observables.datum import (
    DatumIsCloseResult,
    DatumLessThanResult,
    DatumLiteral,
    DatumObservable,
    DatumObservation,
)
from pytao.constraints.observables.ele import (
    EleIsCloseResult,
    EleLessThanResult,
    EleLiteral,
    EleObservable,
)
from pytao.constraints.results import SavedEntry, SavedObservations
from pytao.startup import TaoStartup

DATA_DIR = pathlib.Path(__file__).parent / "data"
LAT_A = DATA_DIR / "lattices" / "lat_a.lat.bmad"


def test_run_timing():
    obs = EleObservable(lattice_id="lat_a", ele_id="BEGINNING")
    config = ConstraintsConfig(
        lattices={"lat_a": TaoStartup(lattice_file=LAT_A, noinit=False)},
        constraints=[EleIsCloseConstraint(obs_a=obs, obs_b=obs)],
    )
    saved, results = run(config, DATA_DIR)
    lat = results.lattices["lat_a"]
    assert lat.loaded
    assert not lat.error
    assert lat.load_time > 0
    assert lat.obs_time >= 0
    assert saved.entries[0].observation.elapsed_time >= 0


def test_run_description_comment():
    config = ConstraintsConfig(
        lattices={},
        constraints=[
            DatumIsCloseConstraint(
                description="first",
                comment="first comment",
                obs_a=DatumLiteral(model_value=1.0, design_value=0.0),
                obs_b=DatumLiteral(model_value=1.0, design_value=0.0),
            ),
            DatumIsCloseConstraint(
                description="second",
                comment="second comment",
                obs_a=DatumLiteral(model_value=2.0, design_value=0.0),
                obs_b=DatumLiteral(model_value=2.0, design_value=0.0),
            ),
        ],
    )
    _, results = run(config, DATA_DIR)
    crs = results.constraints[None]
    assert crs[0].description == "first"
    assert crs[0].comment == "first comment"
    assert crs[1].description == "second"
    assert crs[1].comment == "second comment"


def test_run_lattice_load_failure():
    bad_path = DATA_DIR / "lattices" / "nonexistent.lat.bmad"
    obs_bad = EleObservable(lattice_id="lat_bad", ele_id="BEGINNING")
    obs_lat_a = EleObservable(lattice_id="lat_a", ele_id="BEGINNING")
    lit = EleLiteral(beta_a=3.0, alpha_a=-0.5, beta_b=8.0, alpha_b=2.0)
    config = ConstraintsConfig(
        lattices={
            "lat_bad": TaoStartup(lattice_file=bad_path, noinit=False),
            "lat_a": TaoStartup(lattice_file=LAT_A, noinit=False),
        },
        constraints=[
            EleIsCloseConstraint(obs_a=obs_bad, obs_b=lit, description="bad"),
            EleIsCloseConstraint(obs_a=obs_lat_a, obs_b=obs_lat_a, description="good"),
        ],
    )
    _, results = run(config, DATA_DIR)
    assert not results.lattices["lat_bad"].loaded
    assert results.lattices["lat_bad"].error
    assert results.lattices["lat_a"].loaded
    assert not results.lattices["lat_a"].error
    assert sum(len(v) for v in results.constraints.values()) == 2
    bad_cr = next(cr for _, cr in results.iter_constraints() if cr.description == "bad")
    assert not bad_cr.result.is_satisfied
    assert bad_cr.result.error is not None
    good_cr = next(cr for _, cr in results.iter_constraints() if cr.description == "good")
    assert good_cr.result.is_satisfied


def test_run_continues_on_invalid_element():
    obs_valid = EleObservable(lattice_id="lat_a", ele_id="BEGINNING")
    config = ConstraintsConfig(
        lattices={"lat_a": TaoStartup(lattice_file=LAT_A, noinit=False)},
        constraints=[
            EleIsCloseConstraint(
                obs_a=EleObservable(lattice_id="lat_a", ele_id="FAKE1"), obs_b=obs_valid
            ),
            EleIsCloseConstraint(
                obs_a=EleObservable(lattice_id="lat_a", ele_id="FAKE2"), obs_b=obs_valid
            ),
            EleLessThanConstraint(
                obs_a=EleObservable(lattice_id="lat_a", ele_id="FAKE3"), obs_b=obs_valid
            ),
            EleLessThanConstraint(
                obs_a=EleObservable(lattice_id="lat_a", ele_id="FAKE4"), obs_b=obs_valid
            ),
            EleIsCloseConstraint(obs_a=obs_valid, obs_b=obs_valid, description="valid"),
        ],
    )
    _, results = run(config, DATA_DIR)
    assert results.lattices["lat_a"].loaded
    assert not results.lattices["lat_a"].error
    crs = results.constraints[None]
    assert len(crs) == 5
    for cr in crs[:4]:
        assert not cr.result.is_satisfied
        assert cr.result.error is not None
    assert crs[4].description == "valid"
    assert crs[4].result.is_satisfied


def test_run_error_result_types():
    obs_ele = EleObservable(lattice_id="lat_a", ele_id="FAKE")
    obs_dat = DatumObservable(
        lattice_id="lat_a", data_type="nonexistent_datum", ele_name="END"
    )
    lit_ele = EleLiteral(beta_a=3.0, alpha_a=-0.5, beta_b=8.0, alpha_b=2.0)
    lit_dat = DatumLiteral(model_value=1.0, design_value=0.0)
    config = ConstraintsConfig(
        lattices={"lat_a": TaoStartup(lattice_file=LAT_A, noinit=False)},
        constraints=[
            EleIsCloseConstraint(obs_a=obs_ele, obs_b=lit_ele),
            EleLessThanConstraint(obs_a=obs_ele, obs_b=lit_ele),
            DatumIsCloseConstraint(obs_a=obs_dat, obs_b=lit_dat),
            DatumLessThanConstraint(obs_a=obs_dat, obs_b=lit_dat),
        ],
    )
    _, results = run(config, DATA_DIR)
    crs = results.constraints[None]
    assert len(crs) == 4
    assert isinstance(crs[0].result, EleIsCloseResult)
    assert not crs[0].result.is_satisfied
    assert crs[0].result.error is not None
    assert isinstance(crs[1].result, EleLessThanResult)
    assert not crs[1].result.is_satisfied
    assert crs[1].result.error is not None
    assert isinstance(crs[2].result, DatumIsCloseResult)
    assert not crs[2].result.is_satisfied
    assert crs[2].result.error is not None
    assert isinstance(crs[3].result, DatumLessThanResult)
    assert not crs[3].result.is_satisfied
    assert crs[3].result.error is not None


def test_run_saved_observations():
    obs_a = EleObservable(lattice_id="lat_a", ele_id="BEGINNING")
    obs_b = EleLiteral(beta_a=3.0, alpha_a=-0.5, beta_b=8.0, alpha_b=2.0)
    config = ConstraintsConfig(
        lattices={"lat_a": TaoStartup(lattice_file=LAT_A, noinit=False)},
        constraints=[EleIsCloseConstraint(obs_a=obs_a, obs_b=obs_b)],
    )
    saved, _ = run(config, DATA_DIR)
    assert len(saved.entries) == 1
    assert saved.entries[0].observable == obs_a
    assert saved.entries[0].observation.element is not None


def test_run_regression_no_compare():
    obs = DatumLiteral(model_value=1.0, design_value=0.0)
    config = ConstraintsConfig(
        lattices={},
        constraints=[DatumRegressionConstraint(obs=obs)],
    )
    _, results = run(config, DATA_DIR)
    assert list(results.iter_regression()) == []


def test_run_regression_missing_observation():
    obs = DatumLiteral(model_value=1.0, design_value=0.0)
    config = ConstraintsConfig(
        lattices={},
        constraints=[DatumRegressionConstraint(obs=obs)],
    )
    compare = SavedObservations(entries=[])
    _, results = run(config, DATA_DIR, compare=compare)
    reg = list(results.iter_regression())
    assert len(reg) == 1
    _, rr = reg[0]
    assert not rr.result.is_satisfied
    assert rr.result.error is not None


def test_run_regression_multiple_groups():
    obs_pass = DatumLiteral(model_value=1.0, design_value=0.0)
    obs_fail = DatumLiteral(model_value=2.0, design_value=0.0)
    config = ConstraintsConfig(
        lattices={},
        constraints={
            "A": [
                DatumRegressionConstraint(obs=obs_pass, description="passes", comment="note")
            ],
            "B": [DatumRegressionConstraint(obs=obs_fail)],
        },
    )
    compare = SavedObservations(
        entries=[
            SavedEntry(observable=obs_pass, observation=obs_pass.observe()),
            SavedEntry(
                observable=obs_fail,
                observation=DatumObservation(model_value=999.0, design_value=0.0),
            ),
        ]
    )
    _, results = run(config, DATA_DIR, compare=compare)
    assert sum(len(v) for v in results.regression.values()) == 2
    by_group = {rr.group: rr for _, rr in results.iter_regression()}
    a = by_group["A"]
    assert a.result.is_satisfied
    assert a.label == obs_pass.label
    assert a.description == "passes"
    assert a.comment == "note"
    assert a.group == "A"
    b = by_group["B"]
    assert not b.result.is_satisfied
    assert b.group == "B"


def test_saved_observations_round_trip_datum():
    obs = DatumLiteral(model_value=1.5, design_value=0.5)
    observation = obs.observe()
    saved = SavedObservations(entries=[SavedEntry(observable=obs, observation=observation)])
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "obs.json"
        path.write_text(saved.model_dump_json(indent=2))
        loaded = SavedObservations.model_validate_json(path.read_text())
    assert saved == loaded


def test_saved_observations_round_trip_ele():
    obs = EleObservable(lattice_id="lat_a", ele_id="BEGINNING")
    config = ConstraintsConfig(
        lattices={"lat_a": TaoStartup(lattice_file=LAT_A, noinit=False)},
        constraints=[EleIsCloseConstraint(obs_a=obs, obs_b=obs)],
    )
    saved, _ = run(config, DATA_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "obs.json"
        path.write_text(saved.model_dump_json(indent=2))
        loaded = SavedObservations.model_validate_json(path.read_text())
    assert saved == loaded
