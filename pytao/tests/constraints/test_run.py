import pathlib

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
        lattices={"lat_a": TaoStartup(lattice_file=LAT_A)},
        constraints=[EleIsCloseConstraint(obs_a=obs, obs_b=obs)],
    )
    saved, results = run(config, DATA_DIR)
    lat = results.lattices["lat_a"]
    assert lat.loaded
    assert lat.error is None
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
    saved, results = run(config, DATA_DIR)
    assert results.constraints[0].description == "first"
    assert results.constraints[0].comment == "first comment"
    assert results.constraints[1].description == "second"
    assert results.constraints[1].comment == "second comment"


def test_run_lattice_load_failure():
    bad_path = DATA_DIR / "lattices" / "nonexistent.lat.bmad"
    obs_bad = EleObservable(lattice_id="lat_bad", ele_id="BEGINNING")
    obs_lat_a = EleObservable(lattice_id="lat_a", ele_id="BEGINNING")
    lit = EleLiteral(beta_a=3.0, alpha_a=-0.5, beta_b=8.0, alpha_b=2.0)
    config = ConstraintsConfig(
        lattices={
            "lat_bad": TaoStartup(lattice_file=bad_path),
            "lat_a": TaoStartup(lattice_file=LAT_A),
        },
        constraints=[
            EleIsCloseConstraint(obs_a=obs_bad, obs_b=lit, description="bad"),
            EleIsCloseConstraint(obs_a=obs_lat_a, obs_b=obs_lat_a, description="good"),
        ],
    )
    saved, results = run(config, DATA_DIR)
    assert not results.lattices["lat_bad"].loaded
    assert results.lattices["lat_bad"].error is not None
    assert results.lattices["lat_a"].loaded
    assert results.lattices["lat_a"].error is None
    assert len(results.constraints) == 2
    bad_cr = next(cr for cr in results.constraints if cr.description == "bad")
    assert not bad_cr.result.is_satisfied
    assert bad_cr.result.error is not None
    good_cr = next(cr for cr in results.constraints if cr.description == "good")
    assert good_cr.result.is_satisfied


def test_run_continues_on_invalid_element():
    obs_valid = EleObservable(lattice_id="lat_a", ele_id="BEGINNING")
    config = ConstraintsConfig(
        lattices={"lat_a": TaoStartup(lattice_file=LAT_A)},
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
    saved, results = run(config, DATA_DIR)
    assert results.lattices["lat_a"].loaded
    assert results.lattices["lat_a"].error is None
    assert len(results.constraints) == 5
    for cr in results.constraints[:4]:
        assert not cr.result.is_satisfied
        assert cr.result.error is not None
    assert results.constraints[4].description == "valid"
    assert results.constraints[4].result.is_satisfied


def test_run_error_result_types():
    obs_ele = EleObservable(lattice_id="lat_a", ele_id="FAKE")
    obs_dat = DatumObservable(
        lattice_id="lat_a", data_type="nonexistent_datum", ele_name="END"
    )
    lit_ele = EleLiteral(beta_a=3.0, alpha_a=-0.5, beta_b=8.0, alpha_b=2.0)
    lit_dat = DatumLiteral(model_value=1.0, design_value=0.0)
    config = ConstraintsConfig(
        lattices={"lat_a": TaoStartup(lattice_file=LAT_A)},
        constraints=[
            EleIsCloseConstraint(obs_a=obs_ele, obs_b=lit_ele),
            EleLessThanConstraint(obs_a=obs_ele, obs_b=lit_ele),
            DatumIsCloseConstraint(obs_a=obs_dat, obs_b=lit_dat),
            DatumLessThanConstraint(obs_a=obs_dat, obs_b=lit_dat),
        ],
    )
    saved, results = run(config, DATA_DIR)
    assert len(results.constraints) == 4
    assert isinstance(results.constraints[0].result, EleIsCloseResult)
    assert not results.constraints[0].result.is_satisfied
    assert results.constraints[0].result.error is not None
    assert isinstance(results.constraints[1].result, EleLessThanResult)
    assert not results.constraints[1].result.is_satisfied
    assert results.constraints[1].result.error is not None
    assert isinstance(results.constraints[2].result, DatumIsCloseResult)
    assert not results.constraints[2].result.is_satisfied
    assert results.constraints[2].result.error is not None
    assert isinstance(results.constraints[3].result, DatumLessThanResult)
    assert not results.constraints[3].result.is_satisfied
    assert results.constraints[3].result.error is not None


def test_run_saved_observations():
    obs_a = EleObservable(lattice_id="lat_a", ele_id="BEGINNING")
    obs_b = EleLiteral(beta_a=3.0, alpha_a=-0.5, beta_b=8.0, alpha_b=2.0)
    config = ConstraintsConfig(
        lattices={"lat_a": TaoStartup(lattice_file=LAT_A)},
        constraints=[EleIsCloseConstraint(obs_a=obs_a, obs_b=obs_b)],
    )
    saved, results = run(config, DATA_DIR)
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
    assert results.regression == []


def test_run_regression_missing_observation():
    obs = DatumLiteral(model_value=1.0, design_value=0.0)
    config = ConstraintsConfig(
        lattices={},
        constraints=[DatumRegressionConstraint(obs=obs)],
    )
    compare = SavedObservations(entries=[])
    _, results = run(config, DATA_DIR, compare=compare)
    assert len(results.regression) == 1
    rr = results.regression[0]
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
            SavedEntry(observable=obs_pass, observation=obs_pass()),
            SavedEntry(
                observable=obs_fail,
                observation=DatumObservation(model_value=999.0, design_value=0.0),
            ),
        ]
    )
    _, results = run(config, DATA_DIR, compare=compare)
    assert len(results.regression) == 2
    by_group = {rr.group: rr for rr in results.regression}
    a = by_group["A"]
    assert a.result.is_satisfied
    assert a.label == obs_pass.label
    assert a.description == "passes"
    assert a.comment == "note"
    assert a.group == "A"
    b = by_group["B"]
    assert not b.result.is_satisfied
    assert b.group == "B"
