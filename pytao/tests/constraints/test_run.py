import pathlib

from pytao.constraints.config import (
    ConstraintsConfig,
    DatumIsCloseConstraint,
    EleIsCloseConstraint,
    EleLessThanConstraint,
)
from pytao.constraints.main import run
from pytao.constraints.observables.datum import DatumLiteral
from pytao.constraints.observables.ele import EleLessThan, EleLiteral, EleObservable
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
    assert not bad_cr.result.is_close
    assert bad_cr.result.error is not None
    good_cr = next(cr for cr in results.constraints if cr.description == "good")
    assert good_cr.result.is_close


def test_run_continues_on_invalid_element():
    obs = EleObservable(lattice_id="lat_a", ele_id="BEGINNING")
    wrong_lit = EleLiteral(beta_a=999.0, alpha_a=0.0, beta_b=999.0, alpha_b=0.0)
    small_lit = EleLiteral(beta_a=1.0, alpha_a=-0.5, beta_b=1.0, alpha_b=2.0)
    valid_lit = EleLiteral(beta_a=3.0, alpha_a=-0.5, beta_b=8.0, alpha_b=2.0)
    config = ConstraintsConfig(
        lattices={"lat_a": TaoStartup(lattice_file=LAT_A)},
        constraints=[
            EleIsCloseConstraint(obs_a=obs, obs_b=wrong_lit),
            EleIsCloseConstraint(obs_a=obs, obs_b=wrong_lit),
            EleLessThanConstraint(
                obs_a=obs, obs_b=small_lit, comparison=EleLessThan(beta_a=True)
            ),
            EleLessThanConstraint(
                obs_a=obs, obs_b=small_lit, comparison=EleLessThan(beta_b=True)
            ),
            EleIsCloseConstraint(obs_a=valid_lit, obs_b=valid_lit, description="valid"),
        ],
    )
    saved, results = run(config, DATA_DIR)
    assert results.lattices["lat_a"].loaded
    assert results.lattices["lat_a"].error is None
    assert len(results.constraints) == 5
    assert not results.constraints[0].result.is_close
    assert not results.constraints[1].result.is_close
    assert not results.constraints[2].result.is_less
    assert not results.constraints[3].result.is_less
    valid_cr = results.constraints[4]
    assert valid_cr.description == "valid"
    assert valid_cr.result.is_close


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
