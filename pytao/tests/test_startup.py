import pathlib

import pytest

from ..errors import TaoInitializationError, TaoInvalidArgumentsError
from ..model.ele.ele import ElementNotFoundError
from ..startup import TRISTATE_SWITCHES, TaoStartup
from ..subproc import SubprocessTao
from ..tao import Tao
from .conftest import get_regression_test


def test_examples_can_init(tao_example: TaoStartup) -> None:
    assert tao_example.can_initialize


def test_regression_tests_can_init(tao_regression_test: TaoStartup) -> None:
    assert tao_regression_test.can_initialize


@pytest.mark.parametrize(
    ("startup"),
    [
        pytest.param(TaoStartup("-i foo")),
        pytest.param(TaoStartup("-init foo")),
        pytest.param(TaoStartup("-init_file foo")),
        pytest.param(TaoStartup("-la foo")),
        pytest.param(TaoStartup("-lat foo")),
        pytest.param(TaoStartup("-lattice_file foo")),
    ],
)
def test_can_init(startup: TaoStartup) -> None:
    assert startup.can_initialize
    assert startup.init_file or startup.lattice_file


@pytest.mark.parametrize(
    ("startup"),
    [
        pytest.param(TaoStartup()),
        pytest.param(TaoStartup(external_plotting=True)),
        pytest.param(TaoStartup(var_file="no")),
    ],
)
def test_no_init(startup: TaoStartup) -> None:
    assert not startup.can_initialize


def test_plotting() -> None:
    assert (
        TaoStartup(plot="mpl", external_plotting=True, noplot=True).tao_init
        == "-external_plotting -noplot"
    )


def test_init_override() -> None:
    assert TaoStartup("-init_file foo", init_file="test").tao_init == "-init_file foo"


def test_geometry() -> None:
    assert TaoStartup(geometry="3x3").tao_init == "-geometry 3x3"
    assert TaoStartup(geometry=(32, 23)).tao_init == "-geometry 32x23"


def test_startup_requires_init() -> None:
    with pytest.raises(TaoInitializationError):
        Tao()
    with pytest.raises(TaoInitializationError):
        Tao("bad_init")


def test_startup_bad_file() -> None:
    bad_fn = "/tmp/foooooobarrrr"
    for check in range(1, 3):
        print(f"** Check number {check} **")
        # We do this *twice* because it's actually two different paths to
        # initialization of the underlying Fortran library.
        with pytest.raises(TaoInitializationError) as raises_context:
            Tao(f"-init_file {bad_fn}")
        ex = raises_context.value
        assert hasattr(ex, "tao_output")
        assert bad_fn in ex.tao_output
        assert "TAO INITIALIZATION FILE NOT FOUND" in ex.tao_output
        print(str(ex))


fodo_lattice = """
no_digested
parameter[geometry] = closed
parameter[particle] = electron

beginning[e_tot] = 150e6
beginning[x_position]     = 1e-3
beginning[y_position]     = 1e-3
beginning[z_position]     = 0.

L_tot = 2
Lq = 0.1
Ld = (L_tot - 2*Lq)/2

k1_optimal =  (1/Lq) * 2*sqrt(2) / L_tot

K1: kicker, L=0.1

P0: pipe

P1: P0, L = Ld

q0: quad, L = Lq
q1: q0
q2: q0

O_L: overlay = {p1[L]:(Lcell - 2*Lq)/2}, var = {Lcell}
O_L[Lcell] = L_tot

q1[k1] = k1_optimal
q2[k1] = -k1_optimal

fodo1: line = (k1, p1, q1, p1, q2 )

use, fodo1
"""


def test_startup_from_lattice_contents() -> None:
    # back-compat for now
    with SubprocessTao.from_lattice_contents(fodo_lattice, noplot=True) as tao:
        ids = tao.unique_ele_ids(sort_by=None)
        names = [tao.ele_head(ele_id)["name"] for ele_id in ids]
    assert names == [
        "BEGINNING",
        "K1",
        "O_L",
        "P1",
        "Q1",
        "P1",
        "Q2",
        "END",
    ]


fodo_init_file = """
&tao_design_lattice
  n_universes = 1
  design_lattice(1)%file = "to_be_replaced.bmad"
/
"""

fodo_plot_file = """
&tao_template_plot
  plot%name = "custom_layout"
  plot%n_graph = 1
/

&tao_template_graph
  graph_index = 1
  graph%name = "g"
  graph%type = "lat_layout"
  graph%box = 1, 1, 1, 1
  graph%n_curve = 0
/
"""


def test_startup_from_init_file_contents() -> None:
    with SubprocessTao.from_contents(
        contents=fodo_lattice,
        init_file_contents=fodo_init_file,
        noplot=True,
    ) as tao:
        # The placeholder design_lattice(1)%file was replaced with `fodo_lattice`
        assert tao.ele_gen_attribs("Q1")["K1"] != 0.0


def test_startup_from_init_file_contents_raises() -> None:
    with pytest.raises(ValueError):
        SubprocessTao.from_contents(
            contents=fodo_lattice,
            lattice_contents="foo",  # <- both kwargs
            init_file_contents=fodo_init_file,
            noplot=True,
        )


def test_startup_from_plot_file_contents() -> None:
    with SubprocessTao.from_contents(
        lattice_contents=fodo_lattice,
        init_file_contents=fodo_init_file,
        plot_file_contents=fodo_plot_file,
        noplot=True,
    ) as tao:
        assert "custom_layout" in tao.plot_list("t")


def test_startup_from_contents_requires_contents() -> None:
    with pytest.raises(ValueError):
        Tao.from_contents()


@pytest.mark.parametrize(
    ("contents_key", "file_key"),
    [
        pytest.param("lattice_contents", "lattice_file"),
        pytest.param("init_file_contents", "init_file"),
        pytest.param("plot_file_contents", "plot_file"),
    ],
)
def test_startup_from_contents_conflicts_with_file(contents_key: str, file_key: str) -> None:
    with pytest.raises(ValueError):
        Tao.from_contents(**{contents_key: "contents", file_key: "some_file"})


def test_startup_lattice_file2(tmp_path: pathlib.Path) -> None:
    lat1 = tmp_path / "lat1.bmad"
    lat1.write_text(fodo_lattice)
    lat2 = tmp_path / "lat2.bmad"
    lat2.write_text("q1[k1] = 0.5\n")

    with SubprocessTao(lattice_file=lat1, lattice_file2=lat2, noplot=True) as tao:
        assert tao.ele_gen_attribs("Q1")["K1"] == 0.5


@pytest.mark.parametrize(
    "startup",
    [
        pytest.param(TaoStartup(lattice_file="a.bmad,b.bmad"), id="kwarg"),
        pytest.param(TaoStartup("-lattice_file a.bmad,b.bmad"), id="init_string"),
        pytest.param(
            TaoStartup.from_cli_args(["-lattice_file", "a.bmad,b.bmad"]), id="cli_args"
        ),
        pytest.param(TaoStartup(lattice_file="a.bmad", lattice_file2="b.bmad"), id="explicit"),
        pytest.param(
            TaoStartup(lattice_file="a.bmad,b.bmad", lattice_file2="b.bmad"),
            id="redundant",
        ),
    ],
)
def test_lattice_file2_split(startup: TaoStartup) -> None:
    assert startup.lattice_file == "a.bmad"
    assert startup.lattice_file2 == "b.bmad"
    assert startup.tao_init == "-lattice_file a.bmad,b.bmad"


def test_lattice_file2_split_on_assignment() -> None:
    startup = TaoStartup()
    startup.lattice_file = "a.bmad,b.bmad"
    assert startup.lattice_file == "a.bmad"
    assert startup.lattice_file2 == "b.bmad"


def test_lattice_file2_split_on_first_comma() -> None:
    startup = TaoStartup(lattice_file="a,b,c")
    assert startup.lattice_file == "a"
    assert startup.lattice_file2 == "b,c"


def test_lattice_file2_multi_universe_passthrough() -> None:
    startup = TaoStartup(lattice_file="a,b|c,d")
    assert startup.lattice_file == "a,b|c,d"
    assert startup.lattice_file2 is None


def test_lattice_file2_conflict() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        TaoStartup(lattice_file="a.bmad,b.bmad", lattice_file2="c.bmad")


def test_lattice_file2_multi_universe_conflict() -> None:
    with pytest.raises(ValueError, match="multi-universe"):
        TaoStartup(lattice_file="a.bmad|b.bmad", lattice_file2="c.bmad")


def test_lattice_file2_requires_lattice_file() -> None:
    with pytest.raises(TaoInvalidArgumentsError):
        _ = TaoStartup(lattice_file2="b.bmad").tao_init


def test_lattice_file2_with_path_prefix() -> None:
    startup = TaoStartup(lattice_file="a.bmad,b.bmad").with_path_prefix(
        pathlib.Path("/prefix")
    )
    assert startup.lattice_file == pathlib.Path("/prefix/a.bmad")
    assert startup.lattice_file2 == pathlib.Path("/prefix/b.bmad")


@pytest.mark.parametrize(
    "cli_args, expected_attribute, expected_value",
    [
        pytest.param(["-debug"], "debug", True, id="flag_debug_is_true"),
        pytest.param(["--debug"], "debug", False, id="flag_debug_negated_is_false"),
        pytest.param(
            ["-lattice_file", "my_lattice.bmad"],
            "lattice_file",
            "my_lattice.bmad",
            id="string_arg_lattice_file",
        ),
        pytest.param(
            ["-la", "my_lattice.bmad"],
            "lattice_file",
            "my_lattice.bmad",
            id="string_short_lattice",
        ),
        pytest.param(
            ["-lat", "my_lattice.bmad"],
            "lattice_file",
            "my_lattice.bmad",
            id="string_short_lattice",
        ),
        pytest.param(
            ["-geometry", "1024x768"], "geometry", "1024x768", id="string_arg_geometry"
        ),
        pytest.param([], "prompt_color", "blue", id="default_arg_prompt_color"),
        pytest.param(
            ["-prompt_color", "red"], "prompt_color", "red", id="override_arg_prompt_color"
        ),
        pytest.param(["-quiet", "warnings"], "quiet", "warnings", id="choices_arg_quiet"),
        pytest.param([], "rf_on", None, id="default_flag_rf_on_is_none"),
        pytest.param(["-rf_on"], "rf_on", True, id="flag_rf_on_is_true"),
        pytest.param(["--rf_on"], "rf_on", False, id="negated_flag_rf_on_is_false"),
        pytest.param([], "noplot", None, id="default_flag_noplot_is_none"),
        pytest.param(["--noplot"], "noplot", False, id="negated_flag_noplot_is_false"),
    ],
)
def test_tao_parser_mapping_to_dataclass(
    cli_args: list[str], expected_attribute: str, expected_value
):
    parsed_dataclass = TaoStartup.from_cli_args(cli_args)
    actual_value = getattr(parsed_dataclass, expected_attribute)
    assert actual_value == expected_value


@pytest.mark.parametrize("name", sorted(TRISTATE_SWITCHES))
@pytest.mark.parametrize(
    "switch, kwargs, expected",
    [
        pytest.param("", {}, None, id="absent"),
        pytest.param("-{name}", {}, "-{name}", id="init-string"),
        pytest.param("--{name}", {}, "--{name}", id="init-string-negated"),
        pytest.param("", {"value": True}, "-{name}", id="kwarg-true"),
        pytest.param("", {"value": False}, "--{name}", id="kwarg-false"),
    ],
)
def test_tristate_switch_reaches_tao_init(
    name: str, switch: str, kwargs: dict, expected: str | None
) -> None:
    init = f"-lat foo.bmad {switch.format(name=name)}"
    if kwargs:
        startup = TaoStartup(init=init, **{name: kwargs["value"]})
    else:
        startup = TaoStartup(init=init)
    parts = startup.tao_init.split()
    positive, negated = f"-{name}", f"--{name}"
    if expected is None:
        assert positive not in parts
        assert negated not in parts
    elif expected.startswith("--"):
        assert negated in parts
        assert positive not in parts
    else:
        assert positive in parts
        assert negated not in parts


def test_negated_plain_switch_is_omitted() -> None:
    # For switches not in TRISTATE_SWITCHES, `--x` is the same as no switch.
    startup = TaoStartup(init="-lat foo.bmad -debug --debug")
    assert startup.debug is False
    assert "-debug" not in startup.tao_init.split()
    assert "--debug" not in startup.tao_init.split()


def test_tao_parser_unsupported_choice():
    with pytest.raises(TaoInvalidArgumentsError):
        TaoStartup.from_cli_args(["-quiet", "invalid_level"])


@pytest.mark.parametrize(
    "cli_args, expected_quiet",
    [
        pytest.param(["-quiet"], "all", id="-quiet"),
        pytest.param(["-quiet", "all"], "all", id="-quiet all"),
        pytest.param(["-quiet", "warnings"], "warnings", id="-quiet warnings"),
        pytest.param([], False, id="no-quiet"),
    ],
    ids=[
        "implicit_all",
        "explicit_all",
        "explicit_warnings",
        "blank_returns_none",
    ],
)
def test_quiet_parsing(cli_args: list[str], expected_quiet: str | None):
    res = TaoStartup.from_cli_args(cli_args)
    assert res.quiet == expected_quiet


@pytest.fixture(scope="module")
def tao():
    startup = get_regression_test("tao.init_optics_matching")
    with startup.run_context(use_subprocess=False) as tao:
        yield tao


@pytest.fixture
def tao_mpl():
    startup = get_regression_test("tao.init_optics_matching")
    startup.plot = "mpl"
    with startup.run_context(use_subprocess=False) as tao:
        yield tao


def test_tao_repr(tao: Tao):
    r = repr(tao)
    assert "Tao" in r
    assert "so_lib=" in r


def test_tao_cmds(tao: Tao):
    results = tao.cmds(["show ele 0", "show ele 1"])
    assert isinstance(results, list)
    assert len(results) == 2


def test_tao_cmds_no_suppress(tao: Tao):
    results = tao.cmds(
        ["show ele 0"],
        suppress_lattice_calc=False,
        suppress_plotting=False,
    )
    assert len(results) == 1


def test_tao_version(tao: Tao):
    v = tao.version()
    assert v is not None


def test_invalid_backend_key(tao: Tao):
    with pytest.raises(NotImplementedError):
        tao._get_graph_manager_by_key("nonexistent")


def test_invalid_user_backend(tao_mpl):
    with pytest.raises(ValueError, match="Unsupported backend"):
        tao_mpl._get_user_specified_backend("nonexistent")


def test_update_plot_shapes_no_who(tao_mpl):
    with pytest.raises(ValueError, match="Must specify"):
        tao_mpl.update_plot_shapes(layout=False, floor=False)


def test_matplotlib_property(tao_mpl):
    manager = tao_mpl.matplotlib
    assert manager is not None


def test_plot_manager_property(tao_mpl):
    manager = tao_mpl.plot_manager
    assert manager is tao_mpl.matplotlib


def test_element_not_found(tao: Tao):
    with pytest.raises(ElementNotFoundError):
        tao.ele("nonexistent_element_12345")


def test_evaluate_string_list(tao: Tao):
    result = tao.evaluate("lat::sigma.x[0:5]", flags="")
    assert result is not None
    assert len(result) == 6


RF_LATTICE = """
beginning[beta_a] = 10
beginning[beta_b] = 10
beginning[e_tot] = 1e9
parameter[geometry] = open
d: drift, l = 1
c: rfcavity, l = 1, voltage = 1e6, rf_frequency = 500e6
lat: line = (d, c, d)
use, lat
"""


@pytest.mark.parametrize("init_file_rf_on", [True, False], ids=["init-T", "init-F"])
@pytest.mark.parametrize(
    "switch, kwargs, expected",
    [
        pytest.param("", {}, None, id="absent"),
        pytest.param("-rf_on", {}, True, id="-rf_on"),
        pytest.param("--rf_on", {}, False, id="--rf_on"),
        pytest.param("", {"rf_on": True}, True, id="rf_on=True"),
        pytest.param("", {"rf_on": False}, False, id="rf_on=False"),
    ],
)
def test_rf_on_switch_overrides_init_file(
    tmp_path: pathlib.Path,
    init_file_rf_on: bool,
    switch: str,
    kwargs: dict,
    expected: bool | None,
) -> None:
    # Absent: Tao uses the init file's global%rf_on.  -rf_on / --rf_on force it.
    (tmp_path / "lat.bmad").write_text(RF_LATTICE)
    flag = "T" if init_file_rf_on else "F"
    init_file = tmp_path / "tao.init"
    init_file.write_text(
        "&tao_start\n  n_universes = 1\n/\n"
        f'&tao_design_lattice\n  design_lattice(1)%file = "{tmp_path / "lat.bmad"}"\n/\n'
        f"&tao_params\n  global%rf_on = {flag}\n/\n"
    )
    with SubprocessTao(init=f"-init {init_file} -noplot {switch}", **kwargs) as tao:
        rf_on = tao.tao_global()["rf_on"]
    assert rf_on == (init_file_rf_on if expected is None else expected)
