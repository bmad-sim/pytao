import contextvars
import pathlib
import time
from typing import NamedTuple

import numpy as np
import numpy.testing
import pytest

import pytao
from pytao import SubprocessTao
from pytao.model.ele import Element, Lattice
from pytao.model.base import format_from_filename
from pytao.model.ele.time_stats import _PytaoStatistics, get_pytao_statistics

from ..conftest import no_pytao_debug_logging, timed_section


class TestElement(NamedTuple):
    __test__ = False

    init_file: str | None
    lat_file: str | None
    element: str


test_element: contextvars.ContextVar[TestElement | None] = contextvars.ContextVar(
    "test_element", default=None
)

elements = [
    TestElement(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d",
        lat_file=None,
        element=ele,
    )
    for ele in ("BEGINNING", "Q1", "END")
]

elements.extend(
    [
        TestElement(
            init_file=None,
            lat_file="$ACC_ROOT_DIR/regression_tests/photon_test/photon_test.bmad",
            element=ele,
        )
        for ele in (
            "BEGINNING",
            "PINIT",
            "CST1",
            "CST2",
            "CST3",
            "CST4",
            "END",
        )
    ]
)

elements.extend(
    [
        TestElement(
            init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_grid",
            lat_file=None,
            element=ele,
        )
        for ele in (
            "BEGINNING",
            "SBEND0",
            "END",
        )
    ]
)


@pytest.fixture(scope="module")
def _tao() -> SubprocessTao:
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d",
        noplot=True,
    ) as tao:
        yield tao


@pytest.fixture(
    scope="function",
    params=elements,
    ids=[f"{elem}" for elem in elements],
)
def tao(_tao: SubprocessTao, request: pytest.FixtureRequest) -> SubprocessTao:
    param: TestElement = request.param
    if (
        _tao.init_settings.init_file != param.init_file
        or _tao.init_settings.lattice_file != param.lat_file
    ):
        with pytao.filter_tao_messages_context(
            functions=[
                "track1_cyrstal",  # sic.
                # "closed_orbit_calc",
                # "twiss_propagate1",
                # "tao_init_lattice",
            ]
        ):
            _tao.init(
                init_file=param.init_file,
                lattice_file=param.lat_file,
                noplot=True,
            )
    test_element.set(param)
    return _tao


@pytest.fixture(scope="function")
def ele_id(tao: SubprocessTao) -> SubprocessTao:
    test_el = test_element.get()
    assert test_el is not None
    return test_el.element


def test_element_unfilled(tao: SubprocessTao, ele_id: str):
    ele = Element.from_tao(tao, ele_id, which="model", attrs=True)
    assert ele.which == "model"
    assert ele.ele_id == ele_id

    # Make sure tao gives back the same element given our ElementID
    id = ele.id
    assert Element.from_tao(tao, id).id == id


def test_tao_ele_method(tao: SubprocessTao, ele_id: str):
    ele = tao.ele(ele_id, which="model", attrs=True)
    assert ele.attrs is not None

    (ele,) = tao.eles(ele_id, which="model", attrs=True)
    assert ele.attrs is not None

    eles = tao.eles([ele_id, ele_id], which="model", attrs=True)
    assert len(eles) == 2
    assert eles[0] == eles[1]


def test_element_fill(tao: SubprocessTao, ele_id: str):
    ele = Element.from_tao(tao, ele_id, which="model", attrs=True)
    ele.fill(tao, grid_field_points=True, wall3d_table=True)

    assert ele.attrs is not None

    if ele.head.has_lord_slave:
        assert ele.lord_slave is not None
    if ele.head.has_photon:
        assert ele.photon is not None

    assert ele.orbit is not None

    if ele.head.has_twiss:
        assert ele.twiss is not None
    if ele.head.has_mat6:
        assert ele.mat6 is not None
    if ele.head.num_grid_field:
        assert ele.grid_field is not None
        assert len(ele.grid_field) == ele.head.num_grid_field
        assert ele.grid_field[0].points is not None
        assert len(ele.grid_field[0].points)
    if ele.head.has_wall3d:
        assert ele.chamber_walls is not None
        assert ele.wall3d is not None
    assert ele.multipoles is not None
    if ele.head.has_wake:
        assert ele.wake is not None
    if ele.head.has_control:
        assert ele.control_vars is not None


def test_element_update(tao: SubprocessTao, ele_id: str):
    ele = Element.from_tao(tao, ele_id, which="model", attrs=True)
    assert ele.attrs is not None
    ele.attrs.query(tao)

    # smoke test pretty repr and equality check
    print(repr(ele.attrs))
    assert ele.attrs == ele.attrs


# We're not supporting attribute setting for now
# def test_element_set_attrs(tao: SubprocessTao, ele_id: str, monkeypatch: pytest.MonkeyPatch):
#     ele = Element.from_tao(tao, ele_id, which="model", attrs=True)
#     assert ele.attrs is not None
#
#     cmds = ele.attrs.set_commands
#     assert len(cmds) == len(ele.attrs.settable_fields)
#
#     orig_cmd = tao.cmd
#
#     def should_pass(cmd: str):
#         return cmd == "pipe global" or "global lattice_calc" in cmd or "global plot_on" in cmd
#
#     def mock_command_raise(cmd: str, raises: bool = True):
#         if should_pass(cmd):
#             return orig_cmd(cmd, raises=raises)
#
#         raise TaoCommandError(f"Making {cmd!r} raise")
#
#     monkeypatch.setattr(tao, "cmd", mock_command_raise)
#
#     if not ele.attrs.settable_fields:
#         return
#
#     assert ele.attrs.set(tao, allow_errors=True) is False
#
#     def mock_command_ok(cmd: str, raises: bool = True):
#         if should_pass(cmd):
#             return orig_cmd(cmd, raises=raises)
#         return []
#
#     monkeypatch.setattr(tao, "cmd", mock_command_ok)
#     with ele.attrs.set_context(tao):
#         pass


@pytest.fixture(scope="module")
def cbeta_ffag_tao():
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_ffag/tao.init",
        noplot=True,
    ) as tao:
        yield tao


@pytest.fixture()
def pytao_stats():
    stats = get_pytao_statistics()
    stats.reset()
    t0 = time.monotonic()
    try:
        yield stats
    finally:
        elapsed = time.monotonic() - t0
        print(f"pytao stats (test took: {elapsed:.1f}):")
        print(stats)
        stats.reset()


def test_cbeta_unique(cbeta_ffag_tao: SubprocessTao, pytao_stats: _PytaoStatistics):
    lat = Lattice.from_tao_unique(cbeta_ffag_tao, defaults=True)
    print("Total elements:", len(lat.elements))

    assert len(lat.by_element_index) == len(lat.elements)
    assert sum(len(eles) for eles in lat.by_element_key.values()) == len(lat.elements)
    lat.by_element_name


def test_cbeta_tracking(cbeta_ffag_tao: SubprocessTao, pytao_stats: _PytaoStatistics):
    lat = Lattice.from_tao_tracking(cbeta_ffag_tao, defaults=True)
    print("Total elements:", len(lat.elements))

    assert len(lat.by_element_index) == len(lat.elements)
    assert sum(len(eles) for eles in lat.by_element_key.values()) == len(lat.elements)
    lat.by_element_name


def test_lattice_track_start(cbeta_ffag_tao: SubprocessTao, pytao_stats: _PytaoStatistics):
    full_lat = Lattice.from_tao_tracking(cbeta_ffag_tao)

    last_idx = max(full_lat.by_element_index)
    print(full_lat)

    def slice_full(idx0: int, idx1: int):
        return [idx for idx in full_lat.by_element_index if idx0 <= idx <= idx1]

    from_10 = Lattice.from_tao_tracking(cbeta_ffag_tao, track_start="10")
    print(from_10)

    removed_indices = slice_full(0, 9)
    assert len(removed_indices)
    for idx in removed_indices:
        assert idx not in from_10.by_element_index

    from_10_to_20 = Lattice.from_tao_tracking(cbeta_ffag_tao, track_start="10", track_end="20")
    print(from_10_to_20)

    removed_indices = slice_full(0, 9) + slice_full(21, last_idx)
    assert len(removed_indices)
    for idx in removed_indices:
        assert idx not in from_10_to_20.by_element_index

    until_20 = Lattice.from_tao_tracking(cbeta_ffag_tao, track_end="20")
    print(until_20)

    removed_indices = slice_full(21, last_idx)
    assert len(removed_indices)
    for idx in removed_indices:
        assert idx not in until_20.by_element_index


@pytest.mark.parametrize(
    ("exclude_defaults", "extension"),
    [
        pytest.param(True, ".json", id="json-no-defaults"),
        pytest.param(False, ".json", id="json-with-defaults"),
        pytest.param(True, ".json.gz", id="json-gz-no-defaults"),
        pytest.param(False, ".json.gz", id="json-gz-with-defaults"),
        pytest.param(True, ".msgpack", id="msgpack-no-defaults"),
        pytest.param(False, ".msgpack", id="msgpack-with-defaults"),
        # pytest.param(True, ".yaml", id="yaml-no-defaults"),
        # pytest.param(False, ".yaml", id="yaml-with-defaults"),
    ],
)
def test_lattice_write(
    cbeta_ffag_tao: SubprocessTao,
    tmp_path: pathlib.Path,
    exclude_defaults: bool,
    extension: str,
):
    with no_pytao_debug_logging():
        print()
        with timed_section(description="Lattice.from_tao_tracking"):
            lat = Lattice.from_tao_tracking(
                cbeta_ffag_tao, defaults=True, track_end=10, comb=True
            )

    print()
    fn = tmp_path / f"lattice{extension}"
    t0 = time.monotonic()
    with timed_section(description="lat.write"):
        lat.write(fn, exclude_defaults=exclude_defaults)
    t1 = time.monotonic()

    print()
    with timed_section(description="Lattice.from_file"):
        round_tripped = Lattice.from_file(fn)
    t2 = time.monotonic()
    assert round_tripped.filename == fn

    print("Save time:", t1 - t0)
    print("Load time:", t2 - t1)
    with open(fn, "rb") as fp:
        num_bytes = len(fp.read())

    def clean_json(s: str) -> str:
        # {"key": -0.0} -> {"key": 0.0}
        return s.replace(": -0.0,", ": 0.0,")

    print()
    print(lat)
    print(f"Serialized to: {num_bytes} bytes ({num_bytes // 1024 // 1024} MB)")
    print()
    with timed_section(description="Validation"):
        lat.filename = fn
        # Note: going into details here mostly for debugging purposes; we want to
        # ensure the round-tripped lattice is identical to the tao-dumped one.
        assert len(lat.elements) == len(round_tripped.elements)
        for ele_orig, ele_round_tripped in zip(lat.elements, round_tripped.elements):
            orig_json = clean_json(ele_orig.model_dump_json(indent=1))
            rt_json = clean_json(ele_round_tripped.model_dump_json(indent=1))
            assert orig_json == rt_json, ele_orig.name
            # assert ele_orig == ele_round_tripped, ele_orig.name
        assert clean_json(round_tripped.model_dump_json(indent=1)) == clean_json(
            lat.model_dump_json(indent=1)
        )


@pytest.mark.parametrize(
    "filename, expected_format",
    [
        pytest.param("config.yml", "yaml", id="yaml_short_extension"),
        pytest.param("config.yaml", "yaml", id="yaml_long_extension"),
        pytest.param("CONFIG.YML", "yaml", id="yaml_uppercase_extension"),
        pytest.param("data.msgpack", "msgpack", id="msgpack_full_extension"),
        pytest.param("data.mpk", "msgpack", id="msgpack_short_extension"),
        pytest.param("data.MPK", "msgpack", id="msgpack_SHORT_extension"),
        pytest.param("archive.json.gz", "json.gz", id="json_gz_standard"),
        pytest.param("ARCHIVE.JSON.GZ", "json.gz", id="json_gz_uppercase"),
        pytest.param("data.json", "json", id="json_standard"),
        pytest.param("DATA.JSON", "json", id="json_uppercase"),
        pytest.param("unknown.txt", "json", id="fallback_unknown_extension_to_json"),
        pytest.param("no_extension", "json", id="fallback_no_extension_to_json"),
        pytest.param(
            "multiple.ext.json.gz", "json.gz", id="json_gz_with_multiple_prior_extensions"
        ),
    ],
)
def test_format_from_filename(filename: str, expected_format: str) -> None:
    fn_path = pathlib.Path(filename)
    assert format_from_filename(fn_path) == expected_format


def test_sr_wake_longitudinal():
    LAT = """

    beginning[beta_a] = 1
    beginning[beta_b] = 1
    beginning[e_tot] = 1e6
    parameter[geometry] = open

    p1: pipe, L = 1,
    ! AC Resistive wall wakefield
    ! Adapted from SLAC-PUB-10707
    !    Material        : copper-slac-pub-10707
    !    Conductivity    : 65000000.0 S/m
    !    Relaxation time : 2.7e-14 s
    !    Geometry        : round
    !    Radius          : 0.1 m
    !    s₀              : 9.347497493596404e-05  m
    !    Γ               : 0.086594260886886 
    sr_wake =  
    {z_scale=1, amp_scale=1, scale_with_length=True, z_max=100,
    longitudinal = {3595020714472.61, 7562.955649662248, 19650.316202011512, 0.25, none}}

    lat: line = (p1)

    use, lat
    """
    with SubprocessTao.from_lattice_contents(LAT, noplot=True) as tao:
        wake = tao.ele("p1").wake
        assert wake is not None
        assert wake.sr_long is not None
        numpy.testing.assert_allclose(
            np.asarray(wake.sr_long.table[0][:-1]),
            [
                3595020714472.61,
                7562.955649662248,
                19650.316202011512,
                0.25,
                # "none",
            ],
        )
        assert wake.sr_long.table[0][-1] == "none"
        assert len(wake.sr_long.table) == 1


def test_sr_wake_transverse():
    pytest.skip("segfault - see upstream issue")
    with SubprocessTao(
        lattice_file="$ACC_ROOT_DIR/regression_tests/wake_test/wake_test.bmad", noplot=True
    ) as tao:
        tao.ele("p1").wake
        # TODO value comparison
