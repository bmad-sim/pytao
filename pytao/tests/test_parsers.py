import math
from datetime import datetime

import numpy as np
import pytest

from .. import AnyTao
from ..util.parsers import (
    _value_float_or_none as value_float_or_none,
)
from ..util.parsers import (
    parse_bunch_comb,
    parse_da_aperture,
    parse_derivative,
    parse_ele_ac_kicker,
    parse_ele_cylindrical_map,
    parse_ele_grid_field,
    parse_ele_param,
    parse_evaluate,
    parse_matrix,
    parse_merit,
    parse_pytype,
    parse_show_version,
    parse_tao_python_data,
    parse_taylor_map,
    parse_var_v_array_line,
    parse_wave,
)
from ..errors import TaoDataInvalidError
from ..util import parsers
from .conftest import ensure_successful_parsing, test_root
from .test_interface_commands import new_tao


@pytest.mark.parametrize(
    ["type", "value", "expected"],
    [
        # <component_name>;<type>;<is_variable>;<component_value>
        pytest.param("STR", "1", "1"),
        pytest.param("ENUM", "1", "1"),
        pytest.param("FILE", "1", "1"),
        pytest.param("CRYSTAL", "1", "1"),
        pytest.param("COMPONENT", "1", "1"),
        pytest.param("DAT_TYPE", "1", "1"),
        pytest.param("DAT_TYPE_Z", "1", "1"),
        pytest.param("SPECIES", "1", "1"),
        pytest.param("ELE_PARAM", "1", "1"),
        pytest.param("STR_ARR", "1", ["1"]),
        pytest.param("ENUM_ARR", "1", ["1"]),
        pytest.param("STR_ARR", "1;2", ["1", "2"]),
        pytest.param("ENUM_ARR", "1;2", ["1", "2"]),
        pytest.param("LOGIC", "T", True),
        pytest.param("LOGIC", "F", False),
        pytest.param("INT", "1", 1),
        pytest.param("INUM", "1", 1),
        pytest.param("REAL", "1", 1.0),
        pytest.param("INT_ARR", "0;1;2", np.array([0, 1, 2])),
        pytest.param("REAL_ARR", "0.;1.;2.", np.array([0.0, 1.0, 2.0])),
        pytest.param("INT_ARR", "0", np.array([0])),
        pytest.param("REAL_ARR", "0.", np.array([0.0])),
        pytest.param("COMPLEX", "0.;1.", 1j),
        pytest.param("STRUCT", "n1;INT;0;n2;REAL;1.0", {"n1": 0, "n2": 1.0}),
        pytest.param(
            "STRUCT",
            "width;INT;0;color;ENUM;color;line^pattern;ENUM;pattern",
            {"width": 0, "color": "color", "line^pattern": "pattern"},
        ),
    ],
)
def test_parse_line(type: str, value: str, expected):
    name = "name"

    for settable in "FTI":
        line = f"{name};{type};{settable};{value}"
        parsed_value = parse_tao_python_data([line])[name]

        if isinstance(expected, np.ndarray):
            assert np.all(parsed_value == expected)
        else:
            assert parsed_value == expected


def test_building_wall_list_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert set(tao.building_wall_list(ix_section="")[0].keys()) == {
            "index",
            "name",
            "constraint",
            "shape",
            "color",
            "line_width",
        }


def test_building_wall_list_2(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert set(tao.building_wall_list(ix_section="1")[0].keys()) == {
            "index",
            "z",
            "x",
            "radius",
            "z_center",
            "x_center",
        }


def test_building_wall_graph_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall",
    ) as tao:
        tao.cmd("place -no_buffer r11 floor_plan")
        assert set(tao.building_wall_graph(graph="r11.g")[0].keys()) == {
            "index",
            "point",
            "offset_x",
            "offset_y",
            "radius",
        }


def test_constraints_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        tao.constraints(who="data")


def test_constraints_2(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        tao.constraints(who="var")


def test_data_d2_array_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert "orbit" in tao.data_d2_array(ix_uni="1")


def test_data_parameter_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        assert (
            tao.data_parameter(data_array="twiss.end", parameter="model_value")[0]["index"]
            == 1
        )


def test_datum_has_ele_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        assert tao.datum_has_ele(datum_type="twiss.end") in {
            "no",
            "yes",
            "maybe",
            "provisional",
        }


def test_ele_chamber_wall_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d"
    ) as tao:
        assert set(
            tao.ele_chamber_wall(ele_id="1@0>>1", which="model", index="1", who="x")[0].keys()
        ) == {
            "section",
            "longitudinal_position",
            "z1",
            "-z2",
        }


def test_ele_grid_field_points(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_grid",
    ) as tao:
        assert set(
            tao.ele_grid_field(ele_id="1@0>>1", which="model", index="1", who="points")[
                0
            ].keys()
        ) == {
            "i",
            "j",
            "k",
            "data",
        }


def test_ele_elec_multipoles_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert "data" in tao.ele_elec_multipoles(ele_id="1@0>>1", which="model")


def test_ele_gen_gradients(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_em_field"
    ) as tao:
        assert set(
            tao.ele_gen_gradients(ele_id="1@0>>9", which="model", index="1", who="derivs")[0]
        ) == {"i", "j", "k", "dz", "deriv"}
        assert set(
            tao.ele_gen_gradients(ele_id="1@0>>9", which="model", index="1", who="base")
        ) == {
            "file",
            "field_scale",
            "r0",
            "dz",
            "master_parameter",
            "ele_anchor_pt",
            "nongrid^field_type",
            "g_ref",
            "iz0",
            "iz1",
            "size_of_curve",
        }


def test_ele_lord_slave_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert set(tao.ele_lord_slave(ele_id="1@0>>1")[0].keys()) == {
            "type",
            "location_name",
            "name",
            "key",
            "status",
        }


def test_ele_multipoles_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.ele_multipoles(ele_id="1@0>>1", which="model")

    assert isinstance(res, dict)
    if res["data"]:
        assert "KnL" in res or "An" in res["data"][0]


def test_ele_taylor_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_taylor"
    ) as tao:
        res = tao.ele_taylor(ele_id="1@0>>34", which="model")
    assert isinstance(res, dict)
    assert "data" in res
    assert res["data"][0]["index"] == 1


def test_ele_spin_taylor_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_spin"
    ) as tao:
        res = tao.ele_spin_taylor(ele_id="1@0>>2", which="model")
    assert set(res[0].keys()) == {
        "index",
        "term",
        "coef",
        "exp1",
        "exp2",
        "exp3",
        "exp4",
        "exp5",
        "exp6",
    }


def test_ele_wall3d_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d"
    ) as tao:
        res = tao.ele_wall3d(ele_id="1@0>>1", which="model", index="1", who="table")
    assert "data" in res[0]
    assert res[0]["section"] == 1


def test_em_field_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.em_field(ele_id="1@0>>22", which="model", x="0", y="0", z="0", t_or_z="0")
    assert "B1" in res


def test_enum_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.enum(enum_name="tracking_method")
    assert set(res[0].keys()) == {"number", "name"}


def test_floor_plan_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        tao.cmd("place -no_buffer r13.g floor_plan")
        res = tao.floor_plan(graph="r13.g")
    assert "branch_index" in res[0]


def test_floor_orbit_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_floor_orbit",
        nostartup=True,
    ) as tao:
        tao.cmd("place -no_buffer r33 orbit")
        tao.cmd("set graph r33 floor_plan%orbit_scale = 1")
        res = tao.floor_orbit(graph="r33.g")
    assert isinstance(res, list)
    assert isinstance(res[0], dict)
    assert "index" in res[0]
    assert "orbits" in res[0]


def test_help_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        print(tao.help())


def test_inum_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.inum(who="ix_universe")
    assert isinstance(res, list)
    if res:
        assert isinstance(res[0], int)


def test_lat_calc_done_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert tao.lat_calc_done(branch_name="1@0") in {True, False}


def test_lat_branch_list_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        tao.lat_branch_list(ix_uni="1")


def test_lat_param_units_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert isinstance(tao.lat_param_units(param_name="L"), str)


def test_lord_control(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        d_list = tao.lord_control("sex_20w")
        assert "key" in d_list[0]


def test_slave_control(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        d_list = tao.slave_control("ASYM_IR")
        assert "key" in d_list[0]

        d_list = tao.slave_control("CLEO_SOL")
        assert d_list[0]["value"] is None


def test_plot_lat_layout_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert "ix_ele" in tao.plot_lat_layout(ix_uni="1", ix_branch="0")[0]


def test_plot_line_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_plot_line",
    ) as tao:
        res = tao.plot_line(region_name="beta", graph_name="g", curve_name="a", x_or_y="")
    assert "x" in res[0]


def test_plot_line_2(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_plot_line",
    ) as tao:
        res = tao.plot_line(region_name="beta", graph_name="g", curve_name="a", x_or_y="y")
        assert isinstance(
            res,
            np.ndarray,
        )
        res = tao.plot_line(region_name="beta", graph_name="g", curve_name="a", x_or_y="")
        assert "index" in res[0]


def test_plot_symbol_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_plot_line",
    ) as tao:
        res = tao.plot_symbol(region_name="r13", graph_name="g", curve_name="a", x_or_y="")
    assert "index" in res[0]


def test_shape_list_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert "shape_index" in tao.shape_list(who="floor_plan")[0]


def test_shape_pattern_list_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_shape"
    ) as tao:
        res = tao.shape_pattern_list(ix_pattern="")
    assert set(res[0].keys()) == {
        "name",
        "line_width",
    }


def test_show_1(tao_cls: type[AnyTao]):
    pytest.skip("TODO")
    tao = new_tao(init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init")
    tao.show(line="-python")


def test_species_to_int_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert isinstance(tao.species_to_int(species_str="electron"), int)


def test_species_to_str_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert isinstance(tao.species_to_str(species_int="-1"), str)


def test_spin_invariant_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert isinstance(
            tao.spin_invariant(who="l0", ix_uni="1", ix_branch="0", which="model"),
            np.ndarray,
        )
        res = tao.spin_invariant(
            who="l0",
            ix_uni="1",
            ix_branch="0",
            which="model",
            flags="",
        )
        assert "index" in res[0]


def test_spin_polarization_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.spin_polarization(ix_uni="1", ix_branch="0", which="model")
    assert isinstance(res, dict)
    assert "anom_moment_times_gamma" in res


def test_spin_resonance_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.spin_resonance(ix_uni="1", ix_branch="0", which="model")
    assert isinstance(res, dict)
    assert "spin_tune" in res


def test_super_universe_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.super_universe()
    assert isinstance(res, dict)
    assert "n_universe" in res
    assert "n_v1_var_used" in res
    assert "n_var_used" in res


def test_var_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        res = tao.var(var="quad[1]", slaves="")
    assert isinstance(res, dict)
    assert "weight" in res


def test_var_2(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        res = tao.var(var="quad[1]", slaves="slaves")
    assert isinstance(res[0], dict)
    assert "index" in res[0]


def test_var_general_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.var_general()
    assert isinstance(res[0], dict)
    assert "name" in res[0]


def test_var_v1_array_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.var_v1_array(v1_var="quad_k1")
    assert "ix_v1_var" in res
    assert "data" in res
    assert "name" in res["data"][0]


def test_lat_list_from_chris(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cesr/tao.init"
    ) as tao:
        names = tao.lat_list("*", "ele.name")
    assert isinstance(names[0], str)


def test_plot_graph_1(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        res = tao.plot_graph(graph_name="beta.g")
    assert isinstance(res, dict)
    assert "name" in res


def test_parse_version(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        date_version = tao.version(as_date=True)
        version = tao.version()
    assert isinstance(date_version, datetime)
    assert isinstance(version, str)


@pytest.mark.parametrize(
    ["lines", "expected_date", "expected_str"],
    [
        pytest.param(
            ["Version: 20260710-0"],
            datetime(2026, 7, 10),
            "20260710-0",
            id="tag",
        ),
        pytest.param(
            ["Version: 20260707-0-4-gec0e291a3"],
            datetime(2026, 7, 7),
            "20260707-0-4-gec0e291a3",
            id="git-describe",
        ),
        pytest.param(
            ["Date: 2026/07/07 00:00:00"],
            datetime(2026, 7, 7),
            "2026/07/07 00:00:00",
            id="date-fallback",
        ),
        pytest.param(
            ["garbage"],
            None,
            None,
            id="unparseable",
        ),
        pytest.param(
            ["Version: 20261301-0"],
            None,
            "20261301-0",
            id="invalid-date",
        ),
    ],
)
def test_parse_show_version(
    lines: list[str], expected_date: datetime | None, expected_str: str | None
):
    assert parse_show_version(lines, as_date=True) == expected_date
    assert parse_show_version(lines) == expected_str


def test_parse_wall3d_radius(caplog, tao_cls: type[AnyTao]):
    with ensure_successful_parsing(caplog):
        with new_tao(
            tao_cls,
            "-init $ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d",
            external_plotting=False,
        ) as tao:
            radius = tao.wall3d_radius(
                ix_uni=1,
                ix_branch=0,
                s_position=0.0,
                angle=0.0,
                verbose=True,
            )
        assert isinstance(radius, dict)
        assert len(radius["origin"]) == 3
        assert len(radius["perpendicular"]) == 3
        assert isinstance(radius["wall_radius"], float)


def test_parse_derivative_single_universe():
    """Test parse_derivative with a single universe and simple matrix."""
    # Single universe (iu=1) with 3 data points and 5 variables
    lines = [
        "1;1;1;1.0;2.0;3.0;4.0;5.0",
        "1;2;1;6.0;7.0;8.0;9.0;10.0",
        "1;3;1;11.0;12.0;13.0;14.0;15.0",
    ]

    result = parse_derivative(lines)

    # Check structure
    assert isinstance(result, dict)
    assert 1 in result
    assert isinstance(result[1], np.ndarray)

    # Check shape (3 data points x 5 variables)
    assert result[1].shape == (3, 5)

    # Check values
    expected = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
        ]
    )
    np.testing.assert_array_equal(result[1], expected)


def test_parse_derivative_multiple_universes():
    """Test parse_derivative with multiple universes."""
    lines = [
        "1;1;1;1.0;2.0",
        "1;2;1;3.0;4.0",
        "2;1;1;5.0;6.0",
        "2;2;1;7.0;8.0",
        "2;3;1;9.0;10.0",
    ]

    result = parse_derivative(lines)

    # Check both universes exist
    assert 1 in result
    assert 2 in result

    # Check shapes
    assert result[1].shape == (2, 2)
    assert result[2].shape == (3, 2)

    # Check values
    np.testing.assert_array_equal(result[1], np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(result[2], np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]))


def test_parse_derivative_chunked_variables():
    """Test parse_derivative when variables are split across multiple lines (>10 vars)."""
    # Single data point with 15 variables split into two lines
    # First line: variables 1-10, second line: variables 11-15
    lines = [
        "1;1;1;1.0;2.0;3.0;4.0;5.0;6.0;7.0;8.0;9.0;10.0",
        "1;1;11;11.0;12.0;13.0;14.0;15.0",
    ]

    result = parse_derivative(lines)

    assert result[1].shape == (1, 15)
    expected = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]
    )
    np.testing.assert_array_equal(result[1], expected)


def test_parse_derivative_sparse_matrix():
    """Test parse_derivative with non-contiguous data indices."""
    # Data indices 1, 3, 5 (not all consecutive)
    lines = [
        "1;1;1;1.0;2.0",
        "1;3;1;3.0;4.0",
        "1;5;1;5.0;6.0",
    ]

    result = parse_derivative(lines)

    # Matrix should be sized for max data index (5)
    assert result[1].shape == (5, 2)

    # Check that specified rows have values
    np.testing.assert_array_equal(result[1][0, :], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(result[1][2, :], np.array([3.0, 4.0]))
    np.testing.assert_array_equal(result[1][4, :], np.array([5.0, 6.0]))

    # Check that unspecified rows are NaN
    assert np.isnan(result[1][1, 0])
    assert np.isnan(result[1][3, 0])


def test_parse_derivative_complex_chunking():
    """Test parse_derivative with multiple data points and chunked variables."""
    # 2 data points, 12 variables each (chunked into 10 + 2)
    lines = [
        "1;1;1;1.0;2.0;3.0;4.0;5.0;6.0;7.0;8.0;9.0;10.0",
        "1;1;11;11.0;12.0",
        "1;2;1;21.0;22.0;23.0;24.0;25.0;26.0;27.0;28.0;29.0;30.0",
        "1;2;11;31.0;32.0",
    ]

    result = parse_derivative(lines)

    assert result[1].shape == (2, 12)

    # Check first row
    expected_row1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    np.testing.assert_array_equal(result[1][0, :], expected_row1)

    # Check second row
    expected_row2 = np.array(
        [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]
    )
    np.testing.assert_array_equal(result[1][1, :], expected_row2)


def test_parse_derivative_mixed_universes_and_chunking():
    """Test parse_derivative with multiple universes and chunked variables."""
    lines = [
        # Universe 1: 1 data point, 11 variables
        "1;1;1;1.0;2.0;3.0;4.0;5.0;6.0;7.0;8.0;9.0;10.0",
        "1;1;11;11.0",
        # Universe 2: 2 data points, 3 variables each
        "2;1;1;21.0;22.0;23.0",
        "2;2;1;24.0;25.0;26.0",
    ]

    result = parse_derivative(lines)

    # Check universe 1
    assert result[1].shape == (1, 11)
    expected_u1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]])
    np.testing.assert_array_equal(result[1], expected_u1)

    # Check universe 2
    assert result[2].shape == (2, 3)
    expected_u2 = np.array([[21.0, 22.0, 23.0], [24.0, 25.0, 26.0]])
    np.testing.assert_array_equal(result[2], expected_u2)


@pytest.mark.parametrize(
    ["value", "expected"],
    [
        pytest.param("", None),
        pytest.param("1", 1.0),
        pytest.param("1-5", 1e-5),
        pytest.param("1+5", 1e5),
        pytest.param("-1-5", -1e-5),
        pytest.param("+1-5", 1e-5),
    ],
)
def test_fix_sci_notation(value: str, expected):
    assert value_float_or_none(value) == expected


def test_ele_cartesian_map(tao_cls):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_em_field",
        external_plotting=False,
    ) as tao:
        base = tao.ele_cartesian_map(
            ele_id="1@0>>1", which="model", index="1", who="base", verbose=True
        )
        assert isinstance(base, dict)
        assert "file" in base

        terms = tao.ele_cartesian_map(
            ele_id="1@0>>1", which="model", index="1", who="terms", verbose=True
        )
        assert isinstance(terms, list)
        assert len(terms)
        assert all("coef" in term for term in terms)
        assert all("family" in term for term in terms)


def test_parse_bunch_comb():
    lines = ["0;  1.00000000000000E+00", "1;  2.50000000000000E-01"]
    np.testing.assert_allclose(parse_bunch_comb(lines), [1.0, 0.25])

    passthrough = np.array([1.0, 2.0])
    assert parse_bunch_comb(passthrough) is passthrough


def test_parse_da_aperture():
    lines = [
        "1;1;  1.000000E-02;  2.000000E-03",
        "1;2; -1.000000E-02;  2.000000E-03",
    ]
    points = parse_da_aperture(lines)
    assert points == [
        {"ix_scan": 1, "ix_point": 1, "x": 0.01, "y": 0.002},
        {"ix_scan": 1, "ix_point": 2, "x": -0.01, "y": 0.002},
    ]


def test_parse_ele_ac_kicker():
    assert parse_ele_ac_kicker([]) is None

    amp_vs_time = parse_ele_ac_kicker(
        [
            "has#amp_vs_time",
            "1;  1.00000000000000E+00;  2.00000000000000E-09",
        ]
    )
    assert amp_vs_time == {
        "mode": "amp_vs_time",
        "data": [{"index": 1, "amp": 1.0, "time": 2e-9}],
    }

    frequencies = parse_ele_ac_kicker(
        [
            "has#frequencies",
            "1;  1.30000000000000E+09;  5.00000000000000E-01;  1.57000000000000E+00",
        ]
    )
    assert frequencies == {
        "mode": "frequencies",
        "data": [{"index": 1, "frequency": 1.3e9, "amp": 0.5, "phi": 1.57}],
    }


def test_parse_ele_cylindrical_map_terms():
    terms = parse_ele_cylindrical_map(
        [
            "1;  1.00000000000000E+00; -2.00000000000000E+00;  3.00000000000000E+00;  4.00000000000000E+00"
        ],
        cmd="pipe ele:cylindrical_map m1|model 1 terms",
    )
    assert terms == [{"index": 1, "e_coef": 1 - 2j, "b_coef": 3 + 4j}]


@pytest.mark.parametrize(
    ["who", "values", "expected_shape"],
    [
        pytest.param("ele.mat6", 36, (6, 6)),
        pytest.param("ele.vec0", 6, (6,)),
        pytest.param("ele.c_mat", 4, (2, 2)),
    ],
)
def test_parse_ele_param_matrix(who: str, values: int, expected_shape: tuple[int, ...]):
    line = f"{who};REAL;F;" + ";".join(f"  {float(i)}E+00" for i in range(values))
    (value,) = parse_ele_param([line]).values()
    assert value.shape == expected_shape
    np.testing.assert_allclose(value.ravel(), np.arange(values, dtype=float))


def test_parse_ele_param_scalar():
    assert parse_ele_param(["orbit.vec.1;REAL;F;  1.50000000000000E-03"]) == {
        "orbit_vec_1": 0.0015
    }


def test_parse_wave_params():
    data = parse_wave(
        [
            "wave_data_type;ENUM;T;cbar.12",
            "ix_a1;INT;T;10",
            "A Region Sigma_+/Amp_+;REAL;F;   0.023",
            "Kick |K+|  12.345",
            "Sigma_K+/K+********",
        ],
        cmd="pipe wave params",
    )
    assert isinstance(data, dict)
    assert data["wave_data_type"] == "cbar.12"
    assert data["ix_a1"] == 10
    assert data["A Region Sigma_+/Amp_+"] == 0.023
    assert data["Kick |K+|"] == 12.345
    assert math.isnan(data["Sigma_K+/K+"])


def test_parse_wave_loc_header():
    header = parse_wave(
        [
            "header1;STR;F;Normalized Kick = kick * sqrt(beta)  [urad * sqrt(meter)]",
            "columns;After Dat#;Norm_Kick;s;ix_ele;ele@kick;phi",
        ],
        cmd="pipe wave loc_header",
    )
    assert header == {
        "header1": "Normalized Kick = kick * sqrt(beta)  [urad * sqrt(meter)]",
        "columns": ["After Dat#", "Norm_Kick", "s", "ix_ele", "ele@kick", "phi"],
    }


def test_parse_wave_locations():
    assert parse_wave(
        ["23;12.34;145.20;678;Q03W;0.523"],
        cmd="pipe wave locations",
    ) == [
        {
            "ix_dat_before_kick": 23,
            "amp": 12.34,
            "s": 145.2,
            "ix_ele": 678,
            "ele_name": "Q03W",
            "phi": 0.523,
        }
    ]

    assert parse_wave(
        ["23;0.1234;145.20;678;Q03W;0.523;0.312;0.417;0.105"],
        cmd="pipe wave locations",
    ) == [
        {
            "ix_dat_before_kick": 23,
            "amp": 0.1234,
            "s": 145.2,
            "ix_ele": 678,
            "ele_name": "Q03W",
            "phi_s": 0.523,
            "phi_r": 0.312,
            "phi_a": 0.417,
            "phi_b": 0.105,
        }
    ]


def test_parse_wave_plot():
    assert parse_wave(
        ["1;  1.234560E+00; -2.000000E-01"],
        cmd="pipe wave plot1",
    ) == [{"index": 1, "x": 1.23456, "y": -0.2}]


def test_bunch_comb_string_output(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/regression_tests/pipe_test/csr_beam_tracking/tao.init",
        external_plotting=False,
    ) as tao:
        from_strings = tao.bunch_comb(who="x.beta", flags="")
        from_buffer = tao.bunch_comb(who="x.beta")
        assert isinstance(from_strings, np.ndarray)
        np.testing.assert_allclose(from_strings, from_buffer)


def test_ele_cylindrical_map(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_em_field",
        external_plotting=False,
    ) as tao:
        base = tao.ele_cylindrical_map(ele_id="m1", which="model", index="1", who="base")
        assert isinstance(base, dict)
        assert "file" in base

        terms = tao.ele_cylindrical_map(ele_id="m1", which="model", index="1", who="terms")
        assert isinstance(terms, list)
        assert len(terms)
        assert all(isinstance(term["e_coef"], complex) for term in terms)


def test_da_aperture(tao_cls: type[AnyTao]):
    # `set dynamic_aperture` segfaults Tao when the init file lacks a
    # &tao_dynamic_aperture namelist (unallocated %pz), so the scan is
    # configured entirely at startup here.
    init_file = test_root / "input_files" / "dynamic_aperture" / "tao.init"
    with new_tao(tao_cls, init_file=str(init_file)) as tao:
        points = tao.da_aperture(ix_uni="1")
        assert {point["ix_scan"] for point in points} == {1}
        assert {point["ix_point"] for point in points} == {1, 2, 3}
        assert all(point["y"] >= 0 for point in points)


@pytest.mark.parametrize(
    ["type_", "value", "expected"],
    [
        pytest.param("REAL", ["1.42000000000000+245"], 1.42e245),
        pytest.param("REAL", ["-1.42000000000000-245"], -1.42e-245),
        pytest.param("REAL_ARR", ["1.0", "1.42+245"], np.array([1.0, 1.42e245])),
        pytest.param("COMPLEX", ["1.0-300", "2.0+300"], complex(1e-300, 2e300)),
    ],
)
def test_parse_pytype_malformed_exponent(type_: str, value: list[str], expected):
    parsed = parse_pytype(type_, value)
    if isinstance(expected, np.ndarray):
        assert isinstance(parsed, np.ndarray)
        np.testing.assert_allclose(parsed, expected)
    else:
        assert parsed == expected


def test_malformed_exponents_in_parsers():
    assert parse_evaluate(["1;  1.42000000000000+245"]) == [1.42e245]
    assert parse_merit(["  1.00000000000000+100"]) == 1e100

    taylor = parse_taylor_map(["1;1;  1.00000000000000-300;1;0;0;0;0;0"])
    assert taylor[1][(1, 0, 0, 0, 0, 0)] == 1e-300

    matrix = parse_matrix(["1;1-300;0.0;0.0;0.0;0.0;0.0;2+300"])
    assert matrix["mat6"][0, 0] == 1e-300
    assert matrix["vec0"][0] == 2e300

    points = parse_ele_grid_field(
        ["1;2;3;  1.00000000000000+300; NaN"],
        cmd="pipe ele:grid_field 1@0>>1|model 1 points",
    )
    assert isinstance(points, list)
    assert points[0]["data"][0] == 1e300
    assert math.isnan(points[0]["data"][1])

    var_line = parse_var_v_array_line("1;q[k1];1-300;2-300;3-300;T;T;1+300")
    assert var_line["meas_value"] == 1e-300
    assert var_line["weight"] == 1e300


@pytest.mark.parametrize(
    ["parser", "cmd"],
    [
        pytest.param(parsers.parse_bunch_comb, "", id="bunch_comb"),
        pytest.param(parsers.parse_da_aperture, "", id="da_aperture"),
        pytest.param(parsers.parse_data_d_array, "", id="data_d_array"),
        pytest.param(parsers.parse_data_d1_array, "", id="data_d1_array"),
        pytest.param(parsers.parse_data_d2_array, "", id="data_d2_array"),
        pytest.param(
            parsers.parse_data_parameter,
            "pipe data_parameter twiss.end meas_value",
            id="data_parameter",
        ),
        pytest.param(parsers.parse_datum_has_ele, "", id="datum_has_ele"),
        pytest.param(parsers.parse_derivative, "", id="derivative"),
        pytest.param(parsers.parse_ele_ac_kicker, "", id="ele_ac_kicker"),
        pytest.param(
            parsers.parse_ele_cartesian_map,
            "pipe ele:cartesian_map 1 1 terms",
            id="ele_cartesian_map",
        ),
        pytest.param(parsers.parse_ele_chamber_wall, "", id="ele_chamber_wall"),
        pytest.param(parsers.parse_ele_control_var, "", id="ele_control_var"),
        pytest.param(
            parsers.parse_ele_cylindrical_map,
            "pipe ele:cylindrical_map 1 1 terms",
            id="ele_cylindrical_map",
        ),
        pytest.param(parsers.parse_ele_elec_multipoles, "", id="ele_elec_multipoles"),
        pytest.param(
            parsers.parse_ele_grid_field, "pipe ele:grid_field 1 1 points", id="ele_grid_field"
        ),
        pytest.param(parsers.parse_ele_multipoles, "", id="ele_multipoles"),
        pytest.param(parsers.parse_ele_taylor, "", id="ele_taylor"),
        pytest.param(parsers.parse_ele_wall3d, "pipe ele:wall3d 1 1 table", id="ele_wall3d"),
        pytest.param(parsers.parse_ele_wake, "pipe ele:wake 1 sr_long_table", id="ele_wake"),
        pytest.param(parsers.parse_em_field, "", id="em_field"),
        pytest.param(parsers.parse_enum, "", id="enum"),
        pytest.param(parsers.parse_evaluate, "", id="evaluate"),
        pytest.param(parsers.parse_floor_plan, "", id="floor_plan"),
        pytest.param(parsers.parse_floor_orbit, "", id="floor_orbit"),
        pytest.param(parsers.parse_inum, "", id="inum"),
        pytest.param(parsers.parse_lat_ele_list, "", id="lat_ele_list"),
        pytest.param(parsers.parse_lat_list, "", id="lat_list"),
        pytest.param(parsers.parse_lat_param_units, "", id="lat_param_units"),
        pytest.param(parsers.parse_matrix, "", id="matrix"),
        pytest.param(parsers.parse_merit, "", id="merit"),
        pytest.param(parsers.parse_plot_list, "pipe plot_list r", id="plot_list"),
        pytest.param(parsers.parse_species_to_int, "", id="species_to_int"),
        pytest.param(parsers.parse_species_to_str, "", id="species_to_str"),
        pytest.param(parsers.parse_taylor_map, "", id="taylor_map"),
        pytest.param(parsers.parse_var_v_array, "", id="var_v_array"),
        pytest.param(parsers.parse_wave, "pipe wave params", id="wave"),
    ],
)
def test_invalid_raises(monkeypatch, parser, cmd: str):
    with pytest.raises(TaoDataInvalidError):
        parser(["INVALID"], cmd=cmd)


def test_invalid_appended_after_data(monkeypatch):
    # Tao's invalid() appends INVALID after any lines already written.
    monkeypatch.setattr(parsers.Settings, "ensure_count", False)
    with pytest.raises(TaoDataInvalidError):
        parse_ele_param(["taylor_map_includes_offsets;LOGIC;T;F", "INVALID"])
    with pytest.raises(TaoDataInvalidError):
        parsers.parse_ele_taylor(["taylor_map_includes_offsets;LOGIC;T;F", "INVALID"])


def test_em_field_empty_output_raises():
    with pytest.raises(TaoDataInvalidError):
        parsers.parse_em_field([])
