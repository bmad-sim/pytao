from datetime import datetime
from typing import Type

import numpy as np
import pytest

from .. import AnyTao
from .test_interface_commands import new_tao


def test_building_wall_list_1(tao_cls: Type[AnyTao]):
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


def test_building_wall_list_2(tao_cls: Type[AnyTao]):
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


def test_building_wall_graph_1(tao_cls: Type[AnyTao]):
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


def test_constraints_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        tao.constraints(who="data")


def test_constraints_2(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        tao.constraints(who="var")


def test_data_d2_array_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert "orbit" in tao.data_d2_array(ix_uni="1")


def test_data_parameter_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        assert (
            tao.data_parameter(data_array="twiss.end", parameter="model_value")[0]["index"]
            == 1
        )


def test_datum_has_ele_1(tao_cls: Type[AnyTao]):
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


def test_ele_chamber_wall_1(tao_cls: Type[AnyTao]):
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


def test_ele_elec_multipoles_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert "data" in tao.ele_elec_multipoles(ele_id="1@0>>1", which="model")


def test_ele_gen_grad_map_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_em_field"
    ) as tao:
        assert set(
            tao.ele_gen_grad_map(ele_id="1@0>>9", which="model", index="1", who="derivs")[
                0
            ].keys()
        ) == {"i", "j", "k", "dz", "deriv"}


def test_ele_lord_slave_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert set(tao.ele_lord_slave(ele_id="1@0>>1", which="model")[0].keys()) == {
            "type",
            "location_name",
            "name",
            "key",
            "status",
        }


def test_ele_multipoles_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.ele_multipoles(ele_id="1@0>>1", which="model")

    assert isinstance(res, dict)
    if res["data"]:
        assert "KnL" in res or "An" in res["data"][0]


def test_ele_taylor_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_taylor"
    ) as tao:
        res = tao.ele_taylor(ele_id="1@0>>34", which="model")
    assert isinstance(res, dict)
    assert "data" in res
    assert res["data"][0]["index"] == 1


def test_ele_spin_taylor_1(tao_cls: Type[AnyTao]):
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


def test_ele_wall3d_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d"
    ) as tao:
        res = tao.ele_wall3d(ele_id="1@0>>1", which="model", index="1", who="table")
    assert "data" in res[0]
    assert res[0]["section"] == 1


def test_em_field_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.em_field(ele_id="1@0>>22", which="model", x="0", y="0", z="0", t_or_z="0")
    assert "B1" in res


def test_enum_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.enum(enum_name="tracking_method")
    assert set(res[0].keys()) == {"number", "name"}


def test_floor_plan_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        tao.cmd("place -no_buffer r13.g floor_plan")
        res = tao.floor_plan(graph="r13.g")
    assert "branch_index" in res[0]


def test_floor_orbit_1(tao_cls: Type[AnyTao]):
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


def test_help_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        print(tao.help())


def test_inum_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.inum(who="ix_universe")
    assert isinstance(res, list)
    if res:
        assert isinstance(res[0], int)


def test_lat_calc_done_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert tao.lat_calc_done(branch_name="1@0") in {True, False}


def test_lat_branch_list_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        tao.lat_branch_list(ix_uni="1")


def test_lat_param_units_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert isinstance(tao.lat_param_units(param_name="L"), str)


def test_plot_lat_layout_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert "ix_ele" in tao.plot_lat_layout(ix_uni="1", ix_branch="0")[0]


def test_plot_line_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_plot_line",
    ) as tao:
        res = tao.plot_line(region_name="beta", graph_name="g", curve_name="a", x_or_y="")
    assert "x" in res[0]


def test_plot_line_2(tao_cls: Type[AnyTao]):
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


def test_plot_symbol_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_plot_line",
    ) as tao:
        res = tao.plot_symbol(region_name="r13", graph_name="g", curve_name="a", x_or_y="")
    assert "index" in res[0]


def test_shape_list_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert "shape_index" in tao.shape_list(who="floor_plan")[0]


def test_shape_pattern_list_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_shape"
    ) as tao:
        res = tao.shape_pattern_list(ix_pattern="")
    assert set(res[0].keys()) == {
        "name",
        "line_width",
    }


def test_show_1(tao_cls: Type[AnyTao]):
    pytest.skip("TODO")
    tao = new_tao(init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init")
    tao.show(line="-python")


def test_species_to_int_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert isinstance(tao.species_to_int(species_str="electron"), int)


def test_species_to_str_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        assert isinstance(tao.species_to_str(species_int="-1"), str)


def test_spin_invariant_1(tao_cls: Type[AnyTao]):
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


def test_spin_polarization_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.spin_polarization(ix_uni="1", ix_branch="0", which="model")
    assert isinstance(res, dict)
    assert "anom_moment_times_gamma" in res


def test_spin_resonance_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.spin_resonance(ix_uni="1", ix_branch="0", which="model")
    assert isinstance(res, dict)
    assert "spin_tune" in res


def test_super_universe_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.super_universe()
    assert isinstance(res, dict)
    assert "n_universe" in res
    assert "n_v1_var_used" in res
    assert "n_var_used" in res


def test_var_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        res = tao.var(var="quad[1]", slaves="")
    assert isinstance(res, dict)
    assert "weight" in res


def test_var_2(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        res = tao.var(var="quad[1]", slaves="slaves")
    assert isinstance(res[0], dict)
    assert "index" in res[0]


def test_var_general_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.var_general()
    assert isinstance(res[0], dict)
    assert "name" in res[0]


def test_var_v1_array_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init"
    ) as tao:
        res = tao.var_v1_array(v1_var="quad_k1")
    assert "ix_v1_var" in res
    assert "data" in res
    assert "name" in res["data"][0]


def test_lat_list_from_chris(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cesr/tao.init"
    ) as tao:
        names = tao.lat_list("*", "ele.name")
    assert isinstance(names[0], str)


def test_plot_graph_1(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        res = tao.plot_graph(graph_name="beta.g")
    assert isinstance(res, dict)
    assert "name" in res


def test_parse_version(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls,
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_optics_matching",
    ) as tao:
        res = tao.version()
    assert isinstance(res, datetime)
