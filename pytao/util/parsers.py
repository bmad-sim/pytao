import ast
import datetime
import logging
from typing import Dict, List, Optional

import numpy as np

from ..tao_ctypes.util import parse_bool, parse_tao_python_data

logger = logging.getLogger(__name__)


class Settings:
    ensure_count: bool = False


# Helpers
def _parse_str_bool(s):
    """
    parses str to bool
    '1', 't', 'T' -> True
    '0', 'f', 'F' -> False
    """
    x = s.upper()[0]
    if x in ("T", "1"):
        return True
    elif x in ("F", "0"):
        return False
    else:
        raise ValueError("Unknown bool: " + s)


# Column names and types for parse_data_d_array
DATA_D_COLS = [
    "ix_d1",
    "data_type",
    "merit_type",
    "ele_ref_name",
    "ele_start_name",
    "ele_name",
    "meas_value",
    "model_value",
    "design_value",
    "useit_opt",
    "useit_plot",
    "good_user",
    "weight",
    "exists",
]
DATA_D_TYPES = [
    int,
    str,
    str,
    str,
    str,
    str,
    float,
    float,
    float,
    bool,
    bool,
    bool,
    float,
    bool,
]


def parse_data_d_array(lines, cmd=""):
    """
    Parses the output of the 'python data_d_array' command into a list of dicts.

    This can be easily be case into a table. For example:

    import pandas as pd
    ...
    lines = tao.data_d_array('orbit', 'x')
    dat = parse_data_d_array(lines)
    df = pd.DataFrame(dat)


    Parameters
    ----------
    lines : list of str
        The output of the 'python data_d_array' command to parse

    Returns
    -------
    datums: list of dicts
            Each dict has keys:
            'ix_d1', 'data_type', 'merit_type',
            'ele_ref_name', 'ele_start_name', 'ele_name',
            'meas_value', 'model_value', 'design_value',
            'useit_opt', 'useit_plot', 'good_user',
            'weight', 'exists'

    """
    result = []
    for line in lines:
        d = {}
        result.append(d)
        vals = line.split(";")
        for name, typ, val in zip(DATA_D_COLS, DATA_D_TYPES, vals):
            if typ is bool:
                val = _parse_str_bool(val)
            else:
                val = typ(val)
            d[name] = val

    return result


def parse_derivative(lines, cmd=""):
    """
    Parses the output of tao python derivative

    Parameters
    ----------
    lines : list of str
        The output of the 'python derivative' command to parse

    Returns
    -------
    out : dict
        Dictionary with keys corresponding to universe indexes (int),
        with dModel_dVar as the value:
            np.ndarray with shape (n_data, n_var)
    """
    universes = {}

    # Build up matrices
    for line in lines:
        x = line.split(";")
        if len(x) <= 1:
            continue
        iu = int(x[0])

        if iu not in universes:
            # new universe
            rows = universes[iu] = []
            rowdat = []
            row_id = int(x[1])

        if int(x[1]) == row_id:
            # accumulate more data
            rowdat += x[3:]
        else:
            # Finish row
            rows.append(rowdat)
            rowdat = x[3:]
            row_id = int(x[1])

    # cast to float
    out = {}
    for iu, vals in universes.items():
        out[iu] = np.array(vals).astype(float)

    return out


def parse_ele_control_var(lines, cmd=""):
    """
    Parses the output of tao python ele_control_var

    Parameters
    ----------
    lines : list of str
        The output of the 'python ele_control_var' command to parse

    Returns
    -------
    dict of attributes and values

    """
    d = {}
    for line in lines:
        try:
            # Group controller var has an old_value. Overlay and ramper vars do
            # not.  Ignore 'old_value' here.
            ix, name, value, *_ = line.split(";")
        except ValueError:
            logger.warning("Skipping value: %s", line)
        else:
            d[name] = float(value)
    return d


def parse_lat_ele_list(lines, cmd=""):
    """
    Parses the output of tao python lat_ele_list

    Parameters
    ----------
    lines : list of str
        The output of the 'python lat_ele_list' command to parse

    Returns
    -------
    list of str of element names

    """

    return [s.split(";")[1] for s in lines]


def parse_matrix(lines, cmd=""):
    """
    Parses the output of a tao python matix

    Parameters
    ----------
    lines : list of str
        The output of the 'python matrix' command to parse

    Returns
    -------
    dict with keys:
        'mat6' : np.array of shape (6,6)
        'vec6' : np.array of shape(6)


    """
    m7 = np.array([[float(x) for x in line.split(";")[1:]] for line in lines])
    return {"mat6": m7[:, 0:6], "vec0": m7[:, 6]}


def parse_merit(lines, cmd=""):
    """
    Parses the output of a tao python merit

    Parameters
    ----------
    lines : list of str
        The output of the 'python matrix' command to parse

    Returns
    -------
    merit: float
        Value of the merit function
    """
    assert len(lines) == 1
    return float(lines[0])


def parse_plot_list(lines, cmd=""):
    """
    Parses the output of the `python plot_list` command.

    This could be region or template data.


    Parameters
    ----------
    lines : list of str
        The output of the 'python plot_list' command to parse

    Returns
    -------
    if r_or_g == 't'
        dict with template_name:index

    if r_or_g == 'r'
        list of dicts with keys:
            region
            ix
            plot_name
            visible
            x1, x2, y1, y1

    """

    # infer region or template output
    nv = len(lines[0].split(";"))

    if nv == 2:
        # Template
        output = {}
        for line in lines:
            ix, name = line.split(";")
            output[name] = int(ix)

    elif nv == 8:
        # Region8
        output = []
        for line in lines:
            ix, region_name, plot_name, visible, x1, x2, y1, y2 = line.split(";")
            output.append(
                {
                    "region": region_name,
                    "ix": int(ix),
                    "plot_name": plot_name,
                    "visible": _parse_str_bool(visible),
                    "x1": float(x1),
                    "x2": float(x2),
                    "y1": float(y1),
                    "y2": float(y2),
                }
            )

    else:
        raise ValueError(f"Cannot parse {lines[0]}")

    return output


def parse_spin_invariant(lines, cmd=""):
    """
    Reshape the (3*n) shaped array output of `spin_invariant`
    to be (n, 3)

    Do nothing with lines (list) output.

    """
    if isinstance(lines, np.ndarray):
        return lines.reshape(len(lines) // 3, 3)

    return _parse_by_keys_to_types(
        lines,
        {
            "index": int,
            "spin1": float,
            "spin2": float,
            "spin3": float,
        },
    )


def parse_taylor_map(lines, cmd=""):
    """
    Parses the output of the `python taylor_map` command.

    Parameters
    ----------
    lines : list of str
        The output of the 'python taylor_map' command to parse

    Returns
    -------
    dict of dict of taylor terms:
        {2: { (3,0,0,0,0,0)}: 4.56, ...
            corresponding to: px_out = 4.56 * x_in^3


    """
    tt = {i: {} for i in range(1, 7)}
    for term_str in lines:
        t = term_str.split(";")
        out = int(t[0])
        coef = float(t[2])
        exponents = tuple([int(i) for i in t[3:]])
        tt[out][exponents] = coef
    return tt


def parse_var_v_array_line(line, cmd=""):
    v = line.split(";")
    out = dict(
        ix_v1=int(v[0]),
        var_attrib_name=v[1],
        meas_value=float(v[2]),
        model_value=float(v[3]),
        design_value=float(v[4]),
        useit_opt=_parse_str_bool(v[5]),
        good_user=_parse_str_bool(v[6]),
        weight=float(v[7]),
    )
    return out


def parse_var_v_array(lines, cmd=""):
    """
    Parses the output of `python var_v_array` into a list of dicts

    Returns
    -------
    list of dict
    """
    return [parse_var_v_array_line(line) for line in lines]


def fix_value(value: str, typ: type):
    """
    Apply some fixes for known problematic tao output.

    Parameters
    ----------
    value : str
        The tao output value string.
    typ : type
        The expected Python type.

    Returns
    -------
    typ
    """
    value = value.strip()
    if typ is bool:
        return _parse_str_bool(value)
    if typ is float:
        if ("-" in value or "+" in value) and "e" not in value:
            # TODO: some floating point values like gg%deriv of ele_gen_grad_map
            # are formatted incorrectly
            try:
                return float(value)
            except ValueError:
                return float(value.replace("-", "e-").replace("+", "e+"))

    return typ(value)


def _parse_by_keys_to_types(
    lines: List[str],
    key_to_type: Dict[str, type],
    ensure_count: Optional[bool] = None,
) -> List[dict]:
    """
    Parse Tao command output, with predetermined field names and associated types.

    Each output line is converted according to ``key_to_type``, such that ``N``
    lines of output will result in N dictionaries with keys
    ``key_to_type.keys()`` with corresponding values cast to the indicated
    type.

    Parameters
    ----------
    lines : List[str]
        Raw Tao output.
    key_to_type : Dict[str, type]
        Dictionary of key name to expected Python type.
    ensure_count : bool, optional
        Fail if the number of output fields doesn't match up with the expected
        ones in ``key_to_type``.
        Defaults to ``Settings.ensure_count`` which can be easily toggled
        application-wide.  This is only enabled by default for the test suite.

    Returns
    -------
    list of dict
    """
    if ensure_count is None:
        ensure_count = Settings.ensure_count

    if ensure_count:
        for line in lines:
            assert len(key_to_type) == len(line.split(";"))

    return [
        {
            key: fix_value(value, typ)
            for (key, typ), value in zip(
                key_to_type.items(), line.split(";", len(key_to_type))
            )
        }
        for line in lines
    ]


def _get_cmd_args(cmd: str) -> List[str]:
    """
    Get command arguments.

    (python) (command) [(arg1) (arg2) ... (argN)]

    Parameters
    ----------
    cmd : str
        The raw Tao command, including "python" as the first argument.

    Returns
    -------
    list of str
    """
    _python, _cmd, *args = cmd.strip().split()
    return args


def parse_building_wall_list(lines, cmd=""):
    """
    Parse building_wall_list results.

    Returns
    -------
    list of dicts
    """
    args = _get_cmd_args(cmd)
    if args:
        # global floor positions
        return _parse_by_keys_to_types(
            lines,
            {
                "index": int,
                "z": float,
                "x": float,
                "radius": float,
                "z_center": float,
                "x_center": float,
            },
        )
    return _parse_by_keys_to_types(
        lines,
        {
            "index": int,
            "name": str,
            "constraint": str,
            "shape": str,
            "color": str,
            "line_width": float,
        },
    )


def parse_building_wall_graph(lines, cmd=""):
    """
    Parse building_wall_graph results.

    Returns
    -------
    list of dicts
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "index": int,
            "point": int,
            "offset_x": float,
            "offset_y": float,
            "radius": float,
        },
    )


def parse_constraints(lines, cmd=""):
    """
    Parse constraints results.

    Returns
    -------
    list of dicts
        The keys depend on "data" or "var"
    """
    args = _get_cmd_args(cmd)
    if args and args[0] == "data":
        return _parse_by_keys_to_types(
            lines,
            {
                "datum_name": str,
                "constraint_type_name": str,
                "ele_name": str,
                "ele_start_name": str,
                "ele_ref_name": str,
                "meas_value": float,
                "ref_value": float,
                "model_value": float,
                "base_value": float,
                "weight": float,
                "merit": float,
                "a_name": str,
            },
        )
    elif args and args[0] == "var":
        return _parse_by_keys_to_types(
            lines,
            {
                "var1_name": str,
                "attrib_name": str,
                "meas_value": float,
                "ref_value": float,
                "model_value": float,
                "base_value": float,
                "weight": float,
                # "merit": float,
                # "merit_dvar": float,
            },
        )


def parse_data_d1_array(lines, cmd=""):
    """
    Parse data_d1_array results.

    Returns
    -------
    list of dicts
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "index": str,
            "str2": str,
            "f": str,
            "name": str,
            "line": str,
            "lower_bound": int,
            "upper_bound": int,
        },
    )


def parse_data_d2_array(lines, cmd=""):
    """
    Parse data_d2_array results.

    Returns
    -------
    list of str
    """
    return lines


def parse_data_parameter(lines, cmd=""):
    """
    Parse parameter_1 results.

    Returns
    -------
    list of dict
    """
    args = _get_cmd_args(cmd)
    if len(args) < 2:
        return
    expected_type = {
        "data_type": str,
        "ele_name": str,
        "ele_start_name": str,
        "ele_ref_name": str,
        "merit_type": str,
        "id": str,
        "data_source": str,
        "ix_uni": int,
        "ix_bunch": int,
        "ix_branch": int,
        "ix_ele": int,
        "ix_ele_start": int,
        "ix_ele_ref": int,
        "ix_ele_merit": int,
        "ix_d1": int,
        "ix_data": int,
        "ix_dModel": int,
        "eval_point": int,
        "meas_value": float,
        "ref_value": float,
        "model_value": float,
        "design_value": float,
        "old_value": float,
        "base_value": float,
        "error_rms": float,
        "delta_merit": float,
        "weight": float,
        "invalid_value": float,
        "merit": float,
        "s": float,
        "s_offset": float,
        "err_message_printed": bool,
        "exists": bool,
        "good_model": bool,
        "good_base": bool,
        "good_design": bool,
        "good_meas": bool,
        "good_ref": bool,
        "good_user": bool,
        "good_opt": bool,
        "good_plot": bool,
        "useit_plot": bool,
        "useit_opt": bool,
    }.get(args[1], str)

    def fix_line(line):
        index, *values = line.split(";")
        return {
            "index": int(index),
            "data": [fix_value(val, expected_type) for val in values],
        }

    return [fix_line(line) for line in lines]


def parse_datum_has_ele(lines, cmd=""):
    """
    Parse datum_has_ele results.

    Returns
    -------
    str or None
        "no", "yes", "maybe", "provisional"
    """
    return lines[0] if lines else None


def parse_ele_chamber_wall(lines, cmd=""):
    """
    Parse ele_chamber_wall results.

    Returns
    -------
    list of dict
    """
    return _parse_by_keys_to_types(
        lines,
        {"section": int, "longitudinal_position": float, "z1": float, "-z2": float},
    )


def parse_ele_elec_multipoles(lines, cmd=""):
    """
    Parse ele_elec_multipoles results.

    Returns
    -------
    dict
    """
    logic_lines = [line for line in lines if "LOGIC" in line]
    lines = [line for line in lines if line not in logic_lines]
    key_to_type = {key: float for key in lines[0].split(";")}
    settings = {}
    for line in logic_lines:
        # parse_tao_python_data1 doesn't work as it's missing 'settable'
        # (line) for line in logic_lines
        name, _type, value = line.split(";")
        settings[name] = parse_bool(value)

    # TODO: 'data' is not actually parsed in the test suite
    return {
        **settings,
        "data": _parse_by_keys_to_types(
            lines[1:],
            key_to_type,
        ),
    }


def parse_ele_gen_grad_map(lines, cmd=""):
    """
    Parse ele_gen_grad_map results.

    Returns
    -------
    dict or list of dict
        "derivs" mode will be a list of dictionaries.
        Normal mode will be a single dictionary.
    """

    args = _get_cmd_args(cmd)
    if args[-1] == "derivs":
        return _parse_by_keys_to_types(
            lines,
            {
                "i": int,
                "j": int,
                "k": int,
                "dz": float,
                "deriv": float,
            },
        )
    return parse_tao_python_data(lines)


def parse_ele_lord_slave(lines, cmd=""):
    """
    Parse ele_lord_slave results.

    Returns
    -------
    list of dict
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "type": str,
            "location_name": str,
            "name": str,
            "key": str,
            "status": str,
        },
    )


def parse_ele_multipoles(lines, cmd=""):
    """
    Parse ele_multipoles results.

    Returns
    -------
    dict
    """
    logic_lines = [line for line in lines if "LOGIC" in line]
    lines = [line for line in lines if line not in logic_lines]
    key_to_type = {"index": int}
    for key in lines[0].split(";"):
        key_to_type[key] = float

    settings = parse_tao_python_data(logic_lines)
    return {
        **settings,
        "data": _parse_by_keys_to_types(
            lines[1:],
            key_to_type,
        ),
    }


def parse_ele_taylor(lines, cmd=""):
    """
    Parse ele_taylor results.

    Returns
    -------
    dict
    """

    def split_sections(lines):
        sections = []
        for line in lines:
            if ";ref;" in line:
                sections.append([line])
            else:
                sections[-1].append(line)
        return sections

    def parse_section(section):
        header = section[0]
        index, _, ref = header.split(";")
        info = {
            "index": int(index),
            "ref": float(ref),
        }
        info["data"] = _parse_by_keys_to_types(
            section[1:],
            {
                "i": int,
                "j": int,
                "coef": float,
                "exp1": float,
                "exp2": float,
                "exp3": float,
                "exp4": float,
                "exp5": float,
                "exp6": float,
            },
        )
        return info

    logic_lines = [line for line in lines if "LOGIC" in line]
    lines = [line for line in lines if line not in logic_lines]

    settings = parse_tao_python_data(logic_lines)
    sections = split_sections(lines)
    return {
        **settings,
        "data": [parse_section(section) for section in sections],
    }


def parse_ele_spin_taylor(lines, cmd=""):
    """
    Parse ele_spin_taylor results.

    Returns
    -------
    list of dict
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "index": int,
            "term": str,
            "coef": float,
            "exp1": float,
            "exp2": float,
            "exp3": float,
            "exp4": float,
            "exp5": float,
            "exp6": float,
        },
    )


def parse_ele_wall3d(lines, cmd=""):
    """
    Parse ele_wall3d results.

    Returns
    -------
    list of dict
    """

    def split_sections(lines):
        sections = []
        for line in lines:
            if line.startswith("section;"):
                sections.append([line])
            else:
                sections[-1].append(line)
        return sections

    def parse_section(section):
        header = []
        for line in section:
            if line[0].isalpha():
                header.append(line.replace(";;", ";"))  # TODO a bmad bug?
            else:
                break
        data = section[len(header) :]
        info = parse_tao_python_data(header)
        info["data"] = _parse_by_keys_to_types(
            data,
            {
                "j": int,
                "x": float,
                "y": float,
                "radius_x": float,
                "radius_y": float,
                "tilt": float,
            },
        )
        return info

    args = _get_cmd_args(cmd)
    if args[-1] == "table":
        sections = split_sections(lines)
        return [parse_section(section) for section in sections]

    return parse_tao_python_data(lines)


def parse_em_field(lines, cmd=""):
    """
    Parse em_field results.

    Returns
    -------
    dict
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "B1": float,
            "B2": float,
            "B3": float,
            "E1": float,
            "E2": float,
            "E3": float,
        },
    )[0]


def parse_enum(lines, cmd=""):
    """
    Parse enum results.

    Returns
    -------
    list of dict
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "number": int,
            "name": str,
        },
    )


def parse_floor_plan(lines, cmd=""):
    """
    Parse floor_plan results.

    Returns
    -------
    list of dict
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "branch_index": int,
            "index": int,
            "ele_key": str,
            "end1_r1": float,
            "end1_r2": float,
            "end1_theta": float,
            "end2_r1": float,
            "end2_r2": float,
            "end2_theta": float,
            "line_width": float,
            "shape": str,
            "y1": float,
            "y2": float,
            "color": str,
            "label_name": str,
            # Only for sbend:
            "ele_l": float,
            "ele_angle": float,
            "ele_e1": float,
            "ele_e": float,
        },
        ensure_count=False,
    )


def parse_floor_orbit(lines, cmd=""):
    """
    Parse floor_orbit results.

    Returns
    -------
    list of dict
    """
    res = []
    for line in lines:
        data = _parse_by_keys_to_types(
            [line],
            {
                "branch_index": int,
                "index": int,
                "ele_key": str,
                "axis": str,
            },
            ensure_count=False,
        )[0]
        data["orbits"] = [fix_value(val, float) for val in line.split(";")[3:]]
        res.append(data)

    return res


def parse_help(lines, cmd=""):
    """
    Parse help information.

    Returns
    -------
    str
    """
    return "\n".join(lines)


def parse_inum(lines, cmd=""):
    """
    Parse list of possible values for INUM.

    Returns
    -------
    list of int
    """
    return [int(num) for num in lines]


def parse_lat_calc_done(lines, cmd=""):
    """
    Parse lat_calc_done results.

    Returns
    -------
    bool
    """
    return parse_bool(lines[0])


def parse_lat_branch_list(lines, cmd=""):
    """
    Parse lat_branch_list results.

    Returns
    -------
    list of dict
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "index": int,
            "branch_name": str,
            "n_ele_track": int,
            "n_ele_max": int,
        },
    )


def parse_lat_param_units(lines, cmd=""):
    """
    Parse lat_param_units results.

    Returns
    -------
    str
    """
    return lines[0]


def parse_plot_lat_layout(lines, cmd=""):
    """
    Parse plot_lat_layout results.

    Returns
    -------
    list of dict
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "ix_branch": int,
            "ix_ele": int,
            "ele_s_start": float,
            "ele_s_end": float,
            "line_width": float,
            "shape": str,
            "y1": float,
            "y2": float,
            "color": str,
            "label_name": str,
        },
    )


def parse_plot_graph(lines, cmd=""):
    """
    Parse plot_graph results.

    Returns
    -------
    dict
    """
    # This should work, but there are issues with truncation causing failures.
    # See: https://github.com/bmad-sim/bmad-ecosystem/issues/1018
    # If that issue isn't resolved, we may want to pre-process the data
    # to at least get something back.
    try:
        return parse_tao_python_data(lines)
    except ValueError:
        logger.warning(
            "python plot_graph output failed to parse.  See linked issue "
            "and consider upgrading if possible. "
            "https://github.com/bmad-sim/bmad-ecosystem/issues/1018"
        )
        return lines


def parse_plot_line(lines, cmd=""):
    """
    Parse plot_line results.

    Returns
    -------
    list of dict or np.ndarray
    """
    if isinstance(lines, np.ndarray):
        return lines

    return _parse_by_keys_to_types(
        lines,
        {
            "index": int,
            "x": float,
            "y": float,
        },
    )


def parse_plot_symbol(lines, cmd=""):
    """
    Parse plot_symbol results.

    Returns
    -------
    list of dict or np.ndarray
    """
    if isinstance(lines, np.ndarray):
        return lines
    return _parse_by_keys_to_types(
        lines,
        {
            "index": int,
            "ix_symb": int,
            "x_symb": float,
            "y_symb": float,
        },
    )


def parse_shape_list(lines, cmd=""):
    """
    Parse shape_list results.

    Keys match those on `shape_set` for convenience.

    Returns
    -------
    list of dict
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "shape_index": int,
            "ele_name": str,
            "shape": str,
            "color": str,
            "shape_size": float,
            "type_label": str,
            "shape_draw": bool,
            "multi_shape": bool,
            "line_width": int,
        },
    )


def parse_shape_pattern_list(lines, cmd=""):
    """
    Parse shape_pattern_list results.

    Returns
    -------
    list of dict
    """
    args = _get_cmd_args(cmd)
    if not args:
        return _parse_by_keys_to_types(
            lines,
            {
                "name": str,
                "line_width": float,
            },
        )
    return _parse_by_keys_to_types(
        lines,
        {
            "s": float,
            "y": float,
        },
    )


def parse_show(lines, cmd=""):
    """
    Parse show results.

    Returns
    -------
    list of str
        This is raw list of strings from tao, as parsing is not currently
        supported.
    """
    return lines  # raise NotImplementedError()


def parse_species_to_int(lines, cmd=""):
    """
    Parse species_to_int results.

    Returns
    -------
    int
    """
    return int(lines[0])


def parse_species_to_str(lines, cmd=""):
    """
    Parse species_to_str results.

    Returns
    -------
    str
    """
    return lines[0]


def parse_spin_polarization(lines, cmd=""):
    """
    Parse spin_polarization results.

    Returns
    -------
    dict
    """
    lines = [
        line for line in lines if "[INFO]" not in line and "note: setting" not in line.lower()
    ]
    return parse_tao_python_data(lines)


def parse_spin_resonance(lines, cmd=""):
    """
    Parse spin_resonance results.

    Returns
    -------
    dict
    """
    # Filter lines as INFO/notes may appear in output
    return parse_tao_python_data(
        [
            line
            for line in lines
            if "[INFO]" not in line and "note: setting" not in line.lower()
        ]
    )


def parse_super_universe(lines, cmd=""):
    """
    Parse super_universe results.

    Returns
    -------
    dict
    """

    def fix_line(line):
        bug_prefix = "n_v1_var_used;INT;F"
        if not line.startswith(bug_prefix):
            return line
        if line.startswith(f"{bug_prefix};"):
            return line
        value = line[len(bug_prefix) :]
        return f"{bug_prefix};{value}"

    return parse_tao_python_data([fix_line(line) for line in lines])


def parse_var(lines, cmd=""):
    """
    Parse var results.

    Returns
    -------
    dict, or list of dict
        "slaves" mode will be a list of dicts.
        Normal mode will be a dict.
    """
    args = _get_cmd_args(cmd)
    if "slaves" in args:
        return _parse_by_keys_to_types(
            lines,
            {
                "index": int,
                "ix_branch": int,
                "ix_ele": int,
            },
        )

    return parse_tao_python_data(lines)


def parse_var_general(lines, cmd=""):
    """
    Parse var_general results.

    Returns
    -------
    list of dict
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "name": str,
            "line": str,
            "lbound": int,
            "ubound": int,
        },
    )


def parse_var_v1_array(lines, cmd=""):
    """
    Parse var_v1_array results.

    Returns
    -------
    dict
    """
    ix_v1_var = lines[-1]

    res = parse_tao_python_data([ix_v1_var])
    res["data"] = _parse_by_keys_to_types(
        lines[:-1],
        {
            "name": str,
            "ele_name": str,
            "attrib_name": str,
            "meas_value": float,
            "model_value": float,
            "design_value": float,
            "good_user": bool,
            "useit_opt": bool,
        },
    )
    return res


def parse_lat_list(lines, cmd=""):
    """
    Parse lat_list results.

    Returns
    -------
    list of str
    """
    return lines


def parse_place_buffer(lines, cmd=""):
    """
    Parse place_buffer results.

    Returns
    -------
    list of dict
    """
    return _parse_by_keys_to_types(
        lines,
        {
            "region": str,
            "graph": str,
        },
    )


def parse_show_plot_page(lines, cmd=""):
    """
    Parse 'show plot_page' output.

    Returns
    -------
    list of dict
    """

    def literal_eval(value: str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    result = {}
    for line in lines:
        line = line.strip()
        if not line or "=" not in line:
            continue

        variable, value = line.split("=", 1)
        variable = variable.strip().lstrip("%")
        value = value.rsplit("!")[0].strip()
        if value.startswith('"') or not value:
            value = value.strip('"')
        elif value in {"TF"}:
            value = {"T": True, "F": False}[value]
        else:
            value = [literal_eval(part) for part in value.split()]

            if "," in variable and "%" in variable:
                prefix = variable[: variable.index("%")]
                suffixes = [
                    suffix.strip() for suffix in variable[variable.index("%") :].split(",")
                ]
                result.update(
                    {f"{prefix}%{suffix}": val for suffix, val in zip(suffixes, value)}
                )
                continue
            if len(value) == 1:
                (value,) = value

        result[variable] = value
    return {key.replace("%", "_"): value for key, value in result.items()}


def parse_show_version(lines, cmd=""):
    """
    Parse 'show version' output.

    Returns
    -------
    datetime.datetime or None
    """
    try:
        return datetime.datetime.strptime("".join(lines).strip(), "Date: %Y/%m/%d %H:%M:%S")
    except ValueError:
        logger.warning("Failed to parse version output: %s", lines)
        return None
