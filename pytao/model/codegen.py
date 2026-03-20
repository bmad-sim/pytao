"""
Pydantic models based on the output of Tao pipe commands.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import pathlib
import re
import shutil
import sys
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from types import ModuleType
from typing import Any, Literal, NamedTuple, get_args

import numpy as np
import pydantic

import pytao
import pytao.util
import pytao.util.parsers as custom_parsers
from pytao.util.parsers import parse_pytype, parse_tao_python_data
from pytao import SubprocessTao, filter_tao_messages_context

logger = logging.getLogger(__name__)
DefaultType = (
    bool | int | str | float | complex | Sequence[float] | Sequence[int] | Sequence[str] | None
)

AnyPath = pathlib.Path | str
MODULE_PATH = pathlib.Path(__file__).resolve().parent
header_filename = MODULE_PATH / "header.tpl.py"

ParameterType = Literal[
    "STR",
    "ENUM",
    "FILE",
    "CRYSTAL",
    "COMPONENT",
    "DAT_TYPE",
    "DAT_TYPE_Z",
    "SPECIES",
    "ELE_PARAM",
    "STR_ARR",
    "ENUM_ARR",
    "INT_ARR",
    "REAL_ARR",
    "ENUM_Z",
    "STRUCT",
    "COMPLEX",
    "INT",
    "INUM",
    "REAL",
    "LOGIC",
]
valid_parameter_types = frozenset(get_args(ParameterType))


class PythonType(NamedTuple):
    type: str
    default: DefaultType | None
    default_factory: str | None


param_type_to_python_type = {
    "ENUM": PythonType(type="str", default="", default_factory=None),
    "ENUM_ARR": PythonType(type="Sequence[str]", default=None, default_factory="list"),
    "FILE": PythonType(type="str", default="", default_factory=None),
    "INT": PythonType(type="int", default=0, default_factory=None),
    "INT_ARR": PythonType(type="IntSequence", default=None, default_factory="list"),
    "INUM": PythonType(type="int", default=0, default_factory=None),
    "LOGIC": PythonType(type="bool", default=False, default_factory=None),
    "REAL": PythonType(type="float", default=0.0, default_factory=None),
    "REAL_ARR": PythonType(type="FloatSequence", default=None, default_factory="list"),
    "SPECIES": PythonType(type="str", default="", default_factory=None),
    "STR": PythonType(type="str", default="", default_factory=None),
    "STR_ARR": PythonType(type="Sequence[str]", default=None, default_factory="list"),
    "ELE_PARAM": PythonType(type="str", default="", default_factory=None),
    "COMPLEX": PythonType(type="complex", default=0j, default_factory=None),
}
TaoParameterValueBasic = (
    str | int | float | complex | list[float] | list[int] | list[str] | list[complex]
)
TaoParameterValue = TaoParameterValueBasic | dict[str, TaoParameterValueBasic]


class TypeInformation(pydantic.BaseModel, extra="allow"):
    type: str  # Base type name (e.g., 'INTEGER', 'REAL', 'CHARACTER', 'TYPE')

    allocatable: bool = False  # Whether the variable is allocatable
    asynchronous: bool = False  # Whether the variable can be used in async operations
    bind: str | None = None  # Bind(C) specification
    contiguous: bool = False  # Whether array data is contiguous
    dimension: str | None = None  # Dimension specification
    external: bool = False  # Whether the entity is external
    intent: str | None = None  # Intent specification ('IN', 'OUT', 'INOUT')
    intrinsic: bool = False  # Whether the type is intrinsic
    optional: bool = False  # Whether the variable is optional in a procedure
    parameter: bool = False  # Whether the variable is a parameter (constant)
    pointer: bool = False  # Whether the variable is a pointer
    private: bool = False  # Whether the variable has PUBLIC access
    protected: bool = False  # Whether the variable is protected
    public: bool = False  # Whether the variable has PUBLIC access
    save: bool = False  # Whether the variable has SAVE attribute
    kind: str | None = None  # Size or kind specification
    static: bool = False  # Whether the variable has STATIC attribute
    target: bool = False  # Whether the variable can be target of a pointer
    value: bool = False  # Whether the parameter is passed by value
    volatile: bool = False  # Whether the variable has VOLATILE attribute

    attributes: tuple[str, ...] = ()  # Any other unrecognized attributes


class StructureMember(pydantic.BaseModel, extra="allow"):
    line: int = 0
    definition: str = ""
    type_info: TypeInformation = TypeInformation(type="")
    name: str = ""
    comment: str = ""
    default: bool | int | str | float | None = ""


class ParsedStructure(pydantic.BaseModel, extra="allow"):
    """Parsed structure information, imported from cppbmad's codegen utilities."""

    filename: pathlib.Path = pathlib.Path()
    line: int = 0
    name: str = ""
    module: str = ""
    private: bool = False
    lines: list[str] = []
    comment: str = ""
    members: dict[str, StructureMember] = {}


def get_param_type(param: PipeOutputStructure | PipeOutputParameter) -> str:
    seq = _should_use_sequence(param)
    if isinstance(param, PipeOutputStructure):
        if seq:
            return f"Sequence[{param.class_name}]"
        return param.class_name

    # if seq:
    #     return f"{param.python_type.capitalize()}Sequence"
    return param.python_type


def get_docstring_type(param: PipeOutputStructure | PipeOutputParameter):
    seq = _should_use_sequence(param)
    type_name = "unknown"
    if isinstance(param, PipeOutputParameter):
        type_name = param.python_type
    if isinstance(param, PipeOutputStructure):
        type_name = param.class_name
    if seq:
        return {
            "FloatSequence": "sequence of floats",
            "IntSequence": "sequence of integers",
        }.get(type_name, type_name)

    return type_name


def maybe_raw_string(value: str) -> str:
    if "\\" in value:
        return "r"
    return ""


def _get_default(
    python_type: str, size: str | None, fortran_default
) -> tuple[DefaultType, str]:
    if fortran_default is not None:
        fortran_default = str(fortran_default)
        if fortran_default.lower() == ".false.":
            return False, ""
        if fortran_default.lower() == ".true.":
            return True, ""
        if fortran_default.lower() == "real_garbage$":
            return 0.0, ""
        if fortran_default.lower() == "int_garbage$":
            return 0, ""
        if fortran_default.lower() == "null()":
            # TODO only used for 'descrip' ptr at the moment
            return "", ""
        if fortran_default.endswith("$"):  # some constant?
            ...
        else:
            if python_type in {"float", "FloatSequence"}:
                # 10d3 -> 10e3
                fortran_default = fortran_default.lower().replace("d", "e")
                fortran_default = fortran_default.removesuffix("_rp")
                if "." not in fortran_default and "e" not in fortran_default:
                    fortran_default = f"{fortran_default}.0"
            try:
                default = ast.literal_eval(fortran_default)
                if isinstance(default, list):
                    return tuple(default), ""
                return default, ""
            except (SyntaxError, ValueError):
                pass

    try:
        int(size or "abc")
    except ValueError:
        size = None
    if not size:
        default = {
            "str": "",
            "int": 0,
            "float": 0.0,
            "type": None,
            "bool": False,
            "Complex": 0.0,
        }.get(python_type, None)
        return default, ""
    return "", "list"


def match_default_from_struct(member: PipeOutputParameter, reference: StructureMember):
    default, default_factory = _get_default(
        python_type=member.python_type,
        size=reference.type_info.kind,
        fortran_default=reference.default,
    )
    if member.dimension or reference.type_info.dimension:
        if reference.type_info.dimension == ":":
            assert "Sequence" in member.python_type
        else:
            assert int(member.dimension) == int(reference.type_info.dimension)
            if isinstance(default, (list, tuple)):
                default = list(default)
            else:
                default = [default] * int(reference.type_info.dimension)

    return default, default_factory


class ElementKeyExample(NamedTuple):
    lattice_fn: str
    key: str
    element_names: list[str]


def generate_field_value(param: PipeOutputParameter | PipeOutputStructure) -> str:
    """Generates Pydantic Field definition code."""
    args = []

    # Handle default/default_factory
    if param.default_factory:
        args.append(f"default_factory={param.default_factory}")
    else:
        assert not isinstance(param, PipeOutputStructure)
        default_value = custom_repr(param.default)
        args.append(f"default={default_value}")

    # Handle max_length constraint
    if not isinstance(param, PipeOutputStructure):
        try:
            dim = int(param.dimension)
            if dim != 0:
                args.append(f"max_length={dim}")
        except (ValueError, TypeError, AttributeError):
            pass

    if param.comment:
        clean_comment = param.comment.replace('"', "'")

        if len(param.comment) > 80:
            lines = textwrap.wrap(clean_comment, width=74)
            desc_lines = []
            for i, line in enumerate(lines):
                prefix = maybe_raw_string(line)
                content = f"{line} " if i < len(lines) - 1 else line
                desc_lines.append(f'{prefix}"{content}"')

            joined_desc = "\n        ".join(desc_lines)
            args.append(f"description=({joined_desc})")
        else:
            prefix = maybe_raw_string(param.comment)
            args.append(f'description={prefix}"{clean_comment}"')

    name = getattr(param, "name", "")
    if name != getattr(param, "python_name", ""):
        args.append(f'alias="{name}"')

    if isinstance(param, PipeOutputParameter) and param.param:
        inner_param = param.param
        can_vary = inner_param.can_vary
        if inner_param is None or not can_vary:
            args.append("frozen=True")

    if len(args) == 1 and args[0].startswith("default"):
        if param.default_factory:
            return f" = {param.default_factory}()"
        assert not isinstance(param, PipeOutputStructure)
        return f" = {custom_repr(param.default)}"
    return f" = Field({', '.join(args)})"


def generate_docstring_content(py_struct: PipeOutputStructure) -> str:
    """Generates the docstring text block."""
    lines = []
    if py_struct.comment:
        lines.append(py_struct.comment)

    lines.append("")
    lines.append("Attributes")
    lines.append("----------")

    sorted_members = sorted(py_struct.members.items())

    for name, param in sorted_members:
        type_str = get_docstring_type(param)
        if getattr(param, "optional", False):
            type_str += " or None"

        lines.append(f"{name} : {type_str}")

        if param.comment:
            wrapped = textwrap.fill(param.comment, width=70)
            indented = textwrap.indent(wrapped, "    ")
            lines.append(indented)

    return "\n    ".join(lines)


def _should_use_sequence(obj):
    if isinstance(obj, PipeOutputStructure):
        return obj.is_list
    if isinstance(obj, PipeOutputParameter):
        return obj.dimension is not None and obj.dimension > 1
    raise NotImplementedError(type(obj))


def generate_class_code(py_struct: PipeOutputStructure) -> str:
    """Recursively generates class definitions."""
    output = []

    sorted_members = sorted(py_struct.members.items())

    for _, param in sorted_members:
        if isinstance(param, PipeOutputStructure):
            output.append(generate_class_code(param))

    class_name = py_struct.class_name
    base_class = py_struct.base_class

    class_def = [f"class {class_name}({base_class}):"]

    docstring_content = generate_docstring_content(py_struct)
    raw_prefix = maybe_raw_string(docstring_content)

    class_def.append(f'    {raw_prefix}"""')
    class_def.append(f"    {docstring_content}")
    class_def.append('    """')

    if base_class in ["TaoModel", "TaoSettableModel"]:
        class_def.append(f'    _tao_command_attr_: ClassVar[str] = "{py_struct.tao_command}"')

        if py_struct.tao_set_name and py_struct.tao_set_name != py_struct.tao_command:
            class_def.append(f'    _tao_set_name_: ClassVar[str] = "{py_struct.tao_set_name}"')

        class_def.append(
            f"    _tao_command_default_args_: ClassVar[dict[str, Any]] = {py_struct.tao_command_default_args}"
        )

    if base_class in ["TaoSettableModel", "TaoAttributesModel"]:
        class_def.append(
            f"    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = {py_struct.skip_if_0}"
        )

    sorted_discriminators = sorted(py_struct.discriminators.items())
    for name, value in sorted_discriminators:
        class_def.append(f'    {name}: Literal["{value}"] # = "{value}"')

    for name, param in sorted_members:
        type_name = get_param_type(param)
        if getattr(param, "optional", False):
            type_name += " | None"

        field_def = generate_field_value(param).strip()
        class_def.append(f"    {name}: {type_name}{field_def}")

    output.append("\n".join(class_def))
    return "\n\n".join(output)


def generate_tao_structs_file(
    header: str, all_structs: dict[str, Any], aggregate_types: dict[str, list[Any]]
) -> str:
    """
    Main entry point equivalent to rendering the whole template.
    """
    output_sections = []

    output_sections.append(header)
    sorted_structs = sorted(all_structs.items())

    for _, py_struct in sorted_structs:
        output_sections.append(generate_class_code(py_struct))

    for type_name, structs in aggregate_types.items():
        class_names = [s.class_name for s in structs]
        union_str = " | ".join(class_names)
        output_sections.append(f"{type_name} = {union_str}")

    for type_name, structs in aggregate_types.items():
        lower_name = type_name.lower()

        lines = [f"key_to_{lower_name}: dict[str, type[pydantic.BaseModel]] = {{"]
        for struct in structs:
            key = struct.discriminators.get("key", "UNKNOWN")
            lines.append(f'    "{key}": {struct.class_name},')
        lines.append("}")
        output_sections.append("\n".join(lines))

        # name_to_key
        lines = [f"{lower_name}_to_key: dict[type[pydantic.BaseModel], str] = {{"]
        for struct in structs:
            key = struct.discriminators.get("key", "UNKNOWN")
            lines.append(f'    {struct.class_name}: "{key}",')
        lines.append("}")
        output_sections.append("\n".join(lines))

    return "\n\n".join(output_sections)


# element_key_examples = [
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Beginning_Ele",
#         ["BEGINNING"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Drift",
#         ["DRIFT1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Lcavity",
#         ["LCAVITY1", "LCAVITY2", "LCAVITY3"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Marker",
#         ["END"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "ECollimator",
#         ["ECOLLIMATOR1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "RCollimator",
#         ["RCOLLIMATOR1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Crab_Cavity",
#         ["CRAB_CAVITY1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Patch",
#         ["PATCH1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Quadrupole",
#         ["QUADRUPOLE1", "QUADRUPOLE2", "QUADRUPOLE3", "QUADRUPOLE4", "QUADRUPOLE5"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "RFCavity",
#         ["RFCAVITY1", "RFCAVITY2", "RFCAVITY3"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Taylor",
#         ["TAYLOR1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "SBend",
#         ["SBEND1", "RBEND2", "SBEND3", "RBEND4", "SBEND5", "RBEND6", "SBEND7"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "EM_Field",
#         ["EM_FIELD1", "EM_FIELD2"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Solenoid",
#         ["SOLENOID1", "SOLENOID2"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Octupole",
#         ["OCTUPOLE1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Match",
#         ["MATCH1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Sextupole",
#         ["SEXTUPOLE1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Multipole",
#         ["MULTIPOLE1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "ELSeparator",
#         ["ELSEPARATOR1", "ELSEPARATOR2"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "AB_multipole",
#         ["AB_MULTIPOLE1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "HKicker",
#         ["HKICKER1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Instrument",
#         ["INSTRUMENT1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Kicker",
#         ["KICKER1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Monitor",
#         ["MONITOR1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Sad_Mult",
#         ["SAD_MULT1", "SAD_MULT2"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Sol_Quad",
#         ["SOL_QUAD1", "SOL_QUAD2"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "VKicker",
#         ["VKICKER1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Wiggler",
#         ["WIGGLER_MAP1", "WIGGLER_FLAT1", "WIGGLER_HELI1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "AC_Kicker",
#         ["AC_KICKER1", "AC_KICKER2", "AC_KICKER3"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "BeamBeam",
#         ["BEAMBEAM1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Fiducial",
#         ["FIDUCIAL1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Floor_Shift",
#         ["FLOOR_SHIFT1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "GKicker",
#         ["GKICKER1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/tracking_method_test/tracking_method_test.bmad",
#         "Thick_Multipole",
#         ["THICK_MULTIPOLE1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/slice_test/elements.bmad", "E_Gun", ["E_GUN1"]
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/bookkeeper_test/bookkeeper_test.bmad",
#         "Overlay",
#         ["OV1", "OV2", "OV3"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/bookkeeper_test/bookkeeper_test.bmad",
#         "Group",
#         ["GR1", "GR2", "GR3", "GR4", "GRN"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/bookkeeper_test/ramper.bmad",
#         "Ramper",
#         ["RAMP_O_B", "RAMP_PC", "RAMP_VOLT1", "RAMP_PHASE"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/pipe_test/floor_orbit.bmad",
#         "Pipe",
#         [
#             "P1",
#             "P2#1",
#             "P2#2",
#             "P2#3",
#             "P3#1",
#             "P3#2",
#             "P3#3",
#             "P4#1",
#             "P4#2",
#             "P4#3",
#             "P5",
#             "P2",
#             "P3",
#             "P4",
#         ],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/girder_test/girder_test.bmad",
#         "Girder",
#         ["G1", "G2", "GG"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/multipass_test/branch_fork.bmad",
#         "Photon_Fork",
#         ["PFORK"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/photon_test/photon_test.bmad",
#         "Crystal",
#         ["CST1", "CST2", "CST3", "CST4"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/photon_test/photon_test.bmad",
#         "Photon_Init",
#         ["PINIT"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/synrad3d_test/branch.bmad", "Fork", ["PF"]
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/xraylib_test/xraylib.bmad",
#         "Multilayer_Mirror",
#         ["MULTILAYER_MIRROR1"],
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/photon_test/grid.bmad", "Detector", ["DET1"]
#     ),
#     ElementKeyExample(
#         "$ACC_ROOT_DIR/regression_tests/photon_test/mask.bmad", "Mask", ["MASK1"]
#     ),
# ]


class InvalidParameterTypeError(Exception):
    pass


def python_value_to_param_type(value) -> ParameterType:
    if isinstance(value, float):
        return "REAL"
    if isinstance(value, (int, np.integer)):
        return "INT"
    if isinstance(value, str):
        return "STR"
    if isinstance(value, complex):
        return "COMPLEX"
    if isinstance(value, (list, np.ndarray)):
        if len(value):
            element_type = python_value_to_param_type(value[0])
            return f"{element_type}_ARR"
        return "REAL_ARR"  # a guess
    if isinstance(value, dict):
        return "STRUCT"
    raise NotImplementedError(type(value))


class TaoParameter(pydantic.BaseModel):
    """
    Basic class for holding the properties of a parameter in Tao.

    Attributes
    ----------
    name : str
        The name of the parameter
    type : ParameterType
        "STR", "INT", "REAL", "LOGIC", "ENUM", etc...
    prefix : str or None
        Structure prefix name
    can_vary : bool
        Indicating whether or not the user may change the value of the
        parameter.
    is_ignored : bool
        Indicates that the parameter is to be ignored by the gui
    value :
        The value held in the parameter, should be of the appropriate type for
        the specified param_type
    """

    name: str
    type: ParameterType
    prefix: str | None
    can_vary: bool
    is_ignored: bool
    value: TaoParameterValue | None

    @classmethod
    def from_tao(
        cls,
        param_name: str,
        param_type: ParameterType,
        can_vary: str,
        param_value: TaoParameterValue,
    ):
        # Enums and inums may have a prefix attached to their name, as in
        # axis^type.  In this case, the prefix is removed from the parameter name
        # and stored in the self.prefix variable
        if (param_type in ["ENUM", "INUM"]) and (param_name.count("^") == 1):
            prefix, name = param_name.split("^")
        else:
            prefix = None
            name = param_name

        if isinstance(param_value, np.ndarray):
            param_value = param_value.tolist()

        return cls(
            name=name,
            type=param_type,
            prefix=prefix,
            can_vary=can_vary == "T",
            is_ignored=can_vary == "I",
            value=param_value,
        )

    def __str__(self):
        can_vary = "T" if self.can_vary else "F"
        return f"{self.name};{self.type};{can_vary};{self.value}"

    @classmethod
    def from_str(cls, line: str) -> TaoParameter:  # Lifted/adapted from pytao
        """
        Takes a parameter string (EG: 'param_name;STR;T;abcd')
        and returns a TaoParameter instance
        line MUST have at least 3 semicolons
        """
        parts = line.split(";")
        if len(parts) < 3:
            raise ValueError(f"{line} is not a valid param_string (not enough semicolons)")

        param_name, type_, can_vary = parts[:3]
        component_value = parts[3:]

        if type_ not in valid_parameter_types:
            raise InvalidParameterTypeError(type_)
        data = parse_pytype(type_, component_value)
        return cls.from_tao(param_name, type_, can_vary, data)


def remove_struct_prefix(param: PipeOutputParameter, key: str) -> str:
    if "%" in param.name:
        struct_prefix = param.name.split("%")[0]
        return key.removeprefix(f"{struct_prefix}_")
    return key


def tao_name_to_python_name(name: str) -> str:
    python_name = name
    # NOTE: clearly can be done by regex, but I want the chance to
    # intentionally tweak what these get mapped to
    for ch in (
        "1^",
        ",:)",
        "#",
        "^",
        " ",
        "(",
        ")",
        "/",
        "*",
        "%",
    ):
        python_name = python_name.replace(ch, "_")
    if python_name.startswith("-"):
        python_name = python_name.lstrip("-") + "_neg"

    try:
        int(name)
    except ValueError:
        pass
    else:
        return f"data_{name}"

    while "__" in python_name:
        python_name = python_name.replace("__", "_")
    python_name = python_name.strip("_").lower()
    return {"l": "L"}.get(python_name, python_name)


class PipeOutputParameter(pydantic.BaseModel):
    param: TaoParameter | None
    name: str = ""
    python_name: str = ""
    type: str = ""
    python_type: str = ""
    dimension: int | None = None
    comment: str = ""
    default: DefaultType | None = None
    default_factory: str | None = None
    optional: bool = False

    @classmethod
    def from_value(
        cls,
        name: str,
        value,
        *,
        param: TaoParameter | None = None,
        **kwargs,
    ) -> PipeOutputParameter:
        return cls(
            param=param,
            name=name,
            python_name=tao_name_to_python_name(name),
            type=python_value_to_param_type(value),
            **kwargs,
        )


class PipeOutputStructure(pydantic.BaseModel):
    cmd: TaoCommandAndResult | None
    tao_command: str
    tao_set_name: str | None = None
    class_name: str
    comment: str
    skip_if_0: tuple[str, ...] = pydantic.Field(default_factory=tuple)
    tao_command_default_args: dict = pydantic.Field(default_factory=dict)
    members: dict[str, PipeOutputParameter | PipeOutputStructure] = pydantic.Field(
        default_factory=dict
    )
    path: Sequence[str | int] | None = None
    full_path: Sequence[str | int] | None = None
    default_factory: str = "unknown"
    optional: bool = False
    discriminators: dict[str, str] = pydantic.Field(default_factory=dict)
    base_class: str = "TaoModel"
    # frozen: bool = False

    @property
    def is_list(self) -> bool:
        # TODO somewhat arbitrary
        return self.path and self.path[0] == 0

    @property
    def param(self) -> None:
        # jinja template compat - is this represented as a list of structures?
        return None

    @classmethod
    def from_cmd(
        cls,
        cmd: TaoCommandAndResult,
        *,
        class_name: str,
        full_path: Sequence[int | str] | None = None,
        path: Sequence[int | str] | None = None,
        reference_structures: tuple[ParsedStructure, ...] = (),
        comment: str = "",
        mark_optional: Iterable[str] = (),
        mark_empty_lists: Iterable[str] = (),
        skip_prefixes: Iterable[str] = (),
        tao_command_attr_name: str | None = None,
        tao_set_name: str | None = None,
        **kwargs,
    ):
        optional = set(mark_optional)
        empty_lists = set(mark_empty_lists)
        skip_prefixes = set(skip_prefixes)
        if full_path is None:
            full_path = ()
            path = ()

        result = cmd.result
        for idx in full_path or ():
            result = result[idx]

        if isinstance(result, list) and not isinstance(result[0], (str, int, float)):
            full_path = tuple(full_path or []) + (0,)
            path = tuple(path or []) + (0,)
            result = result[0]

        for key, value in list(result.items()):
            if isinstance(value, list):
                if len(value) == 0:
                    # print(f"{cmd.cmd}: Removing key: {key} (zero length list)")
                    if key not in empty_lists:
                        raise RuntimeError(f"Empty list for {key!r} in command {cmd.cmd}")
                    result.pop(key)
            if any(key.startswith(prefix) for prefix in skip_prefixes):
                result.pop(key)

        structured_value_keys: dict[str, Any] = {
            key: value
            for key, value in result.items()
            if (isinstance(value, list) and not isinstance(value[0], (str, int, float)))
            or isinstance(value, dict)
        }
        members: Mapping[str, PipeOutputParameter | PipeOutputStructure] = (
            tao_special_parser_parameter_dict(
                cmd,
                path=full_path,
                skip_keys=tuple(structured_value_keys),
            )
        )

        for key, member in members.items():
            if isinstance(member, PipeOutputParameter):
                lookup_key = remove_struct_prefix(member, key)
            else:
                lookup_key = key
            for ref_cls in reference_structures:
                ref_struct_member = {
                    name.lower(): member for name, member in ref_cls.members.items()
                }.get(lookup_key.lower())
                if ref_struct_member is not None:
                    assert isinstance(member, PipeOutputParameter)
                    member.comment = ref_struct_member.comment
                    member.default, member.default_factory = match_default_from_struct(
                        member, ref_struct_member
                    )
                    break

            if any(re.match(pat, key) for pat in optional):
                member.optional = True
                member.default = None

        for key in tuple(structured_value_keys):
            value = result[key]

            if isinstance(value, list):
                sub_path = (0,)
            else:
                sub_path = ()

            clsname = f"{class_name}_{key.capitalize()}"
            members[key] = PipeOutputStructure.from_cmd(
                cmd,
                class_name=clsname,
                full_path=full_path + (key,) + sub_path,
                path=sub_path,
                default_factory="list" if isinstance(value, list) else clsname,
            )

        default_comment = f"Structure which corresponds to Tao `pipe {cmd.cmd}`, for example."

        missing_optional_keys = {
            pat for pat in optional if not any(re.match(pat, key) for key in members)
        }
        if missing_optional_keys:
            assert (
                not missing_optional_keys
            ), f"Missing optional key(s) in source example: {missing_optional_keys}"

        if tao_command_attr_name:
            tao_command = tao_command_attr_name
        else:
            tao_command = cmd.cmd.split()[0].replace(":", "_") if cmd else ""

        tao_set_name = tao_set_name or tao_command
        return cls(
            cmd=cmd,
            class_name=class_name,
            comment=comment or default_comment,
            members=members,
            full_path=full_path,
            path=path,
            tao_command=tao_command,
            tao_set_name=tao_set_name,
            **kwargs,
        )


def custom_repr(obj: object) -> str:
    """A tweaked ``repr`` to always return double quotes."""
    if isinstance(obj, str):
        s = obj.removeprefix('"').removesuffix('"').replace("'", r"\'")
        return f"'{s}'"
    result = repr(obj)
    return result.replace("'", '"')


def render_python_source(
    dict_structs: dict[str, PipeOutputStructure],
    *,
    header_filename: AnyPath = header_filename,
) -> str:
    """
    Load the structure yaml file and generate dataclass source code for it.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the structure yaml file.

    Returns
    -------
    str
        Generated Python source code.
    """

    with open(header_filename) as fp:
        header = fp.read()

    aggregate_types = {
        "GeneralAttributes": [
            struct for struct in dict_structs.values() if "key" in struct.discriminators
        ],
    }

    if not aggregate_types["GeneralAttributes"]:
        aggregate_types.pop("GeneralAttributes")

    return generate_tao_structs_file(
        all_structs=dict_structs,
        header=header,
        aggregate_types=aggregate_types,
    ).strip()


def tao_parameter_dict(tao: pytao.Tao, cmd: str) -> dict[str, TaoParameter]:
    """
    Takes a list of strings, each string looks something like: 'param_name;STR;T;abcd'
    and returns a dictionary with keys being the param_name.
    Blank strings will be ignored.
    """
    res = {}

    lines = tao.cmd(cmd)

    for line in lines:
        if not line.strip():
            continue
        param = TaoParameter.from_str(line)
        res[param.name] = param

    return res


def create_module(
    path: pathlib.Path,
    source_code: str,
    module_name_prefix: str | None = None,
) -> ModuleType:
    with open(path, "w") as fp:
        print(source_code, file=fp)

    module_name = path.stem
    if module_name_prefix:
        module_name = module_name_prefix + module_name

    spec = importlib.util.spec_from_file_location(module_name, str(path), loader=None)
    module = importlib.util.module_from_spec(spec)  # pyright: ignore[reportArgumentType]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # pyright: ignore[reportOptionalMemberAccess]
    return module


class TaoCommandAndResult(pydantic.BaseModel):
    cmd: str
    lines: list[str]
    result: Any

    normal_params: dict[str, TaoParameter]
    index: int | None = None

    @classmethod
    def from_tao(
        cls,
        tao: pytao.Tao,
        cmd: str,
        index: int | None = None,
    ) -> TaoCommandAndResult:
        lines = tao.cmd(f"pipe {cmd}")

        tao_cmd = cmd.split(" ")[0].replace(":", "_")
        try:
            parser = getattr(custom_parsers, f"parse_{tao_cmd}")
        except AttributeError:
            res = parse_tao_python_data(lines, clean_key=True)
        else:
            res = parser(lines, cmd=cmd)

        normal_params = {}
        for line in lines:
            if not line.strip():
                continue

            try:
                param = TaoParameter.from_str(line)
            except Exception as ex:
                logger.warning(
                    "Failed to parse parameter in cmd=%r line=%r: %s", cmd, line, ex
                )
            else:
                normal_params[param.name] = param

        return cls(cmd=cmd, lines=lines, result=res, normal_params=normal_params)


def tao_special_parser_parameter_dict(
    cmd: TaoCommandAndResult,
    path: Sequence[int | str] | None = None,
    skip_keys: tuple[str, ...] = (),
) -> dict[str, PipeOutputParameter]:
    res = {}

    cmd_result = cmd.result
    for idx in path or ():
        cmd_result = cmd_result[idx]

    for key, value in cmd_result.items():
        if key in skip_keys:
            continue
        param = cmd.normal_params.get(key, None)
        if param is not None:
            python_type = param_type_to_python_type[param.type]
        else:
            param_type = python_value_to_param_type(value)
            python_type = param_type_to_python_type[param_type]

        param = PipeOutputParameter.from_value(
            name=key,
            value=value,
            param=param,
            python_type=python_type.type,
            default=python_type.default,
            default_factory=python_type.default_factory,
            dimension=len(value) if isinstance(value, (list, np.ndarray)) else 0,
        )
        res[param.python_name] = param
    return res


def add_missing_members(dest: PipeOutputStructure, src: PipeOutputStructure):
    new_keys: set[str] = set()
    for key, member in src.members.items():
        if key not in dest.members:
            dest.members[key] = member
            new_keys.add(key)
    return new_keys


def get_element_index(tao: pytao.Tao, name: str) -> int:
    return tao.lat_list(name, flags="", who="ele.ix_ele")[0]


def generate_structures():
    res: dict[str, PipeOutputStructure] = {}
    with pytao.SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching",
        noplot=True,
    ) as tao:
        res["BeamInit"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "beam_init"),
            class_name="BeamInit",
            reference_structures=(structs_by_name["beam_init_struct"],),
            skip_if_0=("a_emit", "b_emit", "a_norm_emit", "b_norm_emit"),
            mark_optional=("grid_.*", "ellipse_.*"),
            tao_command_default_args={
                "ix_branch": 0,
                "ix_uni": 1,
            },
            base_class="TaoSettableModel",
        )
        res["Beam"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "beam"),
            class_name="Beam",
            reference_structures=(
                structs_by_name["bunch_track_struct"],
                structs_by_name["tao_beam_uni_struct"],
                structs_by_name["tao_beam_branch_struct"],
                structs_by_name["tao_lattice_branch_struct"],
            ),
            tao_command_default_args={
                "ix_branch": 0,
                "ix_uni": 1,
            },
            base_class="TaoSettableModel",
        )

        res["BmadCom"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "bmad_com"),
            class_name="BmadCom",
            reference_structures=(structs_by_name["bmad_common_struct"],),
            tao_command_default_args={},
            base_class="TaoSettableModel",
        )
        # res["BmadCom"].members["d_orb"].default = [1e-5] * 6
        res["SpaceChargeCom"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "space_charge_com"),
            class_name="SpaceChargeCom",
            reference_structures=(structs_by_name["space_charge_common_struct"],),
            tao_command_default_args={},
            base_class="TaoSettableModel",
        )

        res["ElementHead"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:head 1"),
            class_name="ElementHead",
            reference_structures=(structs_by_name["ele_struct"],),
        )

        res["ElementOrbit"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:orbit 1"),
            class_name="ElementOrbit",
            reference_structures=(structs_by_name["coord_struct"],),
        )

        res["ElementTwiss"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:twiss 1"),
            class_name="ElementTwiss",
            reference_structures=(structs_by_name["twiss_struct"],),
        )

    with SubprocessTao(
        lattice_file="$ACC_ROOT_DIR/regression_tests/pipe_test/em_field.bmad",
        noplot=True,
    ) as tao:
        res["ElementGridField"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:grid_field G1 1 base"),
            class_name="ElementGridField",
            reference_structures=(structs_by_name["grid_field_struct"],),
        )
        res["ElementGridFieldPoints"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:grid_field G1 1 points"),
            class_name="ElementGridFieldPoints",
            reference_structures=(structs_by_name["grid_field_pt_struct"],),
        )
        res["ElementGridFieldPoints"].members["data"].dimension = None

    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d",
        noplot=True,
    ) as tao:
        res["ElementMat6"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:mat6 1 mat6"),
            class_name="ElementMat6",
            reference_structures=(),
        )

        res["ElementMat6Vec0"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:mat6 1 vec0"),
            class_name="ElementMat6Vec0",
            reference_structures=(),
        )

        res["ElementMat6Error"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:mat6 1 err"),
            class_name="ElementMat6Error",
            reference_structures=(),
        )

        res["ElementChamberWall"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:chamber_wall 1 1 x"),
            class_name="ElementChamberWall",
            reference_structures=(),
        )

        res["ElementWall3DTable"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:wall3d 1 1 table"),
            class_name="ElementWall3DTable",
            reference_structures=(structs_by_name["wall3d_section_struct"],),
        )

        res["ElementWall3DBase"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:wall3d 1 1 base"),
            class_name="ElementWall3DBase",
            reference_structures=(
                structs_by_name["wall3d_section_struct"],
                structs_by_name["wall3d_struct"],
                structs_by_name["wall3d_vertex_struct"],
            ),
            mark_optional={"superimpose"},
        )

        res["TaoGlobal"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "global"),
            class_name="TaoGlobal",
            reference_structures=(structs_by_name["tao_global_struct"],),
            base_class="TaoSettableModel",
            tao_command_attr_name="tao_global",
            tao_set_name="global",
            mark_optional={},
        )

    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_taylor",
        # lattice_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/custom_tao_with_measured_data/RRNOVAMU2E11172016.bmad",
        noplot=True,
    ) as tao:
        ab_multipole_index = get_element_index(tao, "AB_MULTIPOLE1")
        quad1_index = get_element_index(tao, "QUADRUPOLE1")
        multipole_index = get_element_index(tao, "MULTIPOLE1")

        # A multipole (§4.36) element specifies its magnetic multipole components
        # using normal and skew components with a tilt
        # -> This implicitly generates a corresponding ElementMultipoles_Data.
        multipoles = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, f"ele:multipoles {multipole_index}"),
            class_name="ElementMultipoles",
            reference_structures=(structs_by_name["ele_struct"],),
        )
        res["ElementMultipoles"] = multipoles
        add_missing_members(
            dest=multipoles,
            src=PipeOutputStructure.from_cmd(
                TaoCommandAndResult.from_tao(tao, f"ele:multipoles {quad1_index}"),
                class_name="",
                reference_structures=(structs_by_name["ele_struct"],),
                mark_optional={"scale_multipoles"},
                # mark_empty_lists=("data",),
            ),
        )

        # An ab_multipole (§4.1) specifies magnetic multipoles using normal (Bn)
        # and skew (An) components
        res["ElementMultipolesAB"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, f"ele:multipoles {ab_multipole_index}"),
            class_name="ElementMultipolesAB",
            reference_structures=(structs_by_name["ele_struct"],),
        )
        # Elements like quadrupoles and sextupoles can have assigned to them both
        # magnetic and electric multi- pole fields. In this case, the magnetic
        # fields are specified using the same convention as the ab_multipole. For
        # such non-multipole elements, the magnetic multipole strength is scaled
        res["ElementMultipolesScaled"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, f"ele:multipoles {quad1_index}"),
            class_name="ElementMultipolesScaled",
            reference_structures=(structs_by_name["ele_struct"],),
        )

    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init",
        noplot=True,
    ) as tao:
        res["ElementBunchParams"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "bunch_params 1"),
            class_name="ElementBunchParams",
            reference_structures=(structs_by_name["bunch_params_struct"],),
        )
        res["ElementLordSlave"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, "ele:lord_slave 1 1 x"),
            class_name="ElementLordSlave",
            reference_structures=(),
        )

    with SubprocessTao(
        lattice_file="$ACC_ROOT_DIR/regression_tests/photon_test/mask.bmad",
        noplot=True,
    ) as tao:
        add_missing_members(
            dest=res["ElementWall3DBase"],
            src=PipeOutputStructure.from_cmd(
                TaoCommandAndResult.from_tao(tao, "ele:wall3d 1 1 base"),
                class_name="",
                reference_structures=(
                    structs_by_name["wall3d_section_struct"],
                    structs_by_name["wall3d_struct"],
                    structs_by_name["wall3d_vertex_struct"],
                ),
                mark_optional={"clear_material", "opaque_material", "thickness"},
            ),
        )

        mask1_index = get_element_index(tao, "MASK1")
        res["ElementPhotonBase"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, f"ele:photon {mask1_index} base"),
            class_name="ElementPhotonBase",
            reference_structures=(structs_by_name["photon_element_struct"],),
        )
        res["ElementPhotonCurvature"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, f"ele:photon {mask1_index} curvature"),
            class_name="ElementPhotonCurvature",
            reference_structures=(structs_by_name["photon_element_struct"],),
        )

    with SubprocessTao(
        lattice_file="$ACC_ROOT_DIR/regression_tests/xraylib_test/xraylib.bmad",
        noplot=True,
    ) as tao:
        mirror1_index = get_element_index(tao, "multilayer_mirror1")
        res["ElementPhotonMaterial"] = PipeOutputStructure.from_cmd(
            TaoCommandAndResult.from_tao(tao, f"ele:photon {mirror1_index} material"),
            class_name="ElementPhotonMaterial",
            reference_structures=(structs_by_name["photon_element_struct"],),
            mark_optional=("f0_m1",),
        )

    with filter_tao_messages_context(functions=["twiss_propagate1", "radiation_integrals"]):
        with SubprocessTao(
            lattice_file="$ACC_ROOT_DIR/regression_tests/wake_test/wake_test.bmad",
            noplot=True,
        ) as tao:
            res["ElementWakeBase"] = PipeOutputStructure.from_cmd(
                TaoCommandAndResult.from_tao(tao, "ele:wake P3 base"),
                class_name="ElementWakeBase",
                reference_structures=(
                    structs_by_name["wake_struct"],
                    structs_by_name["wake_sr_struct"],
                    structs_by_name["wake_lr_struct"],
                ),
                mark_optional=(),
            )
            res["ElementWakeSrLong"] = PipeOutputStructure.from_cmd(
                TaoCommandAndResult.from_tao(tao, "ele:wake P3 sr_long"),
                class_name="ElementWakeSrLong",
                reference_structures=(structs_by_name["wake_sr_struct"],),
                mark_optional=(),
            )
            res["ElementWakeSrTrans"] = PipeOutputStructure.from_cmd(
                TaoCommandAndResult.from_tao(tao, "ele:wake P3 sr_long"),
                class_name="ElementWakeSrTrans",
                reference_structures=(structs_by_name["wake_sr_struct"],),
                mark_optional=(),
            )
    # TODO find element that works with these:
    #   sr_long_table, sr_trans_table, lr_mode_table
    # res["ElementWakeSrLongtable"] = PipeOutputStructure.from_cmd(
    #     TaoCommandAndResult.from_tao(tao, "ele:wake P3 sr_long"),
    #     class_name="ElementWakeSrLong",
    #     reference_classes=(structs_by_name["wake_sr_struct"],),
    #     mark_optional=(),
    # )

    assert all(
        key in res["ElementWall3DBase"].members
        for key in (
            "clear_material",
            "ele_anchor_pt",
            "name",
            "opaque_material",
            "superimpose",
            "thickness",
        )
    )
    return res


def write_source(
    fn: pathlib.Path,
    res: dict[str, PipeOutputStructure],
    header_filename: AnyPath = header_filename,
    module_name_prefix: str = "pytao.model.",
):
    python_src = render_python_source(res, header_filename=header_filename)
    python_src = python_src.replace("# noqa: F401", "")
    python_src = python_src.replace("# noqa: F821", "")
    mod = create_module(fn, source_code=python_src, module_name_prefix=module_name_prefix)

    if shutil.which("ruff"):
        os.system(f"ruff check --extend-select=I --fix {fn}")
        os.system(f"ruff format {fn}")

    classes = {
        name: getattr(mod, name) for name in dir(mod) if isinstance(getattr(mod, name), type)
    }
    return mod, classes


def deserialize(
    mod: ModuleType,
    struct: type[PipeOutputStructure],
    cls: type[pydantic.BaseModel],
    tao_python_result: list | dict,
):
    if isinstance(tao_python_result, list):
        for item in tao_python_result:
            deserialize(mod, struct, cls, item)
        return

    print(f"\n\n\n{cls.__name__}:")

    if not hasattr(mod, "GeneralAttributes"):
        obj = cls.model_validate(tao_python_result)
        print(tao_python_result)
        print(obj)
        _inst = cls()  # try parameter-less instantiation
        return

    key = mod.generalattributes_to_key[cls]

    tao_python_result = {"key": key, **tao_python_result}
    obj = cls.model_validate(tao_python_result)

    adapter = pydantic.TypeAdapter(mod.GeneralAttributes)
    obj_discriminator = adapter.validate_python(tao_python_result)

    cls(key=key)  # try parameter-less instantiation

    assert obj_discriminator == obj
    assert type(obj_discriminator) is type(obj)


def try_deserializing(
    mod: ModuleType,
    res: dict[str, PipeOutputStructure],
    classes: dict[str, type[pydantic.BaseModel]],
):
    for key, cls in res.items():
        assert key == cls.class_name

    for name, output_struct in res.items():
        cls: type[pydantic.BaseModel] = classes[name]

        deserialize(mod, output_struct, cls, output_struct.cmd.result)


def main_tao_pystructs():
    structures = generate_structures()
    mod, classes = write_source(
        MODULE_PATH / "_generated.py",
        structures,
    )
    try_deserializing(mod, structures, classes)
    return mod, structures, classes


# def generate_gen_attr_structures():
#     tao = pytao.Tao(
#         init_file="$ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching",
#         noplot=True,
#     )
#
#     res: dict[str, PipeOutputStructure] = {}
#
#     for example in element_key_examples:
#         if tao.init_settings.lattice_file != example.lattice_fn:
#             with pytao.filter_tao_messages_context(
#                 functions=[
#                     "twiss_propagate1",
#                     "closed_orbit_calc",
#                     "track1_cyrstal",
#                 ]
#             ):
#                 tao.init(lattice_file=example.lattice_fn, noplot=True)
#         ele_idx = get_element_index(tao, example.element_names[0])
#         class_name = f"{example.key}Attributes"
#         res[class_name] = PipeOutputStructure.from_cmd(
#             TaoCommandAndResult.from_tao(tao, f"ele:gen_attribs {ele_idx}"),
#             class_name=class_name,
#             skip_prefixes=("units#",),
#             discriminators={"key": example.key},
#             base_class="TaoAttributesModel",
#             # reference_classes=(bmad_structs.genattr,),
#             # mark_optional=(),
#         )
#         # if all(
#         #     not member.param.can_vary for member in res[class_name].members.values()
#         # ):
#         #     res[class_name].frozen = True
#         #     print("frozen", class_name)
#
#         res[class_name].members["units"] = PipeOutputParameter(
#             param=None,
#             comment="Per-attribute unit information.",
#             name="units",
#             python_type="dict[str, str]",
#         )
#
#     return res
#
# def main_gen_attr():
#     structures = generate_gen_attr_structures()
#     mod, classes = write_source(
#         MODULE_PATH / "generated" / "tao_pystructs_gen_attr.py",
#         structures,
#         header_filename=header_filename.with_name("tao_pystructs_gen_attr_header.py"),
#     )
#     try_deserializing(mod, structures, classes)
#     return mod, structures, classes


if __name__ == "__main__":
    structs_json_file = sys.argv[1]
    adapter = pydantic.TypeAdapter(dict[str, ParsedStructure])
    structs_by_name = adapter.validate_json(pathlib.Path(structs_json_file).read_bytes())

    mod, res, classes = main_tao_pystructs()
    # mod_ga, res_ga, classes_ga = main_gen_attr()
