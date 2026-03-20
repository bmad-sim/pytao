from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import NotRequired, TypedDict


class MatrixResult(TypedDict):
    mat6: np.ndarray
    vec0: np.ndarray


class EmFieldResult(TypedDict):
    B1: float
    B2: float
    B3: float
    E1: float
    E2: float
    E3: float


class VarVArrayLineResult(TypedDict):
    ix_v1: int
    var_attrib_name: str
    meas_value: float
    model_value: float
    design_value: float
    useit_opt: bool
    good_user: bool
    weight: float


class DataDArrayInfo(TypedDict):
    ix_d1: int
    data_type: str
    merit_type: str
    ele_ref_name: str
    ele_start_name: str
    ele_name: str
    meas_value: float
    model_value: float
    design_value: float
    useit_opt: bool
    useit_plot: bool
    good_user: bool
    weight: float
    exists: bool


class BuildingWallGraphInfo(TypedDict):
    index: int
    point: int
    offset_x: float
    offset_y: float
    radius: float


class BuildingWallInfo(TypedDict):
    index: int
    name: str
    constraint: str
    shape: str
    color: str
    line_width: float


class BuildingWallGlobalInfo(TypedDict):
    index: int
    z: float
    x: float
    radius: float
    z_center: float
    x_center: float


class DataD1ArrayInfo(TypedDict):
    index: str
    str2: str
    f: str
    name: str
    line: str
    lower_bound: int
    upper_bound: int


EleChamberWallInfo = TypedDict(
    "EleChamberWallInfo",
    {
        "section": int,
        "longitudinal_position": float,
        "z1": float,
        "-z2": float,
    },
)


class EleLordSlaveInfo(TypedDict):
    type: str
    location_name: str
    name: str
    key: str
    status: str


class EleSpinTaylorInfo(TypedDict):
    index: int
    term: str
    coef: float
    exp1: float
    exp2: float
    exp3: float
    exp4: float
    exp5: float
    exp6: float


class EnumInfo(TypedDict):
    number: int
    name: str


class FloorPlanElementInfo(TypedDict):
    branch_index: int
    index: int
    ele_key: str
    end1_r1: float
    end1_r2: float
    end1_theta: float
    end2_r1: float
    end2_r2: float
    end2_theta: float
    line_width: float
    shape: str
    y1: float
    y2: float
    color: str
    label_name: str
    # Only for sbend:
    ele_l: NotRequired[float]
    ele_angle: NotRequired[float]
    ele_e1: NotRequired[float]
    ele_e: NotRequired[float]


class FloorOrbitInfo(TypedDict):
    branch_index: int
    index: int
    ele_key: str
    axis: str
    orbits: list[float]


class LatBranchListInfo(TypedDict):
    index: int
    branch_name: str
    n_ele_track: int
    n_ele_max: int


class LordControlInfo(TypedDict):
    index: int
    name: str
    key: str
    attribute: str
    expression: str
    value: float | None


class SlaveControlInfo(TypedDict):
    branch: int
    index: int
    name: str
    key: str
    attribute: str
    expression: str
    value: float | None


class PlotLatLayoutInfo(TypedDict):
    ix_branch: int
    ix_ele: int
    ele_s_start: float
    ele_s_end: float
    line_width: float
    shape: str
    y1: float
    y2: float
    color: str
    label_name: str


class PlotLineInfo(TypedDict):
    index: int
    x: float
    y: float


class PlotSymbolInfo(TypedDict):
    index: int
    ix_symb: int
    x_symb: float
    y_symb: float


class ShapeListInfo(TypedDict):
    shape_index: int
    ele_name: str
    shape: str
    color: str
    shape_size: float
    type_label: str
    shape_draw: bool
    multi_shape: bool
    line_width: int


class VarGeneralInfo(TypedDict):
    name: str
    line: str
    lbound: int
    ubound: int


class PlaceBufferInfo(TypedDict):
    region: str
    graph: str


class SpinInvariantInfo(TypedDict):
    index: int
    spin1: float
    spin2: float
    spin3: float


class PlotListRegionInfo(TypedDict):
    region: str
    ix: int
    plot_name: str
    visible: bool
    x1: float
    x2: float
    y1: float
    y2: float


class ConstraintDataInfo(TypedDict):
    datum_name: str
    constraint_type_name: str
    ele_name: str
    ele_start_name: str
    ele_ref_name: str
    meas_value: float
    ref_value: float
    model_value: float
    base_value: float
    weight: float
    merit: float
    a_name: str


class ConstraintVarInfo(TypedDict):
    var1_name: str
    attrib_name: str
    meas_value: float
    ref_value: float
    model_value: float
    base_value: float
    weight: float


class EleGridFieldPointInfo(TypedDict):
    i: int
    j: int
    k: int
    data: list[Any]


class EleGenGradMapDerivInfo(TypedDict):
    i: int
    j: int
    k: int
    dz: float
    deriv: float


class VarSlaveInfo(TypedDict):
    index: int
    ix_branch: int
    ix_ele: int


class ShapePatternNameInfo(TypedDict):
    name: str
    line_width: float


class ShapePatternPointInfo(TypedDict):
    s: float
    y: float


class DataParameterLineInfo(TypedDict):
    index: int
    data: list[Any]


class VarV1ArrayDataInfo(TypedDict):
    name: str
    ele_name: str
    attrib_name: str
    meas_value: float
    model_value: float
    design_value: float
    good_user: bool
    useit_opt: bool
