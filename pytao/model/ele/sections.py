"""
Hand-written models for the fillable data sections of an `Element`.

The `get_*` helpers these models query live in `.ele`; they are imported
inside methods to avoid a circular import (`.ele` imports this module for
`Element`'s field types).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import pydantic
from pydantic import Field

from ...util.parsers import Attr, parse_tao_python_data_with_units
from .. import _generated as tao_classes
from ..base import TaoBaseModel, TaoModel
from ..types import NDArray

if TYPE_CHECKING:
    from pytao import Tao

    from .ele import AnyElementID


Which = Literal["base", "model", "design"]
PhotonWho = Literal["base", "material", "curvature"]
ChamberWallWho = Literal["x", "y"]
FloorWhere = Literal["beginning", "center", "end"]


class ElementFloorPosition(TaoBaseModel, extra="forbid"):
    """
    Represents the position and orientation of an element on the floor in a 3D space.

    Attributes
    ----------
    x : float, default 0.0
        The x-coordinate of the position.
    y : float, default 0.0
        The y-coordinate of the position.
    z : float, default 0.0
        The z-coordinate of the position.
    theta : float, default 0.0
        The rotation around the x-axis in radians.
    phi : float, default 0.0
        The rotation around the y-axis in radians.
    psi : float, default 0.0
        The rotation around the z-axis in radians.
    wmat : list of list of float, default empty list
        The transformation matrix representing the orientation of the element.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    psi: float = 0.0
    wmat: NDArray  # NOTE: no default to avoid serialization issues


class ElementFloorItem(TaoBaseModel, extra="forbid"):
    """
    Element floor plan reference and actual position.

    Attributes
    ----------
    reference : ElementFloorPosition
        The reference position on the floor plan.
    actual : ElementFloorPosition
        The actual position on the floor plan.
    """

    reference: ElementFloorPosition = ElementFloorPosition(wmat=np.zeros((3, 3)))
    actual: ElementFloorPosition = ElementFloorPosition(wmat=np.zeros((3, 3)))

    @staticmethod
    def from_tao_output(output: dict[str, np.ndarray]) -> dict[int, ElementFloorItem]:
        """
        Parse the output of tao.ele_floor into a more usable format.
        """
        by_slave = {}
        base_keys = {
            "Reference": {"type": "reference", "slave": 0, "suffix": None},
            "Actual": {"type": "actual", "slave": 0, "suffix": None},
            "Reference-W": {"type": "reference", "slave": 0, "suffix": "-W"},
            "Actual-W": {"type": "actual", "slave": 0, "suffix": "-W"},
        }
        for key, value in output.items():
            groupdict = base_keys.get(key, None)
            if groupdict is None:
                match = ELE_FLOOR_SLAVE_KEY_RE.match(key)

                if match is None:
                    raise ValueError(f"Unexpected key for ele:floor {key!r}")

                groupdict = match.groupdict()

            type_ = groupdict["type"].lower()
            slave_idx = int(groupdict["slave"])
            suffix = groupdict["suffix"]

            if slave_idx not in by_slave:
                by_slave[slave_idx] = ElementFloorItem()
            slave = by_slave[slave_idx]
            vector = slave.actual if type_ == "actual" else slave.reference

            if suffix == "-W":
                vector.wmat = value.reshape(3, 3).T.tolist()
            else:
                vector.x, vector.y, vector.z, vector.theta, vector.phi, vector.psi = value

        return by_slave


ELE_FLOOR_SLAVE_KEY_RE = re.compile(
    r"(?P<type>Reference|Actual)-Slave(?P<slave>\d+)(?P<suffix>-W)?"
)


class ElementFloor(TaoBaseModel, extra="forbid"):
    """
    Represents the floor position of an element.

    Attributes
    ----------
    which : "base", "model", or "design"
    where : "beginning", "center", or "end"
        The location or placement of the element on the floor.
    actual : ElementFloorPosition, optional
        The actual position of the element on the floor.
    reference : ElementFloorPosition, optional
        The reference position of the element on the floor.
    slaves : dict[int, ElementFloorItem]
        A mapping of integer slave numbers to ElementFloorItem
        instances.
    """

    which: Which = pydantic.Field(frozen=True)
    where: FloorWhere = pydantic.Field(frozen=True)

    actual: ElementFloorPosition | None
    reference: ElementFloorPosition | None
    slaves: dict[int, ElementFloorItem]

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        *,
        which: Which,
        where: FloorWhere = "end",
    ):
        from .ele import to_ele_id

        ele = to_ele_id(ele)
        floor = tao.ele_floor(ele, which=which, where=where)
        by_slave = ElementFloorItem.from_tao_output(floor)

        floor = by_slave.pop(0, None)
        return cls(
            which=which,
            where=where,
            slaves=by_slave,
            actual=floor.actual if floor is not None else None,
            reference=floor.reference if floor is not None else None,
        )


class ElementFloorAll(TaoBaseModel, extra="forbid"):
    """
    Element floor positions based on optical trajectory - at its beginning,
    center, or end.

    Attributes
    ----------
    which : "base", "model", or "design"
    beginning : ElementFloor
        The element position at the beginning.
    center : ElementFloor
        The element position at its center.
    end : ElementFloor
        The element position at its end.
    """

    which: Which = pydantic.Field(frozen=True)

    beginning: ElementFloor
    center: ElementFloor
    end: ElementFloor

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        *,
        which: Which,
    ):
        from .ele import to_ele_id

        ele = to_ele_id(ele)
        beginning = ElementFloor.from_tao(tao, ele=ele, which=which, where="beginning")
        center = ElementFloor.from_tao(tao, ele=ele, which=which, where="center")
        end = ElementFloor.from_tao(tao, ele=ele, which=which, where="end")

        return cls(
            which=which,
            beginning=beginning,
            center=center,
            end=end,
        )


class ElementChamberWall(TaoBaseModel, extra="forbid"):
    """
    Represents a chamber wall element in the lattice.

    Attributes
    ----------
    which : "base", "model", or "design"
    index : int
        The index of the chamber wall of the element.
    x : list of ElementChamberWall
        A list of ElementChamberWall objects along the x-axis.
    y : list of ElementChamberWall
        A list of ElementChamberWall objects along the y-axis.
    """

    which: Which = pydantic.Field(frozen=True)
    index: int
    x: list[tao_classes.ElementChamberWall]
    y: list[tao_classes.ElementChamberWall]

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        index: int,
        *,
        which: Which,
    ):
        from .ele import get_chamber_wall, to_ele_id

        ele = to_ele_id(ele)

        return cls(
            which=which,
            index=index,
            x=get_chamber_wall(tao, ele, index, which=which, who="x"),
            y=get_chamber_wall(tao, ele, index, which=which, who="y"),
        )


class ElementWall3D(tao_classes.ElementWall3DBase, extra="forbid"):
    """
    ElementWall3D class representing a 3D wall element in a lattice.

    Attributes
    ----------
    which : "base", "model", or "design"
    index : int
        The index of the wall element.
    table : list of ElementWall3DTable or None, optional
        A table containing wall element details.
    """

    which: Which = pydantic.Field(frozen=True)

    index: int
    table: list[tao_classes.ElementWall3DTable] | None = None

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        index: int,
        *,
        which: Which,
        fill_table: bool = False,
    ):
        from .ele import get_wall3d_base, get_wall3d_table, to_ele_id

        ele = to_ele_id(ele)

        base = get_wall3d_base(tao, ele, index, which=which)
        table = None
        if fill_table:
            table = get_wall3d_table(tao, ele, index, which=which)

        data = base.model_dump()
        data.pop("__class_name__")
        return cls(
            which=which,
            index=index,
            table=table,
            **data,
        )


class ElementPhoton(tao_classes.ElementPhotonBase, extra="forbid"):
    """
    Class representing a element's photon details.

    Attributes
    ----------
    which : "base", "model", or "design"
    has_material : bool
        Whether `material` is present or None.
    has_pixel : bool
        Whether `pixel` is present or None.
    curvature : tao_classes.ElementPhotonCurvature
        Curvature of the photon element.
    material : tao_classes.ElementPhotonMaterial
        Material properties of the photon element.
    """

    which: Which = pydantic.Field(frozen=True)
    curvature: tao_classes.ElementPhotonCurvature
    material: tao_classes.ElementPhotonMaterial

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        *,
        which: Which,
    ):
        from .ele import (
            get_photon_base,
            get_photon_curvature,
            get_photon_material,
            to_ele_id,
        )

        ele = to_ele_id(ele)

        base = get_photon_base(tao, ele, which=which)
        data = base.model_dump()
        data.pop("__class_name__")
        return cls(
            which=which,
            curvature=get_photon_curvature(tao, ele, which=which),
            material=get_photon_material(tao, ele, which=which),
            **data,
        )


class ElementMat6(TaoModel, extra="forbid"):
    """
    Linear transfer map (mat6) data.

    Attributes
    ----------
    mat6 : NDArray of shape (6, 6)
    vec0 : NDArray
    symplectic_error : float
    """

    which: Which = pydantic.Field(frozen=True)

    vec0: NDArray
    mat6: NDArray
    symplectic_error: float = Field(default=0.0, frozen=True)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _handle_legacy_raw_data(cls, data: Any) -> Any:
        """Convert legacy `data_*` keys to a unified `mat6` NDArray."""
        if not isinstance(data, dict):
            return data

        legacy_keys = ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"]

        if "mat6" not in data:
            mat6_raw = [data.pop(k, [0.0] * 6) for k in legacy_keys]
            data["mat6"] = np.asarray(mat6_raw, dtype=float)

        return data

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        *,
        which: Which,
    ):
        from .ele import get_mat6, get_mat6_error, get_mat6_vec0, to_ele_id

        ele = to_ele_id(ele)

        base = get_mat6(tao, ele, which=which)
        vec0 = get_mat6_vec0(tao, ele, which=which)
        err = get_mat6_error(tao, ele, which=which)

        mat6 = np.asarray(
            [base.data_1, base.data_2, base.data_3, base.data_4, base.data_5, base.data_6],
        )
        return cls(
            which=which,
            mat6=mat6,
            vec0=np.asarray(vec0.vec0),
            symplectic_error=err.symplectic_error,
        )


class ElementGridField(tao_classes.ElementGridField, extra="forbid"):
    which: Which = pydantic.Field(frozen=True)

    points: list[tao_classes.ElementGridFieldPoints] | None = None

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        index: int,
        *,
        which: Which,
        fill_points: bool = False,
    ):
        from .ele import get_grid_field_base, get_grid_field_points, to_ele_id

        ele = to_ele_id(ele)

        base = get_grid_field_base(tao, ele, which=which, index=index)
        if fill_points:
            points = get_grid_field_points(tao, ele, which=which, index=index)
        else:
            points = None

        data = base.model_dump()
        data.pop("__class_name__")
        return cls(
            which=which,
            points=points,
            **data,
        )


class ElementSrWakeData(TaoModel):
    """
    Per-element short-range wake data - may be longitudinal or transverse.

    Attributes
    ----------
    z_ref : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_wake"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    z_ref: float = 0.0
    table: list[list[str | float]] = []

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        *,
        which: Which,
        who: Literal["longitudinal", "transverse"],
    ):
        from .ele import to_ele_id

        ele_id = to_ele_id(ele)
        if who == "longitudinal":
            tao_who = "sr_long"
        else:
            tao_who = "sr_trans"

        base_data: dict = tao.ele_wake(ele_id, who=tao_who, which=which)  #  type: ignore
        table_data: list = tao.ele_wake(ele_id, who=f"{tao_who}_table", which=which)  # type: ignore
        return cls(**base_data, table=table_data)


class ElementWake(tao_classes.ElementWakeBase, extra="forbid"):
    which: Which = pydantic.Field(frozen=True)

    sr_long: ElementSrWakeData | None = None
    sr_trans: ElementSrWakeData | None = None
    lr_mode: list[list[str | float]] | None = None

    @pydantic.field_validator("sr_long", "sr_trans", mode="before")
    @classmethod
    def _migrate_sr_wake(cls, value: Any) -> Any:
        if isinstance(value, dict):
            cls_name = value.get("__class_name__")
            if cls_name in ("ElementWakeSrLong", "ElementWakeSrTrans"):
                value_copy = dict(value)
                value_copy["__class_name__"] = "ElementSrWakeData"
                return value_copy

        if isinstance(value, (tao_classes.ElementWakeSrLong, tao_classes.ElementWakeSrTrans)):
            return ElementSrWakeData(z_ref=value.z_ref)

        return value

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        *,
        which: Which,
    ):
        from .ele import get_wake_base, get_wake_sr_long, get_wake_sr_trans, to_ele_id

        ele = to_ele_id(ele)

        base = get_wake_base(tao, ele, which=which)
        sr_long = None
        sr_trans = None
        lr_mode = None
        if base.has_sr_long:
            sr_long = get_wake_sr_long(tao, ele, which=which)
        if base.has_sr_trans:
            sr_trans = get_wake_sr_trans(tao, ele, which=which)
        if base.has_lr_mode:
            lr_mode: list[list[str | float]] = tao.ele_wake(
                ele, which=which, who="lr_mode_table"
            )  # type: ignore

        data = base.model_dump()
        data.pop("__class_name__")
        return cls(
            which=which,
            sr_long=sr_long,
            sr_trans=sr_trans,
            lr_mode=lr_mode,
            **data,
        )


AnyElementMultipoles = (
    tao_classes.ElementMultipoles
    | tao_classes.ElementMultipolesAB
    | tao_classes.ElementMultipolesScaled
)


class _AttributeDict(dict):
    """
    A dictionary-like container that allows for dotted attribute access.
    """

    def __getattr__(self, key: str) -> Any:
        lkey = GeneralAttributes._tao_attr_map_.get(key.lower(), key.lower())
        try:
            return self[lkey]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def _ipython_key_completions_(self) -> list[str]:
        return list(self.keys())

    def __dir__(self) -> list[str]:
        base_dir = set(super().__dir__())
        key_dir = set(self.keys())
        return sorted(base_dir | key_dir)


class GeneralAttributes(TaoModel, extra="allow"):
    # Note: hacky workaround here so we can inspect if attributes can be set
    _tao_command_attr_: ClassVar[str] = "pipe ele:gen_attribs {ele_id}"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    _tao_attr_map_: ClassVar[dict[str, str]] = {
        # Every attribute except for "L" is lowercase - by request
        "l": "L",
    }

    attrs: dict[str, Attr]

    @classmethod
    def _fix_key_case(cls, key: str) -> str:
        return cls._tao_attr_map_.get(key.lower(), key.lower())

    def __getitem__(self, key: str) -> Attr:
        # TODO: GeneralAttributes -> RootModel and then fully override __iter__
        return self.attrs[self._fix_key_case(key)]

    def __setitem__(self, key: str, value) -> None:
        self.attrs[self._fix_key_case(key)].data = value

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _discriminator_validator(
        cls, value: Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> Any:
        if isinstance(value, dict):
            units = value.get("units", None)
            if isinstance(units, dict) and "settable" not in units:
                # Support an older version of attribute storage, where each
                # element key had its own attribute class
                value = dict(value)
                value.pop("units")
                attrs = {
                    key: {
                        "name": key,
                        "data": value,
                        "units": units.get(key),
                        "type": "unknown",
                        "settable": False,
                    }
                    for key, value in value.items()
                    if key not in {"command_args"}
                }
                return handler({"attrs": attrs})

        return handler(value)

    @classmethod
    def _process_tao_data(cls, data) -> dict:
        attrs_by_key = {
            cls._fix_key_case(attr): value
            for attr, value in parse_tao_python_data_with_units(data).items()
        }
        return {"attrs": attrs_by_key}

    # @property
    # def settable_fields(self) -> dict[str, FieldInfo]:
    #     raise NotImplementedError()
