from __future__ import annotations

from dataclasses import dataclass
import functools
import gzip
import json
import pathlib
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
import pydantic
from pydantic import Field
from typing_extensions import Self

from ...errors import TaoCommandError
from ...util import normalize_path
from ...util.parsers import Attr, parse_tao_python_data_with_units
from .. import _generated as tao_classes
from ..base import TaoModel, _check_equality
from .comb import Comb
from .time_stats import _pytao_stats

if TYPE_CHECKING:
    from pytao import Tao


class ElementNotFoundError(Exception):
    pass


class EmptyElementNameError(Exception):
    pass


def to_ele(ele: AnyElementID) -> str:
    """
    Convert input to its string representation.

    Parameters
    ----------
    ele : int, str, or ElementID
        The input element to be converted.

    Returns
    -------
    str
        The string representation of the input element.
    """
    if isinstance(ele, ElementID):
        return ele.tao_string

    if not str(ele):
        raise EmptyElementNameError("No element name specified")
    return str(ele)


def _maybe_int(value: str | None) -> str | int | None:
    """
    Convert a string to an integer if possible.

    Passes through None.

    Parameters
    ----------
    value : str or None

    Returns
    -------
    str, int, or None
    """
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return value


class ElementRange(pydantic.BaseModel, extra="forbid"):
    """
    Multiple Tao elements.
    """

    start: ElementID
    end: ElementID

    @classmethod
    def from_tao(cls, value: str) -> ElementRange:
        """
        Convert a Tao representation of comma-delimited elements to an
        ElementList.

        Parameters
        ----------
        value : str

        Returns
        -------
        ElementList or ElementRange
        """
        if ":" not in value:
            raise ValueError(f"No colon delimiter found in element identifier {value!r}")

        ele_id = ElementID.from_tao(value)
        ele1, ele2 = ele_id.ele_id.split(":")

        common = ele_id.model_dump()
        common.pop("ele_id")

        return cls(
            start=ElementID(ele_id=ele1, **common),
            end=ElementID(ele_id=ele2, **common),
        )

    def as_id(self) -> ElementID:
        common = self.start.model_dump()
        common.pop("ele_id")
        return ElementID(ele_id=f"{self.start.ele_id}:{self.end.ele_id}", **common)

    @property
    def tao_string(self) -> str:
        """This element list represented in Tao command-line interface string form."""
        return self.as_id().tao_string


class ElementList(pydantic.BaseModel, extra="forbid"):
    """
    Multiple Tao elements.
    """

    elements: tuple[ElementID, ...]

    @classmethod
    def from_tao(cls, value: str) -> ElementList:
        """
        Convert a Tao representation of comma-delimited elements to an
        ElementList.

        Parameters
        ----------
        value : str

        Returns
        -------
        ElementList
        """
        if "," not in value:
            raise ValueError(f"No comma delimiter found in element identifier {value!r}")
        return cls(elements=tuple(ElementID.from_tao(part) for part in value.split(",")))

    @property
    def tao_string(self) -> str:
        """This element list represented in Tao command-line interface string form."""
        return ", ".join(ele.tao_string for ele in self.elements)


class ElementIntersection(pydantic.BaseModel, extra="forbid"):
    """
    An intersection of multiple Tao elements.
    """

    elements: tuple[ElementID, ...]

    @classmethod
    def from_tao(cls, value: str) -> ElementIntersection:
        """
        Create an ElementIntersection instance from a Tao format, ampersand-delimited string.

        Parameters
        ----------
        value : str

        Returns
        -------
        ElementIntersection
        """
        if "&" not in value:
            raise ValueError("No intersection found in element identifier {value!r}")
        return cls(elements=tuple(ElementID.from_tao(part) for part in value.split("&")))

    @property
    def tao_string(self) -> str:
        """This element intersection represented in Tao command-line interface string form."""
        return " & ".join(ele.tao_string for ele in self.elements)


class ElementID(pydantic.BaseModel, extra="forbid"):
    """
    An element identifier, which breaks apart a Tao element identifier into its components.

    An element "name" (which can match to multiple elements) in Tao can be of the form
    ```
    {~}{uni@}{branch>>}{key::}ele_id{##N}{+/-offset}
    ```

    where

    | Syntax      | Description                                                                                     |
    | ----------- | ----------------------------------------------------------------------------------------------- |
    | `~`         | Negation character. See below.                                                                  |
    | `key`       | Optional key name ("quadrupole", "sbend", etc.)                                                 |
    | `uni`       | Index of universe.
    | `branch`    | Name or index of branch. May contain the wild cards `*` and `%`.                                |
    | `ele_id`    | Name or index of element. May contain the wild cards `*` and `%`.                               |
    |             | If a name and no branch is given, all branches are searched.                                    |
    |             | If an index and no branch is given, branch 0 is assumed.                                        |
    | `##N`       | N = integer. N^th instance of `ele_id` in the branch.                                           |
    | `+/-offset` | Element offset. For example, `Q1+1` is the element after `Q1` and `Q1-2` is the second element  |
    |             | before `Q1`. Modulo arithmetic is used so the offset wraps around the ends of the lattice.      |
    |             | EG: `BEGINNING-1` gives the END element and `END+1` gives the BEGINNING element.                |

    Note: An old syntax that is still supported is:

    ```
    {key::}{branch>>}ele_id{##N}
    ```

    An element range is of the form:

    ```
    {key::}ele1:ele2
    ```

    Where the range includes ele1 and ele2, and:

    | Parameter | Description                                                                                          |
    |-----------|------------------------------------------------------------------------------------------------------|
    | `key`     | Optional key name ("quadrupole", "sbend", etc.). Also key may be "type", "alias", or "descrip"       |
    |           | in which case the %type, %alias, or %descrip field is matched to instead of the element name.        |
    | `ele1`    | Starting element of the range.                                                                       |
    | `ele2`    | Ending element of the range.                                                                         |

    """

    ele_id: str
    key: str | None = None
    universe: int | Literal["*"] | None = None
    branch: str | int | None = None

    match_number: int | None = None
    match_offset: int | None = None
    negated: bool = False

    @property
    def is_range(self) -> bool:
        return ":" in self.ele_id

    @property
    def without_universe(self) -> ElementID:
        """
        This element ID, excluding the universe number.

        Some Tao commands such as 'show lat' do not accept a universe number.

        Returns
        -------
        ElementID
        """
        return type(self)(
            ele_id=self.ele_id,
            key=self.key,
            # universe=self.universe,
            branch=self.branch,
            match_number=self.match_number,
            match_offset=self.match_offset,
            negated=self.negated,
        )

    @property
    def tao_string(self) -> str:
        """This element represented in Tao command-line interface string form."""
        parts = []
        if self.universe is not None:
            parts.append(f"{self.universe}@")
        if self.negated:
            parts.append("~")
        if self.branch is not None:
            parts.append(f"{self.branch}>>")
        if self.key is not None:
            parts.append(f"{self.key}::")

        if " " in self.ele_id and not self.ele_id.startswith("'"):
            parts.append(f"'{self.ele_id}'")
        else:
            parts.append(self.ele_id)

        if self.match_number is not None:
            parts.append(f"##{self.match_number}")
        if self.match_offset is not None:
            parts.append(f"{self.match_offset:+}")
        return "".join(parts)

    @classmethod
    def from_tao_old_syntax(cls, value: str) -> ElementID:
        """
        Convert a Tao old syntax Element identifier.

        Parameters
        ----------
        value : str

        Returns
        -------
        ElementID
        """
        ele_id = None
        key = None
        universe = None
        branch = None

        match_number = None
        match_offset = None
        negated = False

        if not value.strip():
            raise EmptyElementNameError("No element name specified")

        if "::" in value and ">>" in value:
            assert value.index("::") < value.index(">>")

        remaining = value

        def split_next(delim: str) -> tuple[str | None, str]:
            if delim in remaining:
                return remaining.split(delim, 1)
            return None, remaining

        key, remaining = split_next("::")
        branch, remaining = split_next(">>")

        if "##" in remaining:
            ele_id, match_number = split_next("##")
        else:
            ele_id = remaining

        branch = _maybe_int(branch)
        if match_number is not None:
            match_number = int(match_number)

        if ele_id is None:
            raise ValueError("No element ID found in the string")

        return cls(
            ele_id=ele_id,
            key=key,
            universe=universe,
            branch=branch,
            match_number=match_number,
            match_offset=match_offset,
            negated=negated,
        )

    @classmethod
    def from_tao(cls, value: str) -> ElementID:
        """
        Construct an ElementID from the Tao string representation.

        Parameters
        ----------
        value : str

        Returns
        -------
        ElementID
        """
        if "," in value:
            raise ValueError(
                f"Comma (',') found in the identifier {value!r}.  This indicates multiple elements. "
                f"Use `.from_tao_any` instead."
            )
        if "&" in value:
            raise ValueError(
                f"Ampersand ('&') found in the identifier {value!r}.  This indicates the intersection of multiple elements. "
                f"Use `.from_tao_any` instead."
            )

        value = value.strip()

        if not value:
            raise EmptyElementNameError("No element name specified")

        ele_id = None
        key = None
        universe = None
        branch = None

        match_number = None
        match_offset = None
        negated = False

        if "@" not in value and "::" in value and ">>" in value:
            if value.index("::") < value.index(">>"):
                return cls.from_tao_old_syntax(value)
        remaining = value

        def split_next(delim: str) -> tuple[str, str] | tuple[None, str]:
            if delim in remaining:
                return remaining.split(delim, 1)
            return None, remaining

        # Universe prefix *first*, then negation, then branch/key (mirroring
        # lat_ele_locator).
        universe, remaining = split_next("@")

        if remaining.startswith("~"):
            negated = True
            remaining = remaining[1:]

        branch, remaining = split_next(">>")
        key, remaining = split_next("::")

        if "##" in remaining:
            ele_id, match_number = split_next("##")

            if "+" in match_number:
                match_number, match_offset = match_number.split("+")
                match_offset = int(match_offset)
            elif "-" in match_number:
                match_number, match_offset = match_number.split("-")
                match_offset = -int(match_offset)

            match_number = int(match_number)
            assert match_offset is None or isinstance(match_offset, int)
        elif "+" in remaining:
            ele_id, match_offset = split_next("+")
            match_offset = int(match_offset)
            remaining = ""
        elif "-" in remaining:
            ele_id, match_offset = split_next("-")
            match_offset = -int(match_offset)
            remaining = ""
        else:
            ele_id, remaining = remaining, ""

        branch = _maybe_int(branch)
        universe = _maybe_int(universe)
        if match_number is not None:
            match_number = int(match_number)

        if ele_id is None:
            raise ValueError("No element ID found in the string")

        return cls(
            ele_id=ele_id,
            key=key,
            universe=universe,
            branch=branch,
            match_number=match_number,
            match_offset=match_offset,
            negated=negated,
        )

    @classmethod
    def from_tao_any(
        cls, value: str, *, split_range: bool = False
    ) -> ElementID | ElementList | ElementIntersection | ElementRange:
        """
        Create the most appropriate instance of an Element class for the given
        Tao element string.

        Parameters
        ----------
        value : str
            A string representing an element identifier, a list of elements, or
            an intersection of elements.

        Returns
        -------
        ElementID, ElementList, or ElementIntersection
        """
        if "," in value:
            return ElementList.from_tao(value)
        if split_range and ":" in value.replace("::", " "):
            return ElementRange.from_tao(value)
        if "&" in value:
            return ElementIntersection.from_tao(value)
        return cls.from_tao(value)

    def __str__(self):
        return self.tao_string


AnyElementID = int | str | ElementID

Which = Literal["base", "model", "design"]
PhotonWho = Literal["base", "material", "curvature"]
ChamberWallWho = Literal["x", "y"]
FloorWhere = Literal["beginning", "center", "end"]


def _maybe_reraise(ele: str, ex: TaoCommandError):
    if "Cannot locate element" not in str(ex):
        raise

    if ex.errors:
        msg = ex.errors[0].message
    else:
        msg = "Element not found"

    raise ElementNotFoundError(f"{ele} {msg}") from None


def _catch_element_not_found_error(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TaoCommandError as ex:
            _maybe_reraise(kwargs.get("ele", None), ex)

    return wrapped


def get_element_index(
    tao: Tao,
    ele: AnyElementID,
) -> int:
    """
    Get the index of a specified element from Tao.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The element as a string or ElementID.

    Returns
    -------
    int

    Raises
    ------
    ElementNotFoundError
        If the element cannot be located.

    TaoCommandError
        For other unexpected errors.
    """
    head = get_head(tao, ele=ele, which="model")
    return head.ix_ele


@_catch_element_not_found_error
def get_head(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementHead:
    """
    Retrieve the head of a Tao element.

    Parameters
    ----------
    tao : Tao
        The Tao object instance.
    ele : str or ElementID
        The element identifier, either as a string or an ElementID object.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementHead
    """
    ele = to_ele(ele)
    return tao_classes.ElementHead.from_tao(tao, ele_id=ele, which=which)


@_catch_element_not_found_error
def get_twiss(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementTwiss:
    """
    Retrieve Twiss parameters from a Tao object for a specified element.

    Parameters
    ----------
    tao : Tao
        The Tao object instance.
    ele : str or ElementID
        The element identifier, either as a string or an ElementID object.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementTwiss
        The Twiss parameters of the specified element.
    """
    ele = to_ele(ele)
    return tao_classes.ElementTwiss.from_tao(tao, ele_id=ele, which=which)


@_catch_element_not_found_error
def get_orbit(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementOrbit:
    """
    Get the orbit of an element from the Tao model.

    Parameters
    ----------
    tao : Tao
        The Tao object instance.
    ele : str or ElementID
        The element identifier, either as a string or an ElementID object.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementOrbit
        The orbit of the specified element in the Tao model.
    """
    ele = to_ele(ele)
    return tao_classes.ElementOrbit.from_tao(tao, ele_id=ele, which=which)


@_catch_element_not_found_error
def get_lord_slave(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> list[tao_classes.ElementLordSlave]:
    """
    Retrieve the lord and slave elements from a Tao instance.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The element identifier, either as a string or an ElementID object.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    list of ElementLordSlave
    """
    ele = to_ele(ele)
    adapter = pydantic.TypeAdapter("list[tao_classes.ElementLordSlave]")
    return adapter.validate_python(tao.ele_lord_slave(ele_id=ele))


@_catch_element_not_found_error
def get_chamber_wall(
    tao: Tao,
    ele: AnyElementID,
    index: int,
    who: ChamberWallWho,
    which: Which = "model",
) -> list[tao_classes.ElementChamberWall]:
    """
    Retrieve the chamber wall data for a specified element from Tao.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The identifier of the element. This can be a string or an ElementID instance.
    index : int
        The index of the wall.
    who : ChamberWallWho
        Specifies which chamber wall data to retrieve.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    list of ElementChamberWall
        A list of ElementChamberWall objects containing the chamber wall data for the specified element.
    """
    ele = to_ele(ele)
    adapter = pydantic.TypeAdapter("list[tao_classes.ElementChamberWall]")
    return adapter.validate_python(
        tao.ele_chamber_wall(ele_id=ele, index=index, which=which, who=who)
    )


@_catch_element_not_found_error
def get_wall3d_base(
    tao: Tao,
    ele: AnyElementID,
    index: int,
    which: Which = "model",
) -> tao_classes.ElementWall3DBase:
    """
    Retrieve the 3D wall base information for a specified element from Tao.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The identifier of the element. Can be a string name or an ElementID object.
    index : int
        The index of the wall.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementWall3DBase
        The 3D wall base information of the specified element.
    """
    ele = to_ele(ele)
    return tao_classes.ElementWall3DBase.from_tao(
        tao,
        ele_id=ele,
        index=index,
        which=which,
        who="base",
    )


@_catch_element_not_found_error
def get_wall3d_table(
    tao: Tao,
    ele: AnyElementID,
    index: int,
    which: Which = "model",
) -> list[tao_classes.ElementWall3DTable]:
    """
    Retrieve the 3D wall table for a specified element from Tao.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The element identifier, either as a string or an ElementID.
    index : int
        The index of the wall.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    list[tao_classes.ElementWall3DTable]
        A list of ElementWall3DTable objects for the specified element.
    """
    ele = to_ele(ele)
    adapter = pydantic.TypeAdapter("list[tao_classes.ElementWall3DTable]")
    return adapter.validate_python(
        tao.ele_wall3d(ele_id=ele, index=index, which=which, who="table")
    )


@_catch_element_not_found_error
def get_multipoles(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> AnyElementMultipoles | None:
    """
    Retrieve the multipole coefficients for a specified element in a Tao model.

    Parameters
    ----------
    tao : Tao
        The Tao object.
    ele : str or ElementID
        The identifier of the element. Can be a string name or an ElementID object.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    AnyElementMultipoles or None
        The multipole coefficients for the specified element, or None if not found.
    """
    ele = to_ele(ele)

    multipoles: dict = tao.ele_multipoles(ele_id=ele, which=which)
    adapter = pydantic.TypeAdapter(AnyElementMultipoles)
    if not multipoles["multipoles_on"]:
        return None

    multipoles["command_args"] = {"ele_id": ele, "which": which}
    if not len(multipoles.get("data", [])):
        # perf: it's ambiguous, so choose a general class
        return tao_classes.ElementMultipoles.model_validate(multipoles)
    return adapter.validate_python(multipoles)


@_catch_element_not_found_error
def get_bunch_params(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementBunchParams:
    """
    Retrieve the bunch parameters of a specified element in a Tao instance.

    Parameters
    ----------
    tao : Tao
        Tao instance.
    ele : str or ElementID
        Identifier for the element, either as a string or an ElementID object.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementBunchParams
    """
    ele = to_ele(ele)
    return tao_classes.ElementBunchParams.from_tao(tao, ele_id=ele, which=which)


@_catch_element_not_found_error
def get_photon_base(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementPhotonBase:
    """
    Retrieve the photon base information of a specified element in a Tao instance.

    Parameters
    ----------
    tao : Tao
        Tao instance.
    ele : str or ElementID
        Identifier for the element, either as a string or an ElementID object.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementPhotonBase
        The photon base of the specified element.
    """
    ele = to_ele(ele)
    return tao_classes.ElementPhotonBase.from_tao(tao, ele_id=ele, which=which, who="base")


@_catch_element_not_found_error
def get_photon_material(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementPhotonMaterial:
    """
    Retrieve the photon material properties of a specified element from a Tao object.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The identifier of the element.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementPhotonMaterial
        The photon material properties of the specified element.
    """
    ele = to_ele(ele)
    return tao_classes.ElementPhotonMaterial.from_tao(
        tao, ele_id=ele, which=which, who="material"
    )


@_catch_element_not_found_error
def get_photon_curvature(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementPhotonCurvature:
    """
    Get the photon curvature for a specified element in Tao.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The element identifier.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementPhotonCurvature
        The photon curvature of the specified element.
    """
    ele = to_ele(ele)
    return tao_classes.ElementPhotonCurvature.from_tao(
        tao, ele_id=ele, which=which, who="curvature"
    )


@_catch_element_not_found_error
def get_grid_field_base(
    tao: Tao,
    ele: AnyElementID,
    index: int,
    which: Which = "model",
) -> tao_classes.ElementGridField:
    """
    Get the base grid field of a specified element from Tao.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The element identifier, either as a string or an ElementID object.
    index : int
        The index of the element's grid field.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementGridField
    """
    ele = to_ele(ele)
    return tao_classes.ElementGridField.from_tao(
        tao, ele_id=ele, which=which, index=index, who="base"
    )


@_catch_element_not_found_error
def get_grid_field_points(
    tao: Tao,
    ele: AnyElementID,
    index: int,
    which: Which = "model",
) -> list[tao_classes.ElementGridFieldPoints]:
    """
    Retrieve the grid field points for a specified element in Tao.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The element identifier, either as a string or an ElementID object.
    index : int
        The grid field instance.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    list of ElementGridFieldPoints
        A list of ElementGridFieldPoints corresponding to the specified element.
    """
    ele = to_ele(ele)
    adapter = pydantic.TypeAdapter("list[tao_classes.ElementGridFieldPoints]")
    return adapter.validate_python(
        tao.ele_grid_field(ele_id=ele, which=which, index=index, who="points")
    )


@_catch_element_not_found_error
def get_wake_base(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementWakeBase:
    """
    Get the wake base of a Tao element.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The identifier of the element whose wake base is to be retrieved.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementWakeBase
        The wake base of the specified Tao element.
    """
    ele = to_ele(ele)
    return tao_classes.ElementWakeBase.from_tao(tao, ele_id=ele, which=which, who="base")


@_catch_element_not_found_error
def get_wake_sr_long(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementWakeSrLong:
    """
    Get the short-range longitudinal wake of a specific element in the Tao model.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The element identifier.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementWakeSrLong
        The short-range longitudinal wake of the specified element.
    """
    ele = to_ele(ele)
    return tao_classes.ElementWakeSrLong.from_tao(tao, ele_id=ele, which=which, who="sr_long")


@_catch_element_not_found_error
def get_wake_sr_trans(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementWakeSrTrans:
    """
    Retrieve the short-range transverse wakefield response of a specified
    element in a Tao model.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The element identifier, which can be either a string or an ElementID instance.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementWakeSrTrans
        The transverse wakefield response of the specified element.
    """
    ele = to_ele(ele)
    return tao_classes.ElementWakeSrTrans.from_tao(
        tao, ele_id=ele, which=which, who="sr_trans"
    )


@_catch_element_not_found_error
def get_mat6(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementMat6:
    """
    Get the 6x6 linear transfer map (mat6 matrix) for a specified element in Tao.

    Parameters
    ----------
    tao : Tao
        An instance of the Tao class.
    ele : str or ElementID
        The identifier of the element. This can be a string or an ElementID object.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementMat6
    """
    ele = to_ele(ele)
    return tao_classes.ElementMat6.from_tao(tao, ele_id=ele, which=which, who="mat6")


@_catch_element_not_found_error
def get_mat6_vec0(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementMat6Vec0:
    """
    Retrieve the 6-vector for a specified element from a Tao model.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The identifier of the element.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementMat6Vec0
        The 6-vector associated with the specified element.
    """
    ele = to_ele(ele)
    return tao_classes.ElementMat6Vec0.from_tao(tao, ele_id=ele, which=which, who="vec0")


@_catch_element_not_found_error
def get_mat6_error(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
) -> tao_classes.ElementMat6Error:
    """
    Retrieve the 6x6 linear transfer map matrix error for a specified element.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The identifier of the element for which to retrieve the error matrix.
    which : "base", "model", or "design", default="model"

    Returns
    -------
    tao_classes.ElementMat6Error
    """
    ele = to_ele(ele)
    return tao_classes.ElementMat6Error.from_tao(tao, ele_id=ele, which=which, who="err")


@_catch_element_not_found_error
def get_comb(
    tao: Tao,
    ele: AnyElementID,
    which: Which = "model",
    *,
    head: tao_classes.ElementHead | None = None,
    comb: Comb | None = None,
) -> Comb:
    """
    Retrieve Comb data for the given element.

    Parameters
    ----------
    tao : Tao
        The Tao instance.
    ele : str or ElementID
        The identifier of the element for which to retrieve the error matrix.
    which : "base", "model", or "design", default="model"
    comb : Comb or None, optional
        If available, the provided Comb data can be reused for multiple
        elements and significantly speed up using `get_comb` on a full lattice.

    Returns
    -------
    tao_classes.ElementMat6Error
    """
    ele = to_ele(ele)
    if comb is None:
        comb = Comb.from_tao(tao, which=which)
    if head is None:
        head = get_head(tao=tao, ele=ele, which=which)
    return comb.slice_by_s(head.s_start, head.s)


class ElementFloorPosition(pydantic.BaseModel, extra="forbid"):
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
    wmat: list[list[float]] = Field(default_factory=list)


class ElementFloorItem(pydantic.BaseModel, extra="forbid"):
    """
    Element floor plan reference and actual position.

    Attributes
    ----------
    reference : ElementFloorPosition
        The reference position on the floor plan.
    actual : ElementFloorPosition
        The actual position on the floor plan.
    """

    reference: ElementFloorPosition = Field(default_factory=ElementFloorPosition)
    actual: ElementFloorPosition = Field(default_factory=ElementFloorPosition)

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


class ElementFloor(pydantic.BaseModel, extra="forbid"):
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
        ele = to_ele(ele)
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


class ElementFloorAll(pydantic.BaseModel, extra="forbid"):
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
        ele = to_ele(ele)
        beginning = ElementFloor.from_tao(tao, ele=ele, which=which, where="beginning")
        center = ElementFloor.from_tao(tao, ele=ele, which=which, where="center")
        end = ElementFloor.from_tao(tao, ele=ele, which=which, where="end")

        return cls(
            which=which,
            beginning=beginning,
            center=center,
            end=end,
        )


class ElementChamberWall(pydantic.BaseModel, extra="forbid"):
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
        ele = to_ele(ele)

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
        ele = to_ele(ele)

        base = get_wall3d_base(tao, ele, index, which=which)
        table = None
        if fill_table:
            table = get_wall3d_table(tao, ele, index, which=which)

        return cls(
            which=which,
            index=index,
            table=table,
            **base.model_dump(),
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
        ele = to_ele(ele)

        base = get_photon_base(tao, ele, which=which)
        return cls(
            which=which,
            curvature=get_photon_curvature(tao, ele, which=which),
            material=get_photon_material(tao, ele, which=which),
            **base.model_dump(),
        )


class ElementMat6(
    tao_classes.ElementMat6,
    tao_classes.ElementMat6Vec0,
    tao_classes.ElementMat6Error,
    extra="forbid",
):
    """
    Linear transfer map (mat6) data.

    Attributes
    ----------
    data_1 : sequence of floats
    data_2 : sequence of floats
    data_3 : sequence of floats
    data_4 : sequence of floats
    data_5 : sequence of floats
    data_6 : sequence of floats
    vec0 : sequence of floats
    symplectic_error : float
    """

    which: Which = pydantic.Field(frozen=True)

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        *,
        which: Which,
    ):
        ele = to_ele(ele)

        mat6 = get_mat6(tao, ele, which=which)
        vec0 = get_mat6_vec0(tao, ele, which=which)
        err = get_mat6_error(tao, ele, which=which)

        args = {
            **mat6.model_dump(),
            **vec0.model_dump(),
            **err.model_dump(),
        }
        return cls(
            which=which,
            **args,
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
        ele = to_ele(ele)

        base = get_grid_field_base(tao, ele, which=which, index=index)
        if fill_points:
            points = get_grid_field_points(tao, ele, which=which, index=index)
        else:
            points = None
        return cls(
            which=which,
            points=points,
            **base.model_dump(),
        )


class ElementWake(tao_classes.ElementWakeBase, extra="forbid"):
    which: Which = pydantic.Field(frozen=True)

    sr_long: tao_classes.ElementWakeSrLong | None = None
    sr_trans: tao_classes.ElementWakeSrTrans | None = None

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        *,
        which: Which,
    ):
        ele = to_ele(ele)

        base = get_wake_base(tao, ele, which=which)
        sr_long = None
        sr_trans = None
        if base.has_sr_long:
            sr_long = get_wake_sr_long(tao, ele, which=which)
        if base.has_sr_trans:
            sr_trans = get_wake_sr_trans(tao, ele, which=which)
        return cls(
            which=which,
            sr_long=sr_long,
            sr_trans=sr_trans,
            **base.model_dump(),
        )


AnyElementMultipoles = (
    tao_classes.ElementMultipoles
    | tao_classes.ElementMultipolesAB
    | tao_classes.ElementMultipolesScaled
)


class GeneralAttributes(TaoModel, extra="allow"):
    # Note: hacky workaround here so we can inspect if attributes can be set
    _tao_command_attr_: ClassVar[str] = "pipe ele:gen_attribs {ele_id}"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    attrs: dict[str, Attr]

    def __getitem__(self, key: str) -> Attr:
        # TODO: GeneralAttributes -> RootModel and then fully override __iter__
        return self.attrs[key]

    def __setitem__(self, key: str, value) -> None:
        self.attrs[key].data = value

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
        return {"attrs": parse_tao_python_data_with_units(data)}

    # @property
    # def settable_fields(self) -> dict[str, FieldInfo]:
    #     raise NotImplementedError()


@dataclass
class FillDefault:
    attr: str


class Element(pydantic.BaseModel, extra="forbid"):
    """
    Represents a Tao element with various attributes used in simulations.

    Each attribute marked with a default value may be filled in or updated on-demand.

    Attributes
    ----------
    ele : str
        The element name or identifier.
    which : "base", "model", or "design"
    head : ElementHead
        The head data of the element.
    attrs : GeneralAttributes or None, default=None
        General attributes associated with the element.  The information held
        differs depending on the element's key (i.e., `ele.head.key`).
    chamber_walls : list[ElementChamberWall] or None, default=None
        List of chamber walls.
    control_vars : dict[str, float] or None, default=None
        Dictionary of control variables with their corresponding current
        values.
    floor : ElementFloorAll or None, default=None
        Floor positions.
    grid_field : list[ElementGridField] or None, default=None
        List of grid field data.
    lord_slave : list[tao_classes.ElementLordSlave] or None, default=None
        List of lord-slave relationships.
    mat6 : ElementMat6 or None, default=None
        Mat6 (linear transfer map) information.
    multipoles : AnyElementMultipoles or None, default=None
        Multipoleattributes.
    orbit : ElementOrbit or None, default=None
        Orbit attributes.
    photon : ElementPhoton or None, default=None
        Photon attributes.
    twiss : ElementTwiss or None, default=None
        Twiss parameters.
    wake : ElementWake or None, default=None
        Wake attributes.
    wall3d : list[ElementWall3D] or None, default=None
        List of 3D walls.
    """

    DEFAULTS: ClassVar[set[str]] = {
        "attrs",
        "bunch_params",
        "chamber_walls",
        # "comb",
        "control_vars",
        "floor",
        "grid_field",
        # "grid_field_points",
        "lord_slave",
        "mat6",
        "multipoles",
        "orbit",
        "photon",
        "twiss",
        "wake",
        "wall3d",
        # "wall3d_table",
    }

    ele: str = pydantic.Field(frozen=True)
    which: Which = pydantic.Field(frozen=True)

    head: tao_classes.ElementHead
    attrs: GeneralAttributes | None = None
    bunch_params: tao_classes.ElementBunchParams | None = None
    chamber_walls: list[ElementChamberWall] | None = None
    comb: Comb | None = None
    control_vars: dict[str, float] | None = None
    floor: ElementFloorAll | None = None
    grid_field: list[ElementGridField] | None = None
    lord_slave: list[tao_classes.ElementLordSlave] | None = None
    mat6: ElementMat6 | None = None
    multipoles: AnyElementMultipoles | None = None
    orbit: tao_classes.ElementOrbit | None = None
    photon: ElementPhoton | None = None
    twiss: tao_classes.ElementTwiss | None = None
    wake: ElementWake | None = None
    wall3d: list[ElementWall3D] | None = None

    @property
    def id(self) -> ElementID:
        """The fully-qualified ElementID, including universe/branch/key."""
        return ElementID(
            ele_id=self.head.name,
            key=self.head.key,
            universe=self.head.universe,
            branch=self.head.ix_branch,
            # match_number=self.ele.match_number,
            # match_offset=self.ele.match_offset,
        )

    def __eq__(self, other) -> bool:
        return _check_equality(self, other)

    @property
    def name(self) -> str:
        return self.head.name

    @property
    def key(self) -> str:
        return self.head.key

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele: AnyElementID,
        *,
        which: Which = "model",
        defaults: bool = True,
        # Individually fillable elements:
        attrs: bool | FillDefault = FillDefault("attrs"),
        bunch_params: bool | FillDefault = FillDefault("bunch_params"),
        chamber_walls: bool | FillDefault = FillDefault("chamber_walls"),
        comb: bool | FillDefault = FillDefault("comb"),
        control_vars: bool | FillDefault = FillDefault("control_vars"),
        floor: bool | FillDefault = FillDefault("floor"),
        grid_field: bool | FillDefault = FillDefault("grid_field"),
        grid_field_points: bool | FillDefault = FillDefault("grid_field_points"),
        lord_slave: bool | FillDefault = FillDefault("lord_slave"),
        mat6: bool | FillDefault = FillDefault("mat6"),
        multipoles: bool | FillDefault = FillDefault("multipoles"),
        orbit: bool | FillDefault = FillDefault("orbit"),
        photon: bool | FillDefault = FillDefault("photon"),
        twiss: bool | FillDefault = FillDefault("twiss"),
        wake: bool | FillDefault = FillDefault("wake"),
        wall3d: bool | FillDefault = FillDefault("wall3d"),
        wall3d_table: bool | FillDefault = FillDefault("wall3d_table"),
        comb_data: Comb | None = None,
    ):
        """
        Create an `Element` by querying Tao.

        Use `defaults` to fill the most commonly-used element information.
        To disregard the defaults, individual items may be excluded by passing
        `False`, or included by passing `True`.

        Notes
        -----

        Defaults for the data to query are set as follows:

        >>> from pytao.model import Element
        >>> print(Element.DEFAULTS)
        {'attrs', 'bunch_params', 'chamber_walls', 'control_vars', 'floor',
        'grid_field', 'lord_slave', 'mat6', 'multipoles', 'orbit', 'photon',
        'twiss', 'wake', 'wall3d'}

        With the following, the default will change to only query `attrs`:

        >>> Element.DEFAULTS = {"attrs"}

        Examples
        --------

        Get an Element with the defaults (loads attrs, twiss, orbit, etc.):
        >>> ele = Element.from_tao(tao, "1")

        Get an Element but skip orbit calculations:
        >>> ele = Element.from_tao(tao, "1", orbit=False)

        Get an Element AND add comb data (usually off):
        >>> ele = Element.from_tao(tao, "1", comb=True)

        Get a minimal Element (disable everything explicit):
        >>> ele = Element.from_tao(tao, "1", defaults=False)

        Parameters
        ----------
        tao : Tao
            The Tao instance.
        ele : int, str, or ElementID
            The element identifier.
        which : "base", "model", or "design", optional
            Specifies which Tao lattice to use, by default "model".
        defaults : bool, default=True
            Fill default items.  Defaults are set by name in `Element.DEFAULTS`.
        attrs : bool, optional
            Fill general attributes.
        bunch_params : bool, optional
            Fill bunch parameters.
        chamber_walls : bool, optional
            Fill chamber wall data.
        comb : bool, default=False
            Fill comb data.  If available, pass in `comb_data` as well to avoid
            querying Tao again for the full comb data.
        comb_data : Comb or None, optional
            Only relevant if `comb=True`.
            If available, provide `comb_data` to avoid querying Tao again for
            the full comb data.
        control_vars : bool, optional
            Fill control variables.
        floor : bool, optional
            Fill floor data.
        grid_field : bool, optional
            Fill grid field data.
        grid_field_points : bool, default=False
            Fill grid field points data.
        lord_slave : bool, optional
            Fill lord-slave relationships.
        mat6 : bool, optional
            Fill mat6 data.
        multipoles : bool, optional
            Fill multipole data.
        orbit : bool, optional
            Fill orbit data.
        photon : bool, optional
            Fill photon data.
        twiss : bool, optional
            Fill twiss parameters.
        wake : bool, optional
            Fill wake data.
        wall3d : bool, optional
            Fill 3D wall data.
        wall3d_table : bool, optional
            Fill 3D wall table data.
        """
        ele = to_ele(ele)

        head = get_head(tao=tao, ele=ele, which=which)
        instance = cls(which=which, head=head, ele=ele)

        def should_fill(flag: bool | FillDefault):
            if flag is True or flag is False:
                return flag
            if not isinstance(flag, FillDefault):
                raise ValueError(f"Unexpected flag: {flag}")

            return defaults and (flag.attr in cls.DEFAULTS)

        instance.fill(
            tao,
            head=False,
            attrs=should_fill(attrs),
            bunch_params=should_fill(bunch_params),
            chamber_walls=should_fill(chamber_walls),
            control_vars=should_fill(control_vars),
            comb=should_fill(comb),
            floor=should_fill(floor),
            grid_field=should_fill(grid_field),
            grid_field_points=should_fill(grid_field_points),
            lord_slave=should_fill(lord_slave),
            mat6=should_fill(mat6),
            multipoles=should_fill(multipoles),
            orbit=should_fill(orbit),
            photon=should_fill(photon),
            twiss=should_fill(twiss),
            wake=should_fill(wake),
            wall3d=should_fill(wall3d),
            wall3d_table=should_fill(wall3d_table),
            comb_data=comb_data,
            use_cache=True,
        )
        return instance

    @_pytao_stats.time_decorator
    def _fill_head(self, tao: Tao):
        self.head = get_head(tao=tao, ele=self.ele, which=self.which)

    @_pytao_stats.time_decorator
    def _fill_attrs(self, tao: Tao):
        self.attrs = GeneralAttributes.from_tao(tao=tao, ele_id=self.ele, which=self.which)

    @_pytao_stats.time_decorator
    def _fill_bunch_params(self, tao: Tao):
        self.bunch_params = get_bunch_params(tao=tao, ele=self.ele, which=self.which)

    @_pytao_stats.time_decorator
    def _fill_floor(self, tao: Tao):
        self.floor = ElementFloorAll.from_tao(tao=tao, ele=self.ele, which=self.which)

    @_pytao_stats.time_decorator
    def _fill_comb(self, tao: Tao, comb_data: Comb | None):
        self.comb = get_comb(
            tao=tao, ele=self.ele, which=self.which, head=self.head, comb=comb_data
        )

    @_pytao_stats.time_decorator
    def _fill_control_vars(self, tao: Tao):
        if self.head.has_control:
            self.control_vars = cast(
                dict[str, float], tao.ele_control_var(ele_id=self.ele, which=self.which)
            )
        else:
            self.control_vars = None

    @_pytao_stats.time_decorator
    def _fill_lord_slave(self, tao: Tao):
        if self.head.has_lord_slave:
            self.lord_slave = get_lord_slave(tao=tao, ele=self.ele, which=self.which)
        else:
            self.lord_slave = None

    @_pytao_stats.time_decorator
    def _fill_photon(self, tao: Tao):
        if self.head.has_photon:
            self.photon = ElementPhoton.from_tao(tao=tao, ele=self.ele, which=self.which)
        else:
            self.photon = None

    @_pytao_stats.time_decorator
    def _fill_orbit(self, tao: Tao):
        self.orbit = get_orbit(tao=tao, ele=self.ele, which=self.which)

    @_pytao_stats.time_decorator
    def _fill_twiss(self, tao: Tao):
        if self.head.has_twiss:
            self.twiss = get_twiss(tao=tao, ele=self.ele, which=self.which)
        else:
            self.twiss = None

    @_pytao_stats.time_decorator
    def _fill_multipoles(self, tao: Tao):
        self.multipoles = get_multipoles(tao=tao, ele=self.ele, which=self.which)

    @_pytao_stats.time_decorator
    def _fill_wall3d(self, tao: Tao, fill_table: bool):
        if self.head.has_wall3d > 0:
            self.wall3d = [
                ElementWall3D.from_tao(
                    tao=tao,
                    index=index,
                    ele=self.ele,
                    which=self.which,
                    fill_table=fill_table,
                )
                for index in range(1, self.head.has_wall3d + 1)
            ]
        else:
            self.wall3d = None

    @_pytao_stats.time_decorator
    def _fill_chamber_walls(self, tao: Tao):
        if self.head.has_wall3d > 0:
            self.chamber_walls = [
                ElementChamberWall.from_tao(
                    tao=tao, index=index, ele=self.ele, which=self.which
                )
                for index in range(1, self.head.has_wall3d + 1)
            ]
        else:
            self.chamber_walls = None

    @_pytao_stats.time_decorator
    def _fill_grid_field(self, tao: Tao, points: bool = False):
        if self.head.num_grid_field > 0:
            self.grid_field = [
                ElementGridField.from_tao(
                    tao=tao,
                    ele=self.ele,
                    which=self.which,
                    index=index,
                    fill_points=points,
                )
                for index in range(1, self.head.num_grid_field + 1)
            ]
        else:
            self.grid_field = None

    @_pytao_stats.time_decorator
    def _fill_mat6(self, tao: Tao):
        if self.head.has_mat6:
            self.mat6 = ElementMat6.from_tao(tao=tao, ele=self.ele, which=self.which)
        else:
            self.mat6 = None

    @_pytao_stats.time_decorator
    def _fill_wake(self, tao: Tao):
        if self.head.has_wake:
            self.wake = ElementWake.from_tao(tao=tao, ele=self.ele, which=self.which)
        else:
            self.wake = None

    def fill(
        self,
        tao: Tao,
        *,
        head: bool = True,
        attrs: bool = True,
        bunch_params: bool = True,
        comb: bool = False,
        control_vars: bool = True,
        floor: bool = True,
        lord_slave: bool = True,
        photon: bool = True,
        orbit: bool = True,
        twiss: bool = True,
        grid_field: bool = True,
        grid_field_points: bool = False,
        mat6: bool = True,
        chamber_walls: bool = True,
        wall3d: bool = True,
        wall3d_table: bool = False,
        multipoles: bool = True,
        wake: bool = True,
        # Others:
        comb_data: Comb | None = None,
        use_cache: bool = True,
    ):
        """
        Fills various attributes of the Tao object based on the provided flags.

        Parameters
        ----------
        tao : Tao
            The Tao instance to retrieve information from.
        head : bool, default=True
            Update the head attribute.
        attrs : bool, default=True
            Fill attribute data.
        bunch_params : bool, default=True
            Fill bunch parameters.
        comb : bool or None, default=False
            Fill comb data.  If available, pass in `comb_data` as well to avoid
            querying Tao again for the full comb data.
        comb_data : Comb or None, default=None
            Only relevant if `comb=True`.
            If available, provide `comb_data` to avoid querying Tao again for
            the full comb data.
        control_vars : bool, default=True
            Fill control variables.
        floor : bool, default=True
            Fill the floor attribute.
        lord_slave : bool, default=True
            Fill lord-slave relationships.
        photon : bool, default=True
            Fill the photon attribute.
        orbit : bool, default=True
            Fill orbit data.
        twiss : bool, default=True
            Fill Twiss parameters.
        grid_field : bool, default=True
            Fill grid field data.
        grid_field_points : bool, default=False
            Fill grid field points data. Default is False.
        mat6 : bool, default=True
            Fill MAT6 data.
        chamber_walls : bool, default=True
            Fill chamber wall data.
        wall3d : bool, default=True
            Fill 3D wall data.
        wall3d_table : bool, default=False
            Fill 3D wall table data. Default is False.
        multipoles : bool, default=True
            Fill multipole data.
        wake : bool, default=True
            Fill wake data.
        use_cache : bool, default=True
            use cached data if available.  Will not overwrite already-fetched
            data.
        """

        def should_update(obj):
            return obj is None or not use_cache

        if head and should_update(self.head):
            self._fill_head(tao)
        if attrs and should_update(self.attrs):
            self._fill_attrs(tao)
        if bunch_params and should_update(self.bunch_params):
            self._fill_bunch_params(tao)
        if comb and should_update(self.comb):
            self._fill_comb(tao, comb_data=comb_data)
        if control_vars and should_update(self.control_vars):
            self._fill_control_vars(tao)
        if floor and should_update(self.floor):
            self._fill_floor(tao)
        if lord_slave and should_update(self.lord_slave):
            self._fill_lord_slave(tao)
        if photon and should_update(self.photon):
            self._fill_photon(tao)
        if orbit and should_update(self.orbit):
            self._fill_orbit(tao)
        if twiss and should_update(self.twiss):
            self._fill_twiss(tao)
        if grid_field or grid_field_points:
            if self.grid_field is None:
                have_points = False
            else:
                have_points = any(fld.points is not None for fld in self.grid_field)
            if should_update(self.grid_field) or (not have_points and grid_field_points):
                self._fill_grid_field(tao, points=grid_field_points)
        if mat6 and should_update(self.mat6):
            self._fill_mat6(tao)
        if chamber_walls and should_update(self.chamber_walls):
            self._fill_chamber_walls(tao)
        if wall3d and should_update(self.wall3d):
            self._fill_wall3d(tao, fill_table=wall3d_table)
        if multipoles and should_update(self.multipoles):
            self._fill_multipoles(tao)
        if wake and should_update(self.wake):
            self._fill_wake(tao)


class Lattice(pydantic.BaseModel):
    """
    A class representing a Bmad Lattice.

    Attributes
    ----------
    which : "base", "model", or "design"
    elements : tuple of Element
        A tuple containing elements that make up the lattice.
    """

    which: Which
    elements: tuple[Element, ...]

    filename: pathlib.Path | None = Field(default=None, exclude=True)

    def __str__(self):
        fn = f"file={self.filename} " if self.filename else ""
        return f"<Lattice {fn}which={self.which} with {len(self.elements)} elements>"

    @property
    def by_element_key(self) -> dict[str, list[Element]]:
        """A dictionary of element key to list of `Element`s with that key."""
        by_key = {}
        for ele in self.elements:
            by_key.setdefault(ele.head.key, []).append(ele)
        return by_key

    @property
    def by_element_name(self) -> dict[str, Element]:
        """A dictionary of element name to `Element`."""
        return {ele.head.name: ele for ele in self.elements}

    @property
    def by_element_index(self) -> dict[int, Element]:
        """A dictionary of element index to `Element`."""
        return {ele.head.ix_ele: ele for ele in self.elements}

    @classmethod
    def from_tao_eles(
        cls,
        tao: Tao,
        eles: list[AnyElementID],
        *,
        which: Which = "model",
        comb_data: Comb | None = None,
        comb: bool = False,
        **kwargs,
    ):
        """
        Create a `Lattice` object from a list of element names, indices, or IDs.

        Parameters
        ----------
        tao : Tao
        eles : list of int, str, or ElementID
            Element names, indices, or identifiers.
        which : "base", "model", or "design", default="model"
        **kwargs : dict
            Additional keyword arguments passed to `Element.from_tao`.

        Returns
        -------
        Lattice
        """
        if comb and comb_data is None:
            # If 'comb' is specified for all elements, calculate it once ahead
            # of time.
            comb_data = Comb.from_tao(tao)
        elements = tuple(
            Element.from_tao(
                tao, ele=ele, which=which, comb=comb, comb_data=comb_data, **kwargs
            )
            for ele in eles
        )
        return cls(
            which=which,
            elements=elements,
        )

    @classmethod
    def from_tao_unique(
        cls,
        tao: Tao,
        *,
        which: Which = "model",
        track_start: ElementID | str | int | None = None,
        track_end: ElementID | str | int | None = None,
        **kwargs,
    ):
        """
        Create a `Lattice` object from unique elements of the lattice.

        Parameters
        ----------
        tao : Tao
        which : "base", "model", or "design", default="model"
        track_start : str, optional
            The first element to get information from (inclusive).  Defaults to
            element 0.
            This does not need to be a unique element itself.
        track_end : str, optional
            The last element to get information from (inclusive).  Defaults to
            the last element in the lattice.
            This does not need to be a unique element itself.
        **kwargs : dict
            Additional keyword arguments passed to `Element.from_tao`.

        Returns
        -------
        Lattice
        """
        indices: list[int] = [
            int(idx) for idx in tao.lat_list("*", "ele.ix_ele", flags="-no_slaves")
        ]
        ix_start = get_element_index(tao, track_start) if track_start else 0
        ix_end = get_element_index(tao, track_end) if track_end else max(indices)
        return cls.from_tao_eles(
            tao=tao,
            eles=[ix_ele for ix_ele in indices if ix_start <= ix_ele <= ix_end],
            which=which,
            **kwargs,
        )

    @classmethod
    def from_tao_tracking(
        cls,
        tao: Tao,
        *,
        track_start: ElementID | str | int | None = None,
        track_end: ElementID | str | int | None = None,
        which: Which = "model",
        orbit: bool = True,
        twiss: bool = True,
        **kwargs,
    ):
        """
        Create a `Lattice` object from tracking elements of the lattice.

        Parameters
        ----------
        tao : Tao
        which : "base", "model", or "design", default="model"
        track_start : str, optional
            The first element to get information from (inclusive).  Defaults to
            element 0.
        track_end : str, optional
            The last element to get information from (inclusive).  Defaults to
            the last element in the lattice.
        orbit : bool, default=True
            Orbit information is included by default.
        twiss : bool, default=True
            Twiss information is included by default.
        **kwargs : dict
            Additional keyword arguments passed to `Element.from_tao`.

        Returns
        -------
        Lattice
        """

        indices: list[int] = [
            int(idx) for idx in tao.lat_list("*", "ele.ix_ele", flags="-track_only")
        ]
        ix_start = get_element_index(tao, track_start) if track_start else 0
        ix_end = get_element_index(tao, track_end) if track_end else max(indices)
        return cls.from_tao_eles(
            tao=tao,
            eles=[ix_ele for ix_ele in indices if ix_start <= ix_ele <= ix_end],
            which=which,
            orbit=orbit,
            twiss=twiss,
            **kwargs,
        )

    @classmethod
    def from_file(cls: type[Self], filename: str | pathlib.Path) -> Self:
        fname = normalize_path(filename)
        if fname.suffix.lower() == ".gz":
            with gzip.open(fname, "rt", encoding="utf-8") as fp:
                data = json.load(fp)
        else:
            with open(fname) as fp:
                data = json.load(fp)

        return cls(**data, filename=pathlib.Path(filename))

    def write(
        self,
        filename: str | pathlib.Path,
        *,
        exclude_defaults: bool = False,
        indent: int | str | None = None,
    ) -> None:
        fname = normalize_path(filename)
        data = self.model_dump(exclude_defaults=exclude_defaults)

        if fname.suffix.lower() == ".gz":
            with gzip.open(fname, "wt", encoding="utf-8") as fp:
                json.dump(data, fp, indent=indent)
        else:
            with open(fname, "w") as fp:
                json.dump(data, fp, indent=indent)
