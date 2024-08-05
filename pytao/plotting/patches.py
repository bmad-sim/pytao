from __future__ import annotations
import math
from typing import (
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import matplotlib.axes
import matplotlib.cm
import matplotlib.collections
import matplotlib.patches
import matplotlib.path
import matplotlib.text
import numpy as np
import pydantic.dataclasses as dataclasses
import pydantic
from pydantic.fields import Field


from . import pgplot, util
from .types import Point

_dcls_config = pydantic.ConfigDict()


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchBase:
    edgecolor: Optional[str] = None
    facecolor: Optional[str] = None
    color: Optional[str] = None
    linewidth: Optional[float] = None
    linestyle: Optional[str] = None
    antialiased: Optional[bool] = None
    hatch: Optional[str] = None
    fill: bool = True
    capstyle: Optional[str] = None
    joinstyle: Optional[str] = None
    alpha: float = 1.0

    @property
    def _patch_args(self):
        return {
            "edgecolor": self.edgecolor,
            "facecolor": self.facecolor,
            "color": pgplot.mpl_color(self.color or "black"),
            "linewidth": self.linewidth,
            "linestyle": self.linestyle,
            "antialiased": self.antialiased,
            "hatch": self.hatch,
            "fill": self.fill,
            "capstyle": self.capstyle,
            "joinstyle": self.joinstyle,
            "alpha": self.alpha,
        }

    def to_mpl(self):
        raise NotImplementedError(type(self))

    def plot(self, ax: matplotlib.axes.Axes):
        mpl = self.to_mpl()
        ax.add_patch(mpl)
        return mpl


_point_field = Field(default_factory=lambda: (0.0, 0.0))


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchRectangle(PlotPatchBase):
    xy: Point = _point_field
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0
    rotation_point: Union[Literal["xy", "center"], Point] = "xy"

    @property
    def center(self) -> Point:
        return (
            self.xy[0] + self.width / 2,
            self.xy[1] + self.height / 2,
        )

    def to_mpl(self) -> matplotlib.patches.Rectangle:
        return matplotlib.patches.Rectangle(
            xy=self.xy,
            width=self.width,
            height=self.height,
            angle=self.angle,
            rotation_point=self.rotation_point,
            **self._patch_args,
        )


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchArc(PlotPatchBase):
    xy: Point = _point_field
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0
    theta1: float = 0.0
    theta2: float = 360.0
    fill: bool = False  # override

    @classmethod
    def from_building_wall(
        cls,
        mx: float,
        my: float,
        kx: float,
        ky: float,
        k_radii: float,
        color: str,
        linewidth: float,
    ):
        (c0x, c0y), (c1x, c1y) = util.circle_intersection(mx, my, kx, ky, abs(k_radii))
        # radius and endpoints specify 2 possible circle centers for arcs
        mpx = (mx + kx) / 2
        mpy = (my + ky) / 2
        if (
            np.arctan2((my - mpy), (mx - mpx))
            < np.arctan2(c0y, c0x)
            < np.arctan2((my - mpy), (mx - mpx))
            and k_radii > 0
        ):
            center = (c1x, c1y)
        elif (
            np.arctan2((my - mpy), (mx - mpx))
            < np.arctan2(c0y, c0x)
            < np.arctan2((my - mpy), (mx - mpx))
            and k_radii < 0
        ):
            center = (c0x, c0y)
        elif k_radii > 0:
            center = (c0x, c0y)
        else:
            center = (c1x, c1y)

        m_angle = 360 + math.degrees(np.arctan2((my - center[1]), (mx - center[0])))
        k_angle = 360 + math.degrees(np.arctan2((ky - center[1]), (kx - center[0])))
        if k_angle > m_angle:
            t1 = m_angle
            t2 = k_angle
        else:
            t1 = k_angle
            t2 = m_angle

        if abs(k_angle - m_angle) > 180:
            t1, t2 = t2, t1

        return cls(
            xy=center,
            width=k_radii * 2,
            height=k_radii * 2,
            theta1=t1,
            theta2=t2,
            color=color,
            linewidth=linewidth,
        )

    def to_mpl(self) -> matplotlib.patches.Arc:
        return matplotlib.patches.Arc(
            xy=self.xy,
            width=self.width,
            height=self.height,
            angle=self.angle,
            theta1=self.theta1,
            theta2=self.theta2,
            **self._patch_args,
        )


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchCircle(PlotPatchBase):
    xy: Point = _point_field
    radius: float = 0.0

    def to_mpl(self) -> matplotlib.patches.Ellipse:
        return matplotlib.patches.Circle(
            xy=self.xy,
            radius=self.radius,
            **self._patch_args,
        )


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchPolygon(PlotPatchBase):
    vertices: List[Point] = Field(default_factory=list)

    def to_mpl(self) -> matplotlib.patches.Polygon:
        return matplotlib.patches.Polygon(
            xy=self.vertices,
            **self._patch_args,
        )


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchEllipse(PlotPatchBase):
    xy: Point = _point_field
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0

    def to_mpl(self) -> matplotlib.patches.Ellipse:
        return matplotlib.patches.Ellipse(
            xy=self.xy,
            width=self.width,
            height=self.height,
            angle=self.angle,
            **self._patch_args,
        )


@dataclasses.dataclass(config=_dcls_config)
class PlotPatchSbend(PlotPatchBase):
    spline1: Tuple[Point, Point, Point] = Field(default_factory=tuple)
    spline2: Tuple[Point, Point, Point] = Field(default_factory=tuple)

    def to_mpl(self) -> matplotlib.patches.PathPatch:
        codes = [
            matplotlib.path.Path.MOVETO,
            matplotlib.path.Path.CURVE3,
            matplotlib.path.Path.CURVE3,
            matplotlib.path.Path.LINETO,
            matplotlib.path.Path.CURVE3,
            matplotlib.path.Path.CURVE3,
            matplotlib.path.Path.CLOSEPOLY,
        ]
        vertices = [
            self.spline1[0],
            self.spline1[1],
            self.spline1[2],
            self.spline2[0],
            self.spline2[1],
            self.spline2[2],
            self.spline1[0],
        ]
        return matplotlib.patches.PathPatch(
            matplotlib.path.Path(vertices, codes),
            facecolor="green",
            alpha=0.5,
        )


PlotPatch = Union[
    PlotPatchRectangle,
    PlotPatchArc,
    PlotPatchCircle,
    PlotPatchEllipse,
    PlotPatchPolygon,
    PlotPatchSbend,
]
