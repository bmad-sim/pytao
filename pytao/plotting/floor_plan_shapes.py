from __future__ import annotations

import math
from functools import cached_property
from typing import List, Optional, Union

import numpy as np
import pydantic.dataclasses as dataclasses
from pydantic import ConfigDict
from typing_extensions import Literal

from . import util
from .curves import PlotCurveLine
from .patches import (
    PlotPatch,
    PlotPatchArc,
    PlotPatchCircle,
    PlotPatchRectangle,
    PlotPatchSbend,
)

_dcls_config = ConfigDict()


@dataclasses.dataclass(config=_dcls_config)
class Shape:
    x1: float
    x2: float
    y1: float
    y2: float
    off1: float
    off2: float
    angle_start: float
    angle_end: float = 0.0

    rel_angle_start: float = 0.0  # Only for sbend
    rel_angle_end: float = 0.0  # Only for sbend
    line_width: float = 1.0
    color: str = "black"
    name: str = ""

    @property
    def corner_vertices(self):
        px0 = self.x1 + self.off2 * np.sin(self.angle_start)  # x1 + off2 * sin
        py0 = self.y1 - self.off2 * np.cos(self.angle_start)  # y1 - off2 * cos

        px1 = self.x1 - self.off1 * np.sin(self.angle_start)  # x1 - off1 * sin
        py1 = self.y1 + self.off1 * np.cos(self.angle_start)  # y1 + off1 * cos

        px2 = self.x2 - self.off1 * np.sin(self.angle_start)  # x2 - off1 * sin
        py2 = self.y2 + self.off1 * np.cos(self.angle_start)  # y2 + off1 * cos

        px3 = self.x2 + self.off2 * np.sin(self.angle_start)  # x2 + off2 * sin
        py3 = self.y2 - self.off2 * np.cos(self.angle_start)  # y2 - off2 * cos
        return [
            [px0, px1, px2, px3],
            [py0, py1, py2, py3],
        ]

    @property
    def vertices(self):
        return []

    def to_lines(self) -> List[PlotCurveLine]:
        vertices = self.vertices
        if not vertices:
            return []
        vx, vy = self.vertices
        return [PlotCurveLine(vx, vy, linewidth=self.line_width, color=self.color)]

    def to_patches(self) -> List[PlotPatch]:
        return []


@dataclasses.dataclass(config=_dcls_config)
class LineSegment(Shape):
    @property
    def vertices(self):
        return [[self.x1, self.x2], [self.y1, self.y2]]


@dataclasses.dataclass(config=_dcls_config)
class Circle(Shape):
    def to_patches(self) -> List[PlotPatch]:
        circle = PlotPatchCircle(
            xy=(self.x1 + (self.x2 - self.x1) / 2, self.y1 + (self.y2 - self.y1) / 2),
            radius=self.off1,
            linewidth=self.line_width,
            color=self.color,
            fill=False,
        )
        return [circle]


@dataclasses.dataclass(config=_dcls_config)
class KickerLine(LineSegment):
    @property
    def vertices(self):
        return [[self.x1, self.x2], [self.y1, self.y2]]


@dataclasses.dataclass(config=_dcls_config)
class DriftLine(LineSegment):
    pass


@dataclasses.dataclass(config=_dcls_config)
class BowTie(Shape):
    @property
    def vertices(self):
        l1x = [
            self.x1 + self.off2 * np.sin(self.angle_start),
            self.x2 - self.off1 * np.sin(self.angle_start),
        ]
        l1y = [
            self.y1 - self.off2 * np.cos(self.angle_start),
            self.y2 + self.off1 * np.cos(self.angle_start),
        ]
        l2x = [
            self.x1 - self.off1 * np.sin(self.angle_start),
            self.x2 + self.off2 * np.sin(self.angle_start),
        ]
        l2y = [
            self.y1 + self.off1 * np.cos(self.angle_start),
            self.y2 - self.off2 * np.cos(self.angle_start),
        ]
        return [
            [l1x[0], l1x[1], l2x[0], l2x[1], l1x[0]],
            [l1y[0], l1y[1], l2y[0], l2y[1], l1y[0]],
        ]


@dataclasses.dataclass(config=_dcls_config)
class Box(Shape):
    def to_patches(self) -> List[PlotPatch]:
        patch = PlotPatchRectangle(
            xy=(
                self.x1 + self.off2 * np.sin(self.angle_start),
                self.y1 - self.off2 * np.cos(self.angle_start),
            ),
            width=np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2),
            height=self.off1 + self.off2,
            linewidth=self.line_width,
            color=self.color,
            fill=False,
            angle=math.degrees(self.angle_start),
        )
        return [patch]

    @property
    def vertices(self):
        px, py = self.corner_vertices
        return [px + px[:1], py + py[:1]]


@dataclasses.dataclass(config=_dcls_config)
class XBox(Shape):
    @property
    def vertices(self):
        ((px0, px1, px2, px3), (py0, py1, py2, py3)) = self.corner_vertices
        return [
            [px0, px1, px2, px3, px0, px2, px3, px1],
            [py0, py1, py2, py3, py0, py2, py3, py1],
        ]


@dataclasses.dataclass(config=_dcls_config)
class LetterX(Shape):
    def to_lines(self):
        px, py = self.corner_vertices
        return [
            PlotCurveLine(
                [px[0], px[2]],
                [py[0], py[2]],
                linewidth=self.line_width,
                color=self.color,
            ),
            PlotCurveLine(
                [px[1], px[3]],
                [py[1], py[3]],
                linewidth=self.line_width,
                color=self.color,
            ),
        ]


@dataclasses.dataclass(config=_dcls_config)
class Diamond(Shape):
    @property
    def vertices(self):
        l1x1 = self.x1 + (self.x2 - self.x1) / 2 - self.off1 * np.sin(self.angle_start)
        l1y1 = self.y1 + (self.y2 - self.y1) / 2 + self.off1 * np.cos(self.angle_start)
        l2x1 = self.x1 + (self.x2 - self.x1) / 2 + self.off2 * np.sin(self.angle_start)
        l2y1 = self.y1 + (self.y2 - self.y1) / 2 - self.off2 * np.cos(self.angle_start)

        return [
            [self.x1, l1x1, self.x2, l2x1, self.x1],
            [self.y1, l1y1, self.y2, l2y1, self.y1],
        ]


def _sbend_intersection_to_patch(
    intersection: util.Intersection,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    off2: float,
    angle_start: float,
    angle_end: float,
    rel_angle_start: float,
    rel_angle_end: float,
):
    ix, iy = intersection

    sin_start = np.sin(angle_start - rel_angle_start)
    cos_start = np.cos(angle_start - rel_angle_start)
    sin_end = np.sin(angle_end + rel_angle_end)
    cos_end = np.cos(angle_end + rel_angle_end)

    # corners of sbend
    c1 = (x1 - off1 * sin_start, y1 + off1 * cos_start)
    c2 = (x2 - off1 * sin_end, y2 + off1 * cos_end)
    c3 = (x1 + off2 * sin_start, y1 - off2 * cos_start)
    c4 = (x2 + off2 * sin_end, y2 - off2 * cos_end)

    # radii of sbend arc edges
    outer_radius = np.sqrt(
        (x1 - off1 * sin_start - ix) ** 2 + (y1 + off1 * cos_start - iy) ** 2
    )
    inner_radius = np.sqrt(
        (x1 + off2 * sin_start - ix) ** 2 + (y1 - off2 * cos_start - iy) ** 2
    )
    if angle_start <= angle_end:
        outer_radius *= -1
        inner_radius *= -1

    # midpoints of top and bottom arcs in an sbend
    mid_angle = (angle_start + angle_end) / 2

    top = (
        ix - outer_radius * np.sin(mid_angle),
        iy + outer_radius * np.cos(mid_angle),
    )
    bottom = (
        ix - inner_radius * np.sin(mid_angle),
        iy + inner_radius * np.cos(mid_angle),
    )

    # corresponding control points for a quadratic Bezier curve that
    # passes through the corners and arc midpoint
    top_cp = (
        2 * (top[0]) - 0.5 * (c1[0]) - 0.5 * (c2[0]),
        2 * (top[1]) - 0.5 * (c1[1]) - 0.5 * (c2[1]),
    )
    bottom_cp = (
        2 * (bottom[0]) - 0.5 * (c3[0]) - 0.5 * (c4[0]),
        2 * (bottom[1]) - 0.5 * (c3[1]) - 0.5 * (c4[1]),
    )

    return PlotPatchSbend(
        spline1=(c1, top_cp, c2),
        spline2=(c4, bottom_cp, c3),
        facecolor="green",
        alpha=0.5,
    )


def _create_sbend_patches(
    intersection: util.Intersection,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    off2: float,
    line_width: float,
    color: str,
    angle_start: float,
    angle_end: float,
    rel_angle_start: float,
    rel_angle_end: float,
) -> List[PlotPatch]:
    ix, iy = intersection

    a0 = angle_start - rel_angle_start
    a1 = angle_end + rel_angle_end

    # draw sbend edges if bend angle is 0
    angle1 = 360 + math.degrees(
        np.arctan2(
            y1 + off1 * np.cos(a0) - iy,
            x1 - off1 * np.sin(a0) - ix,
        )
    )
    angle2 = 360 + math.degrees(
        np.arctan2(y2 + off1 * np.cos(a1) - iy, x2 - off1 * np.sin(a1) - ix)
    )
    # angles of further curve endpoints relative to center of circle
    angle3 = 360 + math.degrees(
        np.arctan2(
            y1 - off2 * np.cos(a0) - iy,
            x1 + off2 * np.sin(a0) - ix,
        )
    )
    angle4 = 360 + math.degrees(
        np.arctan2(
            y2 - off2 * np.cos(a1) - iy,
            x2 + off2 * np.sin(a1) - ix,
        )
    )
    # angles of closer curve endpoints relative to center of circle

    if abs(angle1 - angle2) < 180:
        a1 = min(angle1, angle2)
        a2 = max(angle1, angle2)
    else:
        a1 = max(angle1, angle2)
        a2 = min(angle1, angle2)

    if abs(angle3 - angle4) < 180:
        a3 = min(angle3, angle4)
        a4 = max(angle3, angle4)
    else:
        a3 = max(angle3, angle4)
        a4 = min(angle3, angle4)
    # determines correct start and end angles for arcs

    rel_sin = np.sin(a0)
    rel_cos = np.cos(a0)
    width1 = 2.0 * np.sqrt((x1 - off1 * rel_sin - ix) ** 2 + (y1 + off1 * rel_cos - iy) ** 2)
    width2 = 2.0 * np.sqrt((x1 + off2 * rel_sin - ix) ** 2 + (y1 - off2 * rel_cos - iy) ** 2)
    patches: List[PlotPatch] = [
        PlotPatchArc(
            xy=(ix, iy),
            width=width1,
            height=width1,
            theta1=a1,
            theta2=a2,
            linewidth=line_width,
            color=color,
        ),
        PlotPatchArc(
            xy=(ix, iy),
            width=width2,
            height=width2,
            theta1=a3,
            theta2=a4,
            linewidth=line_width,
            color=color,
        ),
    ]
    patch = _sbend_intersection_to_patch(
        intersection=intersection,
        x1=x1,
        x2=x2,
        y1=y1,
        y2=y2,
        off1=off1,
        off2=off2,
        angle_start=angle_start,
        angle_end=angle_end,
        rel_angle_start=rel_angle_start,
        rel_angle_end=rel_angle_end,
    )
    patches.append(patch)
    return patches


@dataclasses.dataclass(config=_dcls_config)
class SBend(Shape):
    @property
    def box_lines(self):
        a0 = self.angle_start - self.rel_angle_start
        a1 = self.angle_end + self.rel_angle_end
        return [
            PlotCurveLine(
                [self.x1 - self.off1 * np.sin(a0), self.x1 + self.off2 * np.sin(a0)],
                [self.y1 + self.off1 * np.cos(a0), self.y1 - self.off2 * np.cos(a0)],
                linewidth=self.line_width,
                color=self.color,
            ),
            PlotCurveLine(
                [self.x2 - self.off1 * np.sin(a1), self.x2 + self.off2 * np.sin(a1)],
                [self.y2 + self.off1 * np.cos(a1), self.y2 - self.off2 * np.cos(a1)],
                linewidth=self.line_width,
                color=self.color,
            ),
        ]

    @cached_property
    def intersection(self) -> Optional[util.Intersection]:
        line1 = util.line(
            (
                self.x1 - self.off1 * np.sin(self.angle_start),
                self.y1 + self.off1 * np.cos(self.angle_start),
            ),
            (
                self.x1 + self.off2 * np.sin(self.angle_start),
                self.y1 - self.off2 * np.cos(self.angle_start),
            ),
        )
        line2 = util.line(
            (
                self.x2 - self.off1 * np.sin(self.angle_end),
                self.y2 + self.off1 * np.cos(self.angle_end),
            ),
            (
                self.x2 + self.off2 * np.sin(self.angle_end),
                self.y2 - self.off2 * np.cos(self.angle_end + self.rel_angle_end),
            ),
        )
        try:
            return util.intersect(line1, line2)
        except util.NoIntersectionError:
            return None

    def to_lines(self) -> List[PlotCurveLine]:
        """Lines to draw when there's no intersection."""
        if self.intersection is not None:
            return []

        a0 = self.angle_start - self.rel_angle_start
        a1 = self.angle_end + self.rel_angle_end
        return [
            PlotCurveLine(
                [self.x1 - self.off1 * np.sin(a0), self.x2 - self.off1 * np.sin(a1)],
                [self.y1 + self.off1 * np.cos(a0), self.y2 + self.off1 * np.cos(a1)],
                linewidth=self.line_width,
                color=self.color,
            ),
            PlotCurveLine(
                [self.x1 + self.off2 * np.sin(a0), self.x2 + self.off2 * np.sin(a1)],
                [self.y1 - self.off2 * np.cos(a0), self.y2 - self.off2 * np.cos(a1)],
                linewidth=self.line_width,
                color=self.color,
            ),
        ]

    def to_patches(self) -> List[PlotPatch]:
        if self.intersection is None:
            return []

        return _create_sbend_patches(
            intersection=self.intersection,
            x1=self.x1,
            x2=self.x2,
            y1=self.y1,
            y2=self.y2,
            off1=self.off1,
            off2=self.off2,
            line_width=self.line_width,
            color=self.color,
            angle_start=self.angle_start,
            angle_end=self.angle_end,
            rel_angle_start=self.rel_angle_start,
            rel_angle_end=self.rel_angle_end,
        )


@dataclasses.dataclass(config=_dcls_config)
class Triangle(Shape):
    orientation: Literal["u", "d", "l", "r"] = "u"

    @property
    def vertices(self):
        p0, p1, p2, p3 = tuple(zip(*self.corner_vertices))

        def midpoint(start, end):
            x0, y0 = start
            x1, y1 = end
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0

        if self.orientation == "u":
            points = [p0, p3, midpoint(p1, p2), p0]
        elif self.orientation == "d":
            points = [p1, p2, midpoint(p0, p3), p1]
        elif self.orientation == "l":
            points = [p2, p3, midpoint(p0, p1), p2]
        elif self.orientation == "r":
            points = [p0, p1, midpoint(p2, p3), p0]
        else:
            raise ValueError(f"Unsupported triangle orientation: {self.orientation}")

        return [tuple(x for x, _ in points), tuple(y for _, y in points)]


AnyFloorPlanShape = Union[
    BowTie,
    Box,
    Circle,
    Diamond,
    DriftLine,
    LineSegment,
    KickerLine,
    LetterX,
    SBend,
    XBox,
    Triangle,
]
