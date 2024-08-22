from __future__ import annotations

from abc import ABC
from typing import List, Tuple, Union

import pydantic.dataclasses as dataclasses
from pydantic import ConfigDict
from typing_extensions import Literal

from .curves import PlotCurveLine
from .patches import (
    PlotPatch,
    PlotPatchEllipse,
    PlotPatchPolygon,
    PlotPatchRectangle,
)

_dcls_config = ConfigDict()


@dataclasses.dataclass(config=_dcls_config)
class LayoutShape:
    s1: float
    s2: float
    y1: float
    y2: float
    name: str = ""
    color: str = "black"
    line_width: float = 1.0
    fill: bool = False

    @property
    def corner_vertices(self):
        return [
            [self.s1, self.s1, self.s2, self.s2],
            [self.y1, self.y2, self.y2, self.y1],
        ]

    @property
    def dimensions(self):
        return (
            self.s2 - self.s1,
            self.y2 - self.y1,
        )

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.s1 + self.s2) / 2,
            (self.y1 + self.y2) / 2,
        )

    @property
    def lines(self):
        return []

    def to_lines(self) -> List[PlotCurveLine]:
        lines = self.lines
        if not lines:
            return []
        return [
            PlotCurveLine(
                [x for x, _ in line],
                [y for _, y in line],
                linewidth=self.line_width,
                color=self.color,
            )
            for line in self.lines
        ]

    def to_patches(self) -> List[PlotPatch]:
        return []

    @property
    def patch_kwargs(self):
        return {
            "linewidth": self.line_width,
            "color": self.color,
            "fill": self.fill,
        }


@dataclasses.dataclass(config=_dcls_config)
class LayoutBox(LayoutShape):
    def to_patches(self) -> List[PlotPatch]:
        width, height = self.dimensions
        return [
            PlotPatchRectangle(
                xy=(self.s1, self.y1),
                width=width,
                height=height,
                **self.patch_kwargs,
            )
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutXBox(LayoutShape):
    @property
    def lines(self):
        return [
            [(self.s1, self.y1), (self.s2, self.y2)],
            [(self.s1, self.y2), (self.s2, self.y1)],
        ]

    def to_patches(self) -> List[PlotPatch]:
        width, height = self.dimensions
        return [
            PlotPatchRectangle(
                xy=(self.s1, self.y1),
                width=width,
                height=height,
                **self.patch_kwargs,
            )
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutLetterX(LayoutShape):
    @property
    def lines(self):
        return [
            [(self.s1, self.y1), (self.s2, self.y2)],
            [(self.s1, self.y2), (self.s2, self.y1)],
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutBowTie(LayoutShape):
    @property
    def lines(self):
        return [
            [
                (self.s1, self.y1),
                (self.s2, self.y2),
                (self.s2, self.y1),
                (self.s1, self.y2),
                (self.s1, self.y1),
            ]
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutRBowTie(LayoutShape):
    @property
    def lines(self):
        return [
            [
                (self.s1, self.y1),
                (self.s2, self.y2),
                (self.s1, self.y2),
                (self.s2, self.y1),
                (self.s1, self.y1),
            ]
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutDiamond(LayoutShape):
    @property
    def lines(self):
        s_mid, _ = self.center
        return [
            [
                (self.s1, 0),
                (s_mid, self.y1),
                (self.s2, 0),
                (s_mid, self.y2),
                (self.s1, 0),
            ]
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutCircle(LayoutShape):
    def to_patches(self) -> List[PlotPatch]:
        s_mid, _ = self.center
        width, height = self.dimensions
        return [
            PlotPatchEllipse(
                xy=(s_mid, 0),
                width=width,
                height=height,
                **self.patch_kwargs,
            )
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutTriangle(LayoutShape):
    orientation: Literal["u", "d", "l", "r"] = "u"

    @property
    def vertices(self):
        s_mid, y_mid = self.center
        if self.orientation == "u":
            return [(self.s1, self.y2), (self.s2, self.y2), (s_mid, self.y1)]
        if self.orientation == "d":
            return [(self.s1, self.y1), (self.s2, self.y1), (s_mid, self.y2)]
        if self.orientation == "l":
            return [(self.s1, y_mid), (self.s2, self.y2), (self.s2, self.y1)]
        if self.orientation == "r":
            return [(self.s1, self.y1), (self.s1, self.y2), (self.s2, y_mid)]
        raise ValueError(f"Unsupported orientation: {self.orientation}")

    def to_patches(self) -> List[PlotPatch]:
        return [PlotPatchPolygon(vertices=self.vertices, **self.patch_kwargs)]


shape_to_class = {
    "box": LayoutBox,
    "xbox": LayoutXBox,
    "x": LayoutLetterX,
    "bowtie": LayoutBowTie,
    "diamond": LayoutDiamond,
    "circle": LayoutCircle,
    "utriangle": LayoutTriangle,
    "dtriangle": LayoutTriangle,
    "ltriangle": LayoutTriangle,
    "rtriangle": LayoutTriangle,
}

AnyNormalLayoutShape = Union[
    LayoutBox,
    LayoutXBox,
    LayoutLetterX,
    LayoutBowTie,
    LayoutDiamond,
    LayoutCircle,
    LayoutTriangle,
]


@dataclasses.dataclass(config=_dcls_config)
class LayoutWrappedShape(ABC):
    s1: float
    s2: float
    y1: float
    y2: float
    s_min: float
    s_max: float
    name: str = ""
    color: str = "black"
    line_width: float = 1.0

    @property
    def lines(self):
        return []

    def to_lines(self) -> List[PlotCurveLine]:
        lines = self.lines
        if not lines:
            return []
        return [
            PlotCurveLine(lx, ly, linewidth=self.line_width, color=self.color)
            for lx, ly in self.lines
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutWrappedBox(LayoutWrappedShape):
    @property
    def lines(self):
        return [
            ([self.s1, self.s_max], [self.y1, self.y1]),
            ([self.s1, self.s_max], [self.y2, self.y2]),
            ([self.s_min, self.s2], [self.y1, self.y1]),
            ([self.s_min, self.s2], [self.y2, self.y2]),
            ([self.s1, self.s1], [self.y1, self.y2]),
            ([self.s2, self.s2], [self.y1, self.y2]),
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutWrappedXBox(LayoutWrappedShape):
    @property
    def lines(self):
        return [
            ([self.s1, self.s_max], [self.y1, self.y1]),
            ([self.s1, self.s_max], [self.y2, self.y2]),
            ([self.s1, self.s_max], [self.y1, 0]),
            ([self.s1, self.s_max], [self.y2, 0]),
            ([self.s_min, self.s2], [self.y1, self.y1]),
            ([self.s_min, self.s2], [self.y2, self.y2]),
            ([self.s_min, self.s2], [0, self.y1]),
            ([self.s_min, self.s2], [0, self.y2]),
            ([self.s1, self.s1], [self.y1, self.y2]),
            ([self.s2, self.s2], [self.y1, self.y2]),
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutWrappedLetterX(LayoutWrappedShape):
    @property
    def lines(self):
        return [
            ([self.s1, self.s_max], [self.y1, 0]),
            ([self.s1, self.s_max], [self.y2, 0]),
            ([self.s_min, self.s2], [0, self.y1]),
            ([self.s_min, self.s2], [0, self.y2]),
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutWrappedBowTie(LayoutWrappedShape):
    @property
    def lines(self):
        return [
            ([self.s1, self.s_max], [self.y1, self.y1]),
            ([self.s1, self.s_max], [self.y2, self.y2]),
            ([self.s1, self.s_max], [self.y1, 0]),
            ([self.s1, self.s_max], [self.y2, 0]),
            ([self.s_min, self.s2], [self.y1, self.y1]),
            ([self.s_min, self.s2], [self.y2, self.y2]),
            ([self.s_min, self.s2], [0, self.y1]),
            ([self.s_min, self.s2], [0, self.y2]),
        ]


@dataclasses.dataclass(config=_dcls_config)
class LayoutWrappedDiamond(LayoutWrappedShape):
    @property
    def lines(self):
        return [
            ([self.s1, self.s_max], [0, self.y1]),
            ([self.s1, self.s_max], [0, self.y2]),
            ([self.s_min, self.s2], [self.y1, 0]),
            ([self.s_min, self.s2], [self.y2, 0]),
        ]


wrapped_shape_to_class = {
    "box": LayoutWrappedBox,
    "xbox": LayoutWrappedXBox,
    "x": LayoutWrappedLetterX,
    "bowtie": LayoutWrappedBowTie,
    "diamond": LayoutWrappedDiamond,
}

AnyWrappedLayoutShape = Union[
    LayoutWrappedBox,
    LayoutWrappedXBox,
    LayoutWrappedLetterX,
    LayoutWrappedBowTie,
    LayoutWrappedDiamond,
]


AnyLayoutShape = Union[AnyNormalLayoutShape, AnyWrappedLayoutShape]
