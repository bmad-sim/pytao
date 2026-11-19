"""Tracked beam centroid and envelope in the coordinates of a floor plan graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import numpy as np

from ..errors import TaoCommandError
from .plot import AnyGraph, FloorPlanGraph
from .types import Limit

if TYPE_CHECKING:
    from ..tao import Tao


BeamExtent = Literal["minmax", "sigma"]


@dataclass
class FloorBeamEnvelope:
    """
    The tracked beam along a branch, in the axes of a floor plan graph.

    Attributes
    ----------
    centroid : numpy.ndarray
        Shape `(n, 2)` horizontal and vertical plot coordinates of the centroid.
    lower, upper : numpy.ndarray
        Shape `(n, 2)` plot coordinates of the two edges of the envelope.
    label : str
        Description of the envelope, for a legend.
    """

    centroid: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    label: str

    @property
    def outline(self) -> np.ndarray:
        """Shape `(2 n, 2)` polygon enclosing the envelope."""
        return np.concatenate([self.lower, self.upper[::-1]])

    def limits(self, padding: float = 0.1) -> tuple[Limit, Limit]:
        """
        Horizontal and vertical plot limits that just contain the envelope.

        Parameters
        ----------
        padding : float, optional
            Margin on every side, as a fraction of the larger extent.
        """
        low, high = self.outline.min(axis=0), self.outline.max(axis=0)
        margin = padding * max(high - low)
        return (low[0] - margin, high[0] + margin), (low[1] - margin, high[1] + margin)

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        graph: FloorPlanGraph,
        *,
        extent: BeamExtent = "minmax",
        n_sigma: float = 3.0,
        magnification: float = 1.0,
        ix_uni: int | None = None,
        ix_branch: int = 0,
    ) -> Self:
        """
        Get the beam envelope from the beam statistics Tao saved while tracking.

        Parameters
        ----------
        tao : Tao
        graph : FloorPlanGraph
            The floor plan the envelope will be drawn on.
        extent : {"minmax", "sigma"}, optional
            What the envelope shows. "minmax" (the default) is the full extent
            of the beam, from its outermost particle on each side. "sigma" is
            `n_sigma` RMS beam sizes either side of the centroid.
        n_sigma : float, optional
            Half-width of the envelope in beam sigmas, for `extent="sigma"`.
            Default is 3.
        magnification : float, optional
            Scale the beam's offsets from the reference orbit by this factor.
        ix_uni : int, optional
            Universe index. Default is the current universe.
        ix_branch : int, optional
            Branch the beam was tracked through. Default is 0.

        Raises
        ------
        ValueError
            If no beam statistics are saved, or `extent` is not recognized.
        NotImplementedError
            If the floor plan is rotated.
        """
        from ..model.floor import FloorTable

        if graph.info["floor_plan_rotation"]:
            raise NotImplementedError("Rotated floor plans are not supported")
        view = graph.info["floor_plan_view"].lower()
        axes = ["xyz".index(view[0]), "xyz".index(view[1])]

        uni = "" if ix_uni is None else str(ix_uni)

        def comb(who: str) -> np.ndarray:
            return np.asarray(tao.bunch_comb(who, ix_uni=uni, ix_branch=str(ix_branch)))

        try:
            s = comb("s")
        except TaoCommandError as ex:
            raise ValueError(
                "No beam statistics are saved. Set `beam%comb_ds_save` and track a beam "
                "(see `Tao.track_beam`)."
            ) from ex
        x, y = comb("x"), comb("y")
        # Offsets of the two envelope edges from the centroid, in local x and y.
        if extent == "minmax":
            edges = {
                "x": (comb("rel_min.1"), comb("rel_max.1")),
                "y": (comb("rel_min.3"), comb("rel_max.3")),
            }
            label = "Beam extent"
        elif extent == "sigma":
            sigma = {
                "x": n_sigma * np.sqrt(comb("sigma.11")),
                "y": n_sigma * np.sqrt(comb("sigma.33")),
            }
            edges = {axis: (-size, size) for axis, size in sigma.items()}
            label = f"±{n_sigma:g}σ"
        else:
            raise ValueError(f"Unknown extent {extent!r}; expected 'minmax' or 'sigma'")
        (x_lower, x_upper), (y_lower, y_upper) = (
            edges[axis] if axis in view else (0.0, 0.0) for axis in "xy"
        )

        table = FloorTable.from_tao(tao, ix_branch=ix_branch, ix_uni=uni)

        def to_plot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return table.points_to_floor(s, x, y, magnification=magnification)[:, axes]

        return cls(
            centroid=to_plot(x, y),
            lower=to_plot(x + x_lower, y + y_lower),
            upper=to_plot(x + x_upper, y + y_upper),
            label=label,
        )


def find_floor_plan_graph(
    graphs: Sequence[AnyGraph], template: str
) -> tuple[int, FloorPlanGraph]:
    """Index of and the first floor plan graph in `graphs`."""
    for index, graph in enumerate(graphs):
        if isinstance(graph, FloorPlanGraph):
            return index, graph
    raise ValueError(f"Template {template!r} has no floor plan graph")
