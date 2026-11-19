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
    from ..model.ele.comb import Comb
    from ..model.floor import FloorTable
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
    def from_comb(
        cls,
        comb: Comb,
        table: FloorTable,
        *,
        view: str = "zx",
        extent: BeamExtent = "minmax",
        n_sigma: float = 3.0,
        magnification: float = 1.0,
        ix_branch: int | None = None,
    ) -> Self:
        """
        Get the beam envelope from beam statistics saved along a branch.

        Parameters
        ----------
        comb : Comb
            Beam statistics, in local coordinates.
        table : FloorTable
            Floor geometry of the branch the statistics are from.
        view : str, optional
            The floor axes drawn across and up the plot, as in Tao's
            `floor_plan_view`. Default is "zx".
        extent : {"minmax", "sigma"}, optional
            What the envelope shows. "minmax" (the default) is the full extent
            of the beam, from its outermost particle on each side. "sigma" is
            `n_sigma` RMS beam sizes either side of the centroid.
        n_sigma : float, optional
            Half-width of the envelope in beam sigmas, for `extent="sigma"`.
            Default is 3.
        magnification : float, optional
            Scale the beam's offsets from the reference orbit by this factor.
        ix_branch : int, optional
            Branch of `table` the statistics are from. May be omitted when the
            table holds a single branch.

        Raises
        ------
        ValueError
            If `comb` is empty, or `extent` is not recognized.
        """
        if len(comb.s) == 0:
            raise ValueError(
                "No beam statistics are saved. Set `beam%comb_ds_save` and track a beam "
                "(see `Tao.track_beam`)."
            )
        view = view.lower()
        axes = ["xyz".index(view[0]), "xyz".index(view[1])]

        # Offsets of the two envelope edges from the centroid, in local x and y.
        if extent == "minmax":
            edges = {
                "x": (comb.rel_min_x, comb.rel_max_x),
                "y": (comb.rel_min_y, comb.rel_max_y),
            }
            label = "Beam extent"
        elif extent == "sigma":
            sigma = {
                "x": n_sigma * np.sqrt(comb.cov_x__x),
                "y": n_sigma * np.sqrt(comb.cov_y__y),
            }
            edges = {axis: (-size, size) for axis, size in sigma.items()}
            label = f"±{n_sigma:g}σ"
        else:
            raise ValueError(f"Unknown extent {extent!r}; expected 'minmax' or 'sigma'")
        (x_lower, x_upper), (y_lower, y_upper) = (
            edges[axis] if axis in view else (0.0, 0.0) for axis in "xy"
        )

        def to_plot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            floor = table.points_to_floor(
                comb.s, x, y, ix_branch=ix_branch, magnification=magnification
            )
            return floor[:, axes]

        return cls(
            centroid=to_plot(comb.mean_x, comb.mean_y),
            lower=to_plot(comb.mean_x + x_lower, comb.mean_y + y_lower),
            upper=to_plot(comb.mean_x + x_upper, comb.mean_y + y_upper),
            label=label,
        )

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        graph: FloorPlanGraph,
        *,
        comb: Comb | None = None,
        extent: BeamExtent = "minmax",
        n_sigma: float = 3.0,
        magnification: float = 1.0,
        ix_uni: int | None = None,
        ix_branch: int = 0,
    ) -> Self:
        """
        Get the beam envelope for a floor plan graph from Tao.

        Parameters
        ----------
        tao : Tao
        graph : FloorPlanGraph
            The floor plan the envelope will be drawn on.
        comb : Comb, optional
            Beam statistics to use, such as from `Comb.from_tao` or a slice of
            one. By default, only the statistics needed are fetched from Tao.
        extent : {"minmax", "sigma"}, optional
            What the envelope shows; see `from_comb`.
        n_sigma : float, optional
            Half-width of the envelope in beam sigmas, for `extent="sigma"`.
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
        from ..model.ele.comb import Comb
        from ..model.floor import FloorTable

        if graph.info["floor_plan_rotation"]:
            raise NotImplementedError("Rotated floor plans are not supported")

        uni = "" if ix_uni is None else str(ix_uni)
        if comb is None:
            fields = {
                "s": "s",
                "mean_x": "x",
                "mean_y": "y",
                "rel_min_x": "rel_min.1",
                "rel_max_x": "rel_max.1",
                "rel_min_y": "rel_min.3",
                "rel_max_y": "rel_max.3",
                "cov_x__x": "sigma.11",
                "cov_y__y": "sigma.33",
            }
            try:
                comb = Comb(
                    **{
                        name: np.asarray(
                            tao.bunch_comb(who, ix_uni=uni, ix_branch=str(ix_branch))
                        )
                        for name, who in fields.items()
                    }
                )
            except TaoCommandError:
                comb = Comb()

        return cls.from_comb(
            comb,
            FloorTable.from_tao(tao, ix_branch=ix_branch, ix_uni=uni),
            view=graph.info["floor_plan_view"],
            extent=extent,
            n_sigma=n_sigma,
            magnification=magnification,
        )


def find_floor_plan_graph(
    graphs: Sequence[AnyGraph], template: str
) -> tuple[int, FloorPlanGraph]:
    """Index of and the first floor plan graph in `graphs`."""
    for index, graph in enumerate(graphs):
        if isinstance(graph, FloorPlanGraph):
            return index, graph
    raise ValueError(f"Template {template!r} has no floor plan graph")
