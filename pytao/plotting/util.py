from __future__ import annotations

import functools
import logging
import sys
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .types import Limit, OptionalLimit

logger = logging.getLogger(__name__)


class NoIntersectionError(Exception):
    pass


def circle_intersection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get 2 intersection points of overlapping circles with equal radii."""
    dx = x2 - x1
    dy = y2 - y1
    d = np.sqrt(dx**2 + dy**2)
    a = d / 2
    h = np.sqrt(r**2 - a**2)
    xm = x1 + dx / 2
    ym = y1 + dy / 2
    xs1 = xm + h * dy / d
    xs2 = xm - h * dy / d
    ys1 = ym - h * dx / d
    ys2 = ym + h * dx / d
    return (xs1, ys1), (xs2, ys2)


Line = Tuple[float, float, float]
Intersection = Tuple[float, float]


def line(p1: Tuple[float, float], p2: Tuple[float, float]) -> Line:
    """returns lines based on given points to be used with intersect"""
    p1x, p1y = p1
    p2x, p2y = p2
    return p1y - p2y, p2x - p1x, -(p1x * p2y - p2x * p1y)


def intersect(L1: Line, L2: Line) -> Intersection:
    """Intersection point of 2 lines from the line function, or false if the lines don't intersect"""
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]

    if D == 0:
        raise NoIntersectionError()

    x = Dx / D
    y = Dy / D
    return x, y


def apply_factor_to_limits(low: float, high: float, factor: float) -> Tuple[float, float]:
    center = (high + low) / 2
    span = factor * (high - low)
    return center - span / 2.0, center + span / 2.0


@functools.cache
def is_jupyter() -> bool:
    """
    Determine if we're in a Jupyter notebook session.

    This works by way of interacting with IPython display and seeing what
    choice it makes regarding reprs.

    Returns
    -------
    bool
    """
    if "IPython" not in sys.modules or "IPython.display" not in sys.modules:
        return False

    from IPython.display import display

    class ReprCheck:
        def _repr_html_(self) -> str:
            self.mode = "jupyter"
            logger.info("Detected Jupyter. Using the notebook graph backend.")
            return "<!-- Detected Jupyter. -->"

        def __repr__(self) -> str:
            self.mode = "console"
            return ""

    check = ReprCheck()
    display(check)
    return check.mode == "jupyter"


@functools.cache
def select_graph_manager_class():
    from .mpl import MatplotlibGraphManager

    if not is_jupyter():
        return MatplotlibGraphManager

    from .bokeh import select_graph_manager_class as select_bokeh_class

    return select_bokeh_class()


def fix_grid_limits(
    limits: Union[OptionalLimit, Sequence[OptionalLimit]],
    num_graphs: int,
) -> List[Optional[Limit]]:
    if not limits:
        return [None] * num_graphs

    if all(isinstance(v, (float, int)) for v in limits):
        res = [limits]
    else:
        res = list(limits or [None])

    if len(res) >= num_graphs:
        return res[:num_graphs]

    return res + [res[-1]] * (num_graphs - len(res))
