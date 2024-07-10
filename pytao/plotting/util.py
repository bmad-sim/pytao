import numpy as np
from typing import Tuple


class NoIntersectionError(Exception): ...


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
