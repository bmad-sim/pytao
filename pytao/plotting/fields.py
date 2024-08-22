from __future__ import annotations

import typing

import numpy as np
from pydantic import dataclasses

if typing.TYPE_CHECKING:
    from .. import Tao


def get_field_grid(
    tao: Tao,
    ele_id: str,
    radius: float = 0.015,
    num_points: int = 100,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ele_head = tao.ele_head(ele_id)
    s0 = ele_head["s_start"]
    s1 = ele_head["s"]
    s_length = s1 - s0

    S, X = np.meshgrid(
        np.linspace(0, s_length, num_points),
        np.linspace(-radius, radius, num_points),
        indexing="ij",
    )

    @np.vectorize
    def get_By(s: float, x: float) -> float:
        # x, y, s, t in the element frame
        em_field = tao.em_field(ele_id=ele_id, x=x, y=0, z=s, t_or_z=0)
        return em_field["B2"]

    By = get_By(S, X)
    return S + s0, X, By


@dataclasses.dataclass
class ElementField:
    ele_id: str
    s: typing.List[typing.List[float]]
    x: typing.List[typing.List[float]]
    by: typing.List[typing.List[float]]

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele_id: str,
        num_points: int = 100,
        radius: float = 0.015,
    ):
        s, x, by = get_field_grid(tao, ele_id, radius=radius, num_points=num_points)
        return cls(ele_id=ele_id, s=s.tolist(), x=x.tolist(), by=by.tolist())
