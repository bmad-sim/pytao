from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Literal,
)

import pydantic
import pydantic.dataclasses as dataclasses

_dcls_config = pydantic.ConfigDict()


@dataclasses.dataclass(config=_dcls_config)
class PlotCurveLine:
    xs: list[float]
    ys: list[float]
    color: str = "black"
    linestyle: str = "solid"
    linewidth: float = 1.0


@dataclasses.dataclass(config=_dcls_config)
class PlotCurveSymbols:
    xs: list[float]
    ys: list[float]
    color: str
    markerfacecolor: str
    markersize: float
    marker: str
    markeredgewidth: float
    linewidth: float = 0


@dataclasses.dataclass(config=_dcls_config)
class PlotHistogram:
    xs: list[float]
    bins: int | Sequence[float] | str | None
    weights: list[float]
    histtype: Literal["bar", "barstacked", "step", "stepfilled"]
    color: str


class TaoCurveSettings(pydantic.BaseModel, extra="forbid", validate_assignment=True):
    """
    TaoCurveSettings are per-curve settings for Tao's `set curve` command.

    All parameters are `None` by default and will only be applied if
    user-specified.

    Attributes
    ----------
    ele_ref_name : str
        Name or index or the reference element.  An empty string means no
        reference element.
    ix_ele_ref : int
        Same as setting `ele_ref_name`. -1 = No reference element.
    component : str
        Who to plot. Eg: 'meas - design'
        A "data" graph is used to draw lattice parameters such as orbits, or
        Tao data, or variable values such as quadrupole strengths. The
        data values will depend upon where the data comes from. This is
        determined, in part, by the setting of the component parameter in the
        tao_template_graph namelist. The component may be one of:

            "model", for model values. This is the default.
            "design", for design values.
            "base", for base values.
            "meas", for data values.
            "ref", for reference data values.
            "beam_chamber_wall", for beam chamber wall data.

        Additionally, component may be set to plot a linear combination of the
        above. For example:
            "model - design"
        This will plot the difference between the model and design values.
    ix_branch : int
        Branch index.
    ix_bunch : int
        Bunch index.
    ix_universe : int
        Universe index.
    symbol_every : int
        Symbol skip number.
    y_axis_scale_factor : int
        Scaling of y axis
    draw_line : bool
        Draw a line through the data points?
    draw_symbols : bool
        Draw a symbol at the data points?
    draw_symbol_index : bool
        Draw the symbol index number curve%ix_symb?
    """

    ele_ref_name: str | None = pydantic.Field(
        default=None,
        max_length=40,
        description="Reference element.",
    )
    ix_ele_ref: int | None = pydantic.Field(
        default=None,
        description="Index in lattice of reference element.",
    )
    component: str | None = pydantic.Field(
        default=None,
        max_length=60,
        description="Who to plot. Eg: 'meas - design'",
    )
    ix_branch: int | None = pydantic.Field(
        default=None,
    )
    ix_bunch: int | None = pydantic.Field(
        default=None,
        description="Bunch to plot.",
    )
    ix_universe: int | None = pydantic.Field(
        default=None,
        description="Universe where data is. -1 => use s%global%default_universe",
    )
    symbol_every: int | None = pydantic.Field(
        default=None,
        description="Symbol every how many points.",
    )
    y_axis_scale_factor: float | None = pydantic.Field(
        default=None,
        description="y-axis conversion from internal to plotting units.",
    )
    draw_line: bool | None = pydantic.Field(
        default=None,
        description="Draw a line through the data points?",
    )
    draw_symbols: bool | None = pydantic.Field(
        default=None,
        description="Draw a symbol at the data points?",
    )
    draw_symbol_index: bool | None = pydantic.Field(
        default=None,
        description="Draw the symbol index number curve%ix_symb?",
    )

    def get_commands(
        self,
        region_name: str,
        graph_name: str,
        curve_index: int,
    ) -> list[str]:
        return [
            f"set curve {region_name}.{graph_name}.c{curve_index} {key} = {value}"
            for key, value in self.model_dump().items()
            if value is not None
        ]


CurveIndexToCurve = dict[int, TaoCurveSettings]
