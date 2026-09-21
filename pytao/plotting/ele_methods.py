from __future__ import annotations

import itertools
import typing
import zlib
from collections.abc import Sequence
from typing import ClassVar

from pydantic import dataclasses

if typing.TYPE_CHECKING:
    from .. import Tao
    from ..model.ele import Which


# OFF_COLOR = "#00000000"  # transparent
OFF_COLOR = "black"
GARBAGE_COLOR = "red"

KNOWN_METHOD_COLORS: dict[str, str] = {
    "Off": OFF_COLOR,
    "GARBAGE!": GARBAGE_COLOR,
    # tracking_method_name
    "Bmad_Standard": "steelblue",
    "Symp_Lie_PTC": "darkorange",
    "Runge_Kutta": "forestgreen",
    "Linear": "mediumpurple",
    "Time_Runge_Kutta": "lightgreen",
    "Custom": "sienna",
    "Taylor": "orchid",
    "Fixed_Step_Runge_Kutta": "darkolivegreen",
    "Symp_Lie_Bmad": "darkturquoise",
    "Fixed_Step_Time_Runge_kutta": "paleturquoise",
    "MAD": "goldenrod",
    # spin_tracking_method_name
    "Transverse_Kick": "sandybrown",
    "Tracking": "thistle",
    "Magnus": "darkmagenta",
    "Sprint": "lightsteelblue",
    # mat6_calc_method_name
    "Auto": "gray",
    # csr_method_name
    "1_Dim": "yellowgreen",
    "Steady_State_3D": "darkslateblue",
    # space_charge_method_name
    "Slice": "green",
    "FFT_3D": "royalblue",
    "Cathode_FFT_3D": "violet",
    # field_calc_name
    "FieldMap": "mediumseagreen",
    "Planar_Model": "rosybrown",
    "Refer_to_Lords.": "silver",
    "No_Field": "lightpink",
    "Helical_Model": "mediumvioletred",
    "Soft_edge": "darkkhaki",
    # ptc_integration_type_name
    "Drift_Kick": "palegreen",
    "Matrix_Kick": "cornflowerblue",
    "Ripken_Kick": "darkgoldenrod",
}

METHOD_COLORS: dict[str, str] = {
    name.lower(): color for name, color in KNOWN_METHOD_COLORS.items()
}

CATEGORY_PALETTE: tuple[str, ...] = tuple(
    color for color in KNOWN_METHOD_COLORS.values() if color not in (OFF_COLOR, GARBAGE_COLOR)
)


def is_garbage_value(value: str) -> bool:
    """A "GARBAGE!" method value indicates an issue with bmad itself."""
    return value.lower() == "garbage!"


def color_for_value(value: str) -> str:
    """Get the display color for a categorical method value."""
    key = value.lower()
    try:
        return METHOD_COLORS[key]
    except KeyError:
        # Fallback - based on Mayes' original; perhaps a bit excessive
        return CATEGORY_PALETTE[zlib.crc32(key.encode()) % len(CATEGORY_PALETTE)]


def _is_active(value: str | None) -> bool:
    return value is not None and value.lower() != "off"


@dataclasses.dataclass
class ElementMethodsPlotData:
    """
    Per-element method settings gathered for `plot_ele_methods`.

    All per-element lists share the same length and ordering (by increasing
    longitudinal position).
    """

    METHOD_COLUMNS: ClassVar[tuple[str, ...]] = (
        "tracking_method",
        "mat6_calc_method",
        "spin_tracking_method",
        "csr_method",
        "space_charge_method",
        "field_calc",
        "ptc_integration_type",
    )

    ix_eles: list[int]
    names: list[str]
    s_start: list[float]
    s_end: list[float]
    methods: dict[str, list[str | None]]
    csr_ds_step: list[float | None]
    space_charge_mesh_size: list[int]
    csr3d_mesh_size: list[int]
    n_bin: int
    ds_track_step: float

    def validate_columns(self, columns: Sequence[str] | None) -> list[str]:
        """Resolve and validate the method columns to plot."""
        if columns is None:
            columns = list(self.methods)
        else:
            columns = list(columns)
            missing = [col for col in columns if col not in self.methods]
            if missing:
                raise ValueError(
                    f"No data for method column(s) {missing}; "
                    f"available columns: {list(self.methods)}"
                )
        if not self.names or not columns:
            raise ValueError("No elements with method data to plot")
        return columns

    def value_runs(self, column: str) -> list[tuple[int, int, str]]:
        """
        Contiguous runs of equal, non-None method values for a column.

        Merges s-adjacent elements with the same value so that each run can be
        drawn as a single block, as `(first_index, last_index, value)` tuples.
        """
        runs: list[tuple[int, int, str]] = []
        for idx, value in enumerate(self.methods[column]):
            if value is None:
                continue
            if runs:
                first, last, run_value = runs[-1]
                run_end = self.s_end[last]
                contiguous = self.s_start[idx] <= run_end + 1e-9 * max(1.0, abs(run_end))
                if value == run_value and contiguous:
                    runs[-1] = (first, idx, value)
                    continue
            runs.append((idx, idx, value))
        return runs

    def value_transitions(self, column: str) -> list[tuple[int, str, str]]:
        """
        Method value changes between adjacent elements for a column.

        Returns `(index, before, after)` tuples; the transition sits at the
        boundary between elements `index` and `index + 1`, i.e. `s_end[index]`.
        """
        values = self.methods[column]
        return [
            (idx, before, after)
            for idx, (before, after) in enumerate(itertools.pairwise(values))
            if before is not None and after is not None and before != after
        ]

    def csr_ds_step_segments(self) -> list[tuple[float, float, float, str | None]]:
        """
        `csr_ds_step` segments as `(s_start, s_end, step, csr_method)` tuples,
        skipping elements without a value.
        """
        csr_methods = self.methods.get("csr_method", [None] * len(self.names))
        return [
            (self.s_start[idx], self.s_end[idx], step, csr_methods[idx])
            for idx, step in enumerate(self.csr_ds_step)
            if step is not None
        ]

    @property
    def csr_on(self) -> bool:
        """CSR is active (not "Off") for at least one element."""
        return any(_is_active(value) for value in self.methods.get("csr_method", []))

    @property
    def csr_3d_on(self) -> bool:
        """3D CSR (Steady_State_3D) is active for at least one element."""
        return any(
            value is not None and value.lower() == "steady_state_3d"
            for value in self.methods.get("csr_method", [])
        )

    @property
    def settings_summary(self) -> dict[str, list[str]]:
        """
        Formatted global space charge/CSR settings, keyed by the method column
        they relate to.
        """
        sc_mesh = "×".join(str(v) for v in self.space_charge_mesh_size)
        csr_mesh = "×".join(str(v) for v in self.csr3d_mesh_size)
        return {
            "space_charge_method": [f"space_charge_mesh_size: {sc_mesh}"],
            "csr_method": [
                f"csr3d_mesh_size: {csr_mesh}",
                f"n_bin: {self.n_bin}",
                f"ds_track_step: {self.ds_track_step}",
            ],
        }

    @property
    def space_charge_on(self) -> bool:
        """Space charge is active (not "Off") for at least one element."""
        return any(_is_active(value) for value in self.methods.get("space_charge_method", []))

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele_id: str = "*",
        *,
        ix_uni: str | int | None = None,
        ix_branch: str = "0",
        which: Which = "model",
        include_zero_length: bool = False,
    ) -> ElementMethodsPlotData:
        """
        Gather element method settings from Tao.

        Parameters
        ----------
        tao : Tao
        ele_id : str, default="*"
            Element match string, using the same syntax as `Tao.eles`
            (e.g., `"*"`, `"1:20"`, `"quad::*"`).
            Only tracking elements are considered; lord elements are excluded.
        ix_uni : str, default=""
            Universe index.  Defaults to the default universe.
        ix_branch : str, default="0"
            Branch index.  Defaults to branch 0.
        which : "model", "base", or "design", default="model"
        include_zero_length : bool, default=False
            Include zero-length elements.
        """

        conf = tao.get_config()
        sc_com = conf.space_charge_com

        # Lord elements overlap their slaves longitudinally, which would draw
        # conflicting blocks on top of each other.
        elements = tao.eles(
            ele_id,
            ix_uni=ix_uni,
            ix_branch=ix_branch,
            which=which,
            track_only=True,
            defaults=False,
            attrs=True,
            methods=True,
        )

        ix_eles: list[int] = []
        names: list[str] = []
        s_start: list[float] = []
        s_end: list[float] = []
        csr_ds_step: list[float | None] = []
        methods: dict[str, list[str | None]] = {col: [] for col in cls.METHOD_COLUMNS}

        for ele in sorted(elements, key=lambda ele: ele.head.s):
            attrs = ele.attribs
            length = attrs.get("L", 0.0)
            if length <= 0.0 and not include_zero_length:
                continue

            ix_eles.append(ele.head.ix_ele)
            names.append(ele.head.name)
            s_start.append(ele.head.s_start)
            s_end.append(ele.head.s)

            # csr_ds_step 0 -> use default from sc com
            csr_ds_step.append(float(attrs.get("csr_ds_step", 0.0)) or sc_com.ds_track_step)

            for col in cls.METHOD_COLUMNS:
                methods[col].append(
                    getattr(ele.methods, col) if ele.methods is not None else None
                )

        methods = {
            col: values
            for col, values in methods.items()
            if any(value is not None for value in values)
        }

        if not names or not methods:
            raise ValueError(f"No elements with method data matched: {ele_id!r}")

        return cls(
            ix_eles=ix_eles,
            names=names,
            s_start=s_start,
            s_end=s_end,
            methods=methods,
            csr_ds_step=csr_ds_step,
            ds_track_step=sc_com.ds_track_step,
            space_charge_mesh_size=list(sc_com.space_charge_mesh_size),
            csr3d_mesh_size=list(sc_com.csr3d_mesh_size),
            n_bin=sc_com.n_bin,
        )
