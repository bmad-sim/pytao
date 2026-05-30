from __future__ import annotations

import logging
import re
import typing
from functools import cached_property
from typing import ClassVar, Literal, Pattern

import pydantic
import pydantic.dataclasses as dataclasses
from pydantic import ConfigDict
from pydantic.fields import Field
from typing_extensions import TypedDict

from ..model import Element, Lattice
from ..model.base import TaoBaseModel
from .plot import (
    GraphBase,
    NoLayoutError,
    _clean_pytao_output,
    _point_field,
    get_plot_graph_info,
)
from .types import PlotGraphInfo, PlotRegionInfo, Point

if typing.TYPE_CHECKING:
    from .. import Tao

logger = logging.getLogger(__name__)

_dcls_config = ConfigDict(arbitrary_types_allowed=True)


class BoxData(TypedDict):
    """Pre-bucketed magnet rectangles, one entry per element."""

    left: list[float]
    right: list[float]
    top: list[float]
    bottom: list[float]
    color: list[str]
    name: list[str]
    type: list[str]
    L: list[float]
    s_start: list[float]
    s_end: list[float]


class StemData(TypedDict):
    """Vertical stem segments connecting beamline to off-axis markers."""

    x0: list[float]
    y0: list[float]
    x1: list[float]
    y1: list[float]
    color: list[str]


class MarkerData(TypedDict):
    """One bucket per marker shape — driven into a single scatter call."""

    x: list[float]
    y: list[float]
    color: list[str]
    size: list[float]
    name: list[str]
    type: list[str]


Position = Literal["center", "above", "below"]
# "box" and "stem" are special-cased renderers (rectangles and vertical lines);
# every other value is a Bokeh marker name passed straight through to
# ``fig.scatter(marker=...)``.
Shape = Literal[
    "box",
    "stem",
    "asterisk",
    "circle",
    "circle_cross",
    "circle_dot",
    "circle_x",
    "circle_y",
    "cross",
    "dash",
    "diamond",
    "diamond_cross",
    "diamond_dot",
    "dot",
    "hex",
    "hex_dot",
    "inverted_triangle",
    "plus",
    "square",
    "square_cross",
    "square_dot",
    "square_pin",
    "square_x",
    "star",
    "star_dot",
    "triangle",
    "triangle_dot",
    "triangle_pin",
    "x",
    "y",
]


class ElementStyle(TaoBaseModel, extra="forbid"):
    label: str
    color: str = "#888"
    shape: Shape = "box"
    height: float = 16.0
    size: float = 8.0
    position: Position = "center"
    tier: float | None = None


class ClassifyRule(TaoBaseModel, extra="forbid"):
    """
    Map (Element.key, Element.name) → a style key by regex.

    Either ``style`` is set explicitly, or ``name_pattern`` carries a
    ``(?P<style>...)`` group whose capture is used as the style key.
    """

    key_pattern: str
    name_pattern: str | None = None
    style: str | None = None

    _key_re: Pattern[str] = pydantic.PrivateAttr()
    _name_re: Pattern[str] | None = pydantic.PrivateAttr(default=None)

    def model_post_init(self, _ctx) -> None:
        self._key_re = re.compile(self.key_pattern)
        self._name_re = re.compile(self.name_pattern) if self.name_pattern else None
        if self.style is None and (
            self._name_re is None or "style" not in self._name_re.groupindex
        ):
            raise ValueError(
                "ClassifyRule must set `style` or supply a name_pattern with a "
                "(?P<style>...) named group"
            )

    def match(self, key: str, name: str) -> str | None:
        if not self._key_re.search(key):
            return None
        if self._name_re is None:
            return self.style
        m = self._name_re.search(name)
        if m is None:
            return None
        if "style" in self._name_re.groupindex:
            return m.group("style")
        return self.style


class LayoutSection(TaoBaseModel, extra="forbid"):
    name: str
    s_min: float
    s_max: float | None = None


class ModernLayoutConfig(TaoBaseModel, extra="forbid"):
    """Configuration for the modern lattice layout plot.

    A non-``None`` config on a :class:`GraphManager` replaces the default
    Tao-driven ``lat_layout`` graph with a :class:`ModernLatticeLayoutGraph`.
    """

    styles: dict[str, ElementStyle] = pydantic.Field(default_factory=dict)
    skip_keys: set[str] = pydantic.Field(default_factory=set)
    rules: list[ClassifyRule] = pydantic.Field(default_factory=list)
    sections: list[LayoutSection] | None = None
    section_pattern: str | None = None

    @classmethod
    def default(cls) -> ModernLayoutConfig:
        return cls(
            styles=dict(_DEFAULT_STYLES),
            skip_keys=set(_DEFAULT_SKIP_KEYS),
            rules=[ClassifyRule(**r) for r in _DEFAULT_RULES],
        )

    def with_overrides(self, **patches) -> ModernLayoutConfig:
        """
        Return a deep-merged copy.

        ``styles=`` accepts partial dicts: ``{"SBEND": {"color": "red"}}``
        merges into the existing ``SBEND`` style instead of replacing it.
        """
        data = self.model_dump()
        for key, value in patches.items():
            if key == "styles" and isinstance(value, dict):
                merged = dict(data.get("styles") or {})
                for sk, sv in value.items():
                    if isinstance(sv, dict) and sk in merged:
                        merged[sk] = {**merged[sk], **sv}
                    else:
                        merged[sk] = sv
                data["styles"] = merged
            else:
                data[key] = value
        return type(self).model_validate(data)

    def classify(self, key: str, name: str) -> ElementStyle | None:
        """Return the matched ``ElementStyle``, or None to skip the element.

        Rules are checked before ``skip_keys`` so a rule can promote an
        element whose Bmad key would otherwise be hidden (e.g. a ``MARKER``
        whose name identifies it as a BPM).
        """
        key = key.upper()
        if key in self.styles:
            return self.styles[key]
        for rule in self.rules:
            style_key = rule.match(key, name)
            if style_key is not None and style_key in self.styles:
                return self.styles[style_key]
        if key in self.skip_keys:
            return None
        return None


# Defaults below mirror the script the user provided so out-of-the-box
# behavior matches it.

_DEFAULT_SKIP_KEYS: set[str] = {
    "OVERLAY",
    "PIPE",
    "MARKER",
    "EM_FIELD",
    "PATCH",
    "THICK_MULTIPOLE",
    "GKICKER",
    "E_GUN",
    "BEGINNING_ELE",
}

_DEFAULT_STYLES: dict[str, ElementStyle] = {
    # Magnets — drawn as boxes centered on the beamline.
    "SBEND": ElementStyle(label="Dipole", color="#CC0000", shape="box", height=30),
    "QUADRUPOLE": ElementStyle(label="Quadrupole", color="#0044CC", shape="box", height=24),
    "SEXTUPOLE": ElementStyle(label="Sextupole", color="#CCCC00", shape="box", height=18),
    "OCTUPOLE": ElementStyle(label="Octupole", color="#7B2D8E", shape="box", height=16),
    "LCAVITY": ElementStyle(label="RF Cavity", color="#228B22", shape="box", height=28),
    "SOLENOID": ElementStyle(label="Solenoid", color="#9B30FF", shape="box", height=22),
    "WIGGLER": ElementStyle(label="Wiggler", color="#E8A317", shape="box", height=26),
    "ECOLLIMATOR": ElementStyle(label="Collimator", color="#222222", shape="box", height=20),
    "KICKER": ElementStyle(label="Kicker", color="#888888", shape="box", height=12),
    "HKICKER": ElementStyle(label="H-Corrector", color="#888888", shape="box", height=12),
    "VKICKER": ElementStyle(label="V-Corrector", color="#888888", shape="box", height=12),
    # Diagnostics — markers on tier offsets above (or below) the beamline.
    "BPM": ElementStyle(
        label="BPM", color="#0066FF", shape="diamond", size=8, position="above", tier=22
    ),
    "BSCR": ElementStyle(
        label="Screen", color="#FF6600", shape="triangle", size=10, position="above", tier=42
    ),
    "WRSC": ElementStyle(
        label="Wire Scanner",
        color="#009933",
        shape="circle",
        size=8,
        position="above",
        tier=62,
    ),
    "BLEN": ElementStyle(
        label="Bunch Length",
        color="#CC00CC",
        shape="square",
        size=8,
        position="above",
        tier=82,
    ),
    "SLM": ElementStyle(
        label="Synch. Light",
        color="#FFD700",
        shape="star",
        size=10,
        position="above",
        tier=102,
    ),
    "BSI": ElementStyle(
        label="Beam Stop", color="#333333", shape="cross", size=10, position="above", tier=42
    ),
    "FCUP": ElementStyle(
        label="Faraday Cup", color="#8B4513", shape="circle", size=8, position="above", tier=62
    ),
    "DG": ElementStyle(
        label="Diag. Line",
        color="#FF1493",
        shape="triangle",
        size=10,
        position="above",
        tier=82,
    ),
    "SPEC": ElementStyle(
        label="Spectrometer",
        color="#4169E1",
        shape="square",
        size=10,
        position="above",
        tier=102,
    ),
    "TCAV": ElementStyle(
        label="Transv. Cavity",
        color="#20B2AA",
        shape="diamond",
        size=10,
        position="above",
        tier=102,
    ),
    "BCM": ElementStyle(
        label="Current Mon.",
        color="#00CED1",
        shape="square",
        size=8,
        position="above",
        tier=22,
    ),
    # Stem-only marker (drawn below beamline as a vertical tick).
    "BLM": ElementStyle(
        label="BLM", color="#CC5599", shape="stem", size=6, position="below", tier=-15
    ),
}

# Default classification rules. Direct key matches in `styles` win first;
# these only run if the key isn't itself a style key.
_DEFAULT_RULES: list[dict] = [
    # MONITOR/INSTRUMENT named like "<prefix>_<TYPE>[digits]" → style=<TYPE>.
    {
        "key_pattern": r"^(MONITOR|INSTRUMENT)$",
        "name_pattern": r"^[^_]+_(?P<style>[A-Za-z]+)\d*",
    },
    # Fallback substring matches for the diagnostic types.
    {"key_pattern": r"^(MONITOR|INSTRUMENT)$", "name_pattern": r"BPM", "style": "BPM"},
    {"key_pattern": r"^(MONITOR|INSTRUMENT)$", "name_pattern": r"BSCR", "style": "BSCR"},
    {"key_pattern": r"^(MONITOR|INSTRUMENT)$", "name_pattern": r"WRSC", "style": "WRSC"},
    {"key_pattern": r"^(MONITOR|INSTRUMENT)$", "name_pattern": r"BLEN", "style": "BLEN"},
    {"key_pattern": r"^(MONITOR|INSTRUMENT)$", "name_pattern": r"SLM", "style": "SLM"},
    {"key_pattern": r"^(MONITOR|INSTRUMENT)$", "name_pattern": r"BSI", "style": "BSI"},
    {"key_pattern": r"^(MONITOR|INSTRUMENT)$", "name_pattern": r"FCUP", "style": "FCUP"},
    # Some lattices use the ``MARKER`` key for BPMs. If "BPM" appears
    # anywhere in the name, treat it as a BPM; other markers fall through
    # to the default skip behavior.
    {"key_pattern": r"^MARKER$", "name_pattern": r"BPM", "style": "BPM"},
]


@dataclasses.dataclass(config=_dcls_config)
class ModernElement:
    info: Element
    style: ElementStyle


class LayoutData(TaoBaseModel, extra="forbid"):
    """Plotting-agnostic per-branch lattice layout data."""

    branch: int = 0
    universe: int = 1
    branch_name: str = ""
    s_min: float = 0.0
    s_max: float = 0.0
    elements: list[Element] = pydantic.Field(default_factory=list)
    sections: list[LayoutSection] = pydantic.Field(default_factory=list)

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        *,
        ix_uni: int = 1,
        ix_branch: int = 0,
        sections: list[LayoutSection] | None = None,
        section_pattern: str | None = None,
    ) -> LayoutData:
        lat = Lattice.from_tao_tracking(
            tao,
            ix_uni=str(ix_uni),
            ix_branch=str(ix_branch),
            defaults=False,
        )

        elements = list(lat.elements)

        s_min = float(min(ele.head.s_start for ele in elements)) if elements else 0.0
        s_max = float(max(ele.head.s for ele in elements)) if elements else 0.0

        if sections is not None:
            resolved_sections = list(sections)
        elif section_pattern is not None:
            resolved_sections = _sections_from_pattern(elements, section_pattern)
        else:
            resolved_sections = []

        branch1 = tao.branch1(ix_uni=ix_uni, ix_branch=ix_branch)

        return cls(
            branch=ix_branch,
            universe=ix_uni,
            branch_name=branch1.get("name", f"{ix_uni}@{ix_branch}"),
            s_min=s_min,
            s_max=s_max,
            elements=elements,
            sections=resolved_sections,
        )


def _sections_from_pattern(
    elements: list[Element],
    pattern: str,
) -> list[LayoutSection]:
    """
    Walk elements, emitting a section each time the regex matches a new name.

    If the pattern carries a ``(?P<name>...)`` group, that capture is the
    section name; otherwise the matched substring is used. Adjacent
    duplicates collapse into one section.
    """
    rx = re.compile(pattern)
    out: list[LayoutSection] = []
    last: str | None = None
    for elem in elements:
        m = rx.search(elem.head.name)
        if m is None:
            continue
        name = m.groupdict().get("name") or m.group(0)
        if name == last:
            continue
        if out:
            out[-1].s_max = elem.head.s_start
        out.append(LayoutSection(name=name, s_min=elem.head.s_start))
        last = name
    if out:
        out[-1].s_max = elements[-1].head.s if elements else out[-1].s_min
    return out


@dataclasses.dataclass(config=_dcls_config)
class ModernLatticeLayoutGraph(GraphBase):
    """Replacement for ``LatticeLayoutGraph`` driven by ``ModernLayoutConfig``.

    Used in place of the default ``lat_layout`` graph when a non-``None``
    ``layout_style`` is configured on the active :class:`GraphManager`. The
    Bokeh and matplotlib renderers in ``pytao.plotting.bokeh`` /
    ``pytao.plotting.mpl`` consume the pre-bucketed ``boxes`` / ``stems`` /
    ``markers`` views.
    """

    graph_type: ClassVar[str] = "modern_lat_layout"

    data: LayoutData = Field(default_factory=LayoutData)
    elements: list[ModernElement] = Field(default_factory=list)
    sections: list[LayoutSection] = Field(default_factory=list)
    config: ModernLayoutConfig = Field(default_factory=ModernLayoutConfig.default)
    branch: int = 0
    universe: int = 1
    border_xlim: Point = _point_field

    @property
    def is_s_plot(self) -> bool:
        return True

    @property
    def y_min(self) -> float:
        return self.ylim[0]

    @property
    def y_max(self) -> float:
        return self.ylim[1]

    @cached_property
    def boxes(self) -> BoxData:
        data: BoxData = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": [],
            "color": [],
            "name": [],
            "type": [],
            "L": [],
            "s_start": [],
            "s_end": [],
        }
        for se in self.elements:
            if se.style.shape != "box":
                continue
            h = se.style.height
            head = se.info.head
            data["left"].append(head.s_start)
            data["right"].append(head.s)
            data["top"].append(h / 2.0)
            data["bottom"].append(-h / 2.0)
            data["color"].append(se.style.color)
            data["name"].append(head.name)
            data["type"].append(se.style.label)
            data["L"].append(head.s - head.s_start)
            data["s_start"].append(head.s_start)
            data["s_end"].append(head.s)
        return data

    @cached_property
    def stems(self) -> StemData:
        data: StemData = {"x0": [], "y0": [], "x1": [], "y1": [], "color": []}
        for se in self.elements:
            if se.style.shape == "box":
                continue
            head = se.info.head
            s = (head.s_start + head.s) / 2.0
            tier = _resolve_tier(se.style)
            if se.style.position == "below":
                y0, y1 = tier, 0.0
            else:
                y0, y1 = 0.0, tier
            data["x0"].append(s)
            data["x1"].append(s)
            data["y0"].append(y0)
            data["y1"].append(y1)
            data["color"].append(se.style.color)
        return data

    @cached_property
    def markers(self) -> dict[Shape, MarkerData]:
        out: dict[Shape, MarkerData] = {}
        for se in self.elements:
            shape = se.style.shape
            if shape in ("box", "stem"):
                continue
            head = se.info.head
            s = (head.s_start + head.s) / 2.0
            tier = _resolve_tier(se.style)
            bucket = out.setdefault(
                shape,
                {"x": [], "y": [], "color": [], "size": [], "name": [], "type": []},
            )
            bucket["x"].append(s)
            bucket["y"].append(tier)
            bucket["color"].append(se.style.color)
            # Script doubled viz_config "size" to get bokeh screen-px diameter
            # parity. Renderers can scale further if they want.
            bucket["size"].append(se.style.size * 2)
            bucket["name"].append(head.name)
            bucket["type"].append(se.style.label)
        return out

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        region_name: str = "lat_layout",
        graph_name: str = "g",
        *,
        config: ModernLayoutConfig | None = None,
        info: PlotGraphInfo | None = None,
        template_name: str | None = None,
        template_graph_index: int | None = None,
        **_unused,
    ) -> ModernLatticeLayoutGraph:
        cfg = config if config is not None else ModernLayoutConfig.default()

        if info is None:
            try:
                info = get_plot_graph_info(tao, region_name, graph_name)
            except ValueError:
                raise NoLayoutError(f"No layout named {region_name}.{graph_name}") from None

        region_info = _clean_pytao_output(tao.plot1(region_name), PlotRegionInfo)

        universe = 1 if info["ix_universe"] == -1 else info["ix_universe"]
        ix_branch = info["-1^ix_branch"]
        if ix_branch < 0:
            ix_branch = 0

        data = LayoutData.from_tao(
            tao,
            ix_uni=universe,
            ix_branch=ix_branch,
            sections=cfg.sections,
            section_pattern=cfg.section_pattern,
        )

        elements: list[ModernElement] = []
        for elem in data.elements:
            head = elem.head
            style = cfg.classify(head.key, head.name)
            if style is None:
                continue
            elements.append(ModernElement(info=elem, style=style))

        ylim = _ylim_from_styles(cfg, elements)

        return cls(
            data=data,
            info=info,
            region_info=region_info,
            region_name=region_name,
            graph_name=graph_name,
            template_name=template_name,
            template_graph_index=template_graph_index,
            xlim=(info["x_min"], info["x_max"]),
            ylim=ylim,
            border_xlim=(1.1 * info["x_min"], 1.1 * info["x_max"]),
            universe=universe,
            branch=ix_branch,
            elements=elements,
            sections=list(data.sections),
            config=cfg,
        )


def _resolve_tier(style: ElementStyle) -> float:
    if style.tier is not None:
        return style.tier
    if style.position == "above":
        return 22.0
    if style.position == "below":
        return -15.0
    return 0.0


def _ylim_from_styles(
    cfg: ModernLayoutConfig,
    elements: list[ModernElement],
) -> Point:
    """Derive a vertical range that contains every glyph plus headroom.

    Bottom is a fixed pad below the axis (stems hanging down rarely need
    much room); top is the tallest glyph extent plus headroom for labels.
    """
    y_hi = 1.0
    for se in elements:
        if se.style.shape == "box":
            y_hi = max(y_hi, se.style.height / 2.0)
        else:
            tier = _resolve_tier(se.style)
            y_hi = max(y_hi, tier + se.style.size + 8.0)
    return (-50.0, y_hi + 10.0)
