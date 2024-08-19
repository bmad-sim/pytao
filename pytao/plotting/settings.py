from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union

import pydantic
from typing_extensions import Literal

from .types import Limit

tao_colors = frozenset(
    {
        "Not_Set",
        "White",
        "Black",
        "Red",
        "Green",
        "Blue",
        "Cyan",
        "Magenta",
        "Yellow",
        "Orange",
        "Yellow_Green",
        "Light_Green",
        "Navy_Blue",
        "Purple",
        "Reddish_Purple",
        "Dark_Grey",
        "Light_Grey",
        "Transparent",
    }
)


@pydantic.dataclasses.dataclass
class QuickPlotPoint:
    """
    Tao QuickPlot Point.

    Attributes
    ----------
    x : float
    y : float
    units : str, optional
    """

    x: float
    y: float
    units: Optional[str] = pydantic.Field(max_length=16, default=None)

    def get_commands(
        self,
        region_name: str,
        graph_name: str,
        parent_name: str,
    ) -> List[str]:
        """
        Get command strings to apply these settings with Tao.

        Parameters
        ----------
        region_name : str
            Region name.
        graph_name : str
            Graph name.
        parent_name : str
            Parent item name.

        Returns
        -------
        list of str
            Commands to send to Tao to apply these settings.
        """
        return [
            f"set graph {region_name}.{graph_name} {parent_name}%{key} = {value}"
            for key, value in asdict(self).items()
            if value is not None
        ]


QuickPlotPointTuple = Tuple[float, float, str]


@pydantic.dataclasses.dataclass
class QuickPlotRectangle:
    """
    Tao QuickPlot Rectangle.

    Attributes
    ----------
    x1 : float
    x2 : float
    y1 : float
    y2 : float
    units : str, optional
    """

    x1: float
    x2: float
    y1: float
    y2: float
    units: Optional[str] = pydantic.Field(default=None, max_length=16)

    def get_commands(
        self,
        region_name: str,
        graph_name: str,
        parent_name: str,
    ) -> List[str]:
        """
        Get command strings to apply these settings with Tao.

        Parameters
        ----------
        region_name : str
            Region name.
        graph_name : str
            Graph name.
        parent_name : str
            Parent item name.

        Returns
        -------
        list of str
            Commands to send to Tao to apply these settings.
        """
        return [
            f"set graph {region_name}.{graph_name} {parent_name}%{key} = {value}"
            for key, value in asdict(self).items()
            if value is not None
        ]


QuickPlotRectangleTuple = Tuple[float, float, float, float, str]


class TaoAxisSettings(pydantic.BaseModel, extra="forbid", validate_assignment=True):
    """
    Tao per-axis (x, x2, y, or y2) settings in a graph.

    Intended for use with:

    `tao.plot(..., settings=TaoGraphSettings(y=TaoAxisSettings(...)))`.

    All attributes may be `None`. A value of `None` indicates that the setting
    should not be configured in Tao.

    Not all parameters are useful for PyTao plotting.  This class intends to
    support Tao's internal plotting mechanism as well for users who prefer
    to use it instead.

    Attributes
    ----------
    bounds : One of {"zero_at_end", "zero_symmetric", "general", "exact"}
        Axis bounds.
    min : float
        Left or bottom axis number range.
    max : float
        Right or top axis number range.
    number_offset : float
        Offset from the axis line in inches.
    label_offset : float
        Offset from numbers in inches.
    major_tick_len : float
        Major tick length in inches.
    minor_tick_len : float
        Minor tick length in inches.
    label_color : str
        Color of the label string.
    major_div : int
        Number of major divisions.
    major_div_nominal : int
        Major divisions nominal value.
    minor_div : int
        Number of minor divisions, where 0 is automatic.
    minor_div_max : int
        Maximum minor divisions, if `minor_div` is set to automatic (0).
    places : int
        Number of digits to print.
    tick_side : -1, 0, or 1
        * 1: draw ticks to the inside
        * 0: draw ticks both inside and outside
        * -1: draw ticks to the outside
    number_side : -1 or 1
        * 1: draw numbers to the inside
        * -1: draw numbers to the outside
    label : str
        Axis label string.
    type : "log" or "linear"
        Axis type.
    draw_label : bool
        Draw the label string.
    draw_numbers : bool
        Draw the numbers.
    """

    bounds: Optional[Literal["zero_at_end", "zero_symmetric", "general", "exact"]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    number_offset: Optional[float] = None
    label_offset: Optional[float] = None
    major_tick_len: Optional[float] = None
    minor_tick_len: Optional[float] = None
    label_color: Optional[str] = pydantic.Field(max_length=16, default=None)
    major_div: Optional[int] = None
    major_div_nominal: Optional[int] = None
    minor_div: Optional[int] = None
    minor_div_max: Optional[int] = None
    places: Optional[int] = None
    tick_side: Optional[Literal[-1, 0, 1]] = None
    number_side: Optional[Literal[-1, 1]] = None
    label: Optional[str] = pydantic.Field(max_length=80, default=None)
    type: Optional[Literal["log", "linear"]] = None
    draw_label: Optional[bool] = None
    draw_numbers: Optional[bool] = None

    scale: Optional[Tuple[float, float]] = None
    scale_gang: Optional[bool] = None

    def get_commands(
        self,
        region_name: str,
        axis_name: str,
    ) -> List[str]:
        """
        Get command strings to apply these settings with Tao.

        Parameters
        ----------
        region_name : str
            Region name.
        axis_name : str
            Axis name.

        Returns
        -------
        list of str
            Commands to send to Tao to apply these settings.
        """
        items = {key: value for key, value in self.model_dump().items() if value is not None}
        scale = items.pop("scale", None)
        scale_gang = items.pop("scale_gang", None)

        commands = []
        if scale is not None:
            scale_low, scale_high = scale
            scale_cmd = {
                "x": "x_scale",
                "x2": "x_scale",
                "y": "scale -y",
                "y2": "scale -y2",
            }[axis_name]
            if scale_gang:
                scale_cmd = f"{scale_cmd} -gang"
            elif scale_gang is False:  # note: may be None
                scale_cmd = f"{scale_cmd} -nogang"
            commands.append(f"{scale_cmd} {region_name} {scale_low} {scale_high}")

        return commands + [
            f"set graph {region_name} {axis_name}%{key} = {value}"
            for key, value in items.items()
        ]


class TaoFloorPlanSettings(pydantic.BaseModel, extra="forbid", validate_assignment=True):
    """
    Tao graph settings specifically for floor plans.

    Intended for use with:

    `tao.plot(..., settings=TaoGraphSettings(floor_plan=TaoFloorPlanSettings(...)))`.

    All attributes may be `None`. A value of `None` indicates that the setting
    should not be configured in Tao.

    Not all parameters are useful for PyTao plotting.  This class intends to
    support Tao's internal plotting mechanism as well for users who prefer
    to use it instead.

    Attributes
    ----------
    correct_distortion : bool
        Correct distortion. By default, the horizontal or vertical margins of
        the graph will be increased so that the horizontal scale (meters per
        plotting inch) is equal to the vertical scale. If `correct_distortion`
        is set to False, this scaling will not be done.
    size_is_absolute : bool
        Shape sizes scaled to absolute dimensions.
        The size_is_absolute logical is combined with the <size> setting for a
        shape to determine the size transverse to the center line curve of the
        drawn shape. If size_is_absolute is False (the default), <size> is
        taken to be the size of the shape in points (72 points is approximately
        1 inch). If size_is_absolute is True, <size> is taken to be the size in
        meters. That is, if size_is_absolute is False, zooming in or out will
        not affect the size of an element shape while if size_is_absolute is
        True, the size of an element will scale when zooming.
    draw_only_first_pass : bool
        Suppresses drawing of multipass_slave lattice elements that are
        associated with the second and higher passes. Setting to True is only
        useful in some extreme circumstances where the plotting of additional
        passes leads to large pdf/ps file sizes.
    flip_label_side : bool
        Draw element label on other side of element.
    rotation : float
        Rotation of floor plan plot: 1.0 -> 360 deg.
        An overall rotation of the floor plan can be controlled by setting rotation
        parameter. A setting of 1.0 corresponds to 360◦. Positive values correspond
        to counter-clockwise rotations. Alternatively, the global coordinates at
        the start of the lattice can be defined in the lattice file and this can
        rotate the floor plan. Unless there is an offset specified in the lattice
        file, a lattice will start at (x, y) = (0, 0). Assuming that the machine
        lies in the horizontal plane with no negative bends, the reference orbit
        will start out pointing in the negative x direction and will circle
        clockwise in the (x, y) plane.
    orbit_scale : float
        Scale for the orbit.  If 0 (the default), no orbit will be drawn.
    orbit_color : str
        Orbit color.
    orbit_lattice : One of {"model", "design", "base"}
        Orbit lattice.
    orbit_pattern : str
        Orbit pattern.
    orbit_width : int
        Orbit width.
    view : One of {"xy", "xz", "yx", "yz", "zx", "zy"}
    """

    correct_distortion: Optional[bool] = None
    size_is_absolute: Optional[bool] = None
    draw_only_first_pass: Optional[bool] = None
    flip_label_side: Optional[bool] = None
    rotation: Optional[float] = None
    orbit_scale: Optional[float] = None
    orbit_color: Optional[str] = None
    orbit_lattice: Optional[Literal["model", "design", "base"]] = None
    orbit_pattern: Optional[str] = None
    orbit_width: Optional[int] = None
    view: Optional[Literal["xy", "xz", "yx", "yz", "zx", "zy"]] = None

    def get_commands(
        self,
        region_name: str,
        graph_name: str,
    ) -> List[str]:
        """
        Get command strings to apply these settings with Tao.

        Parameters
        ----------
        region_name : str
            Region name.
        graph_name : str
            Graph name.

        Returns
        -------
        list of str
            Commands to send to Tao to apply these settings.
        """
        return [
            f"set graph {region_name} floor_plan%{key} = {value}"
            for key, value in self.model_dump().items()
            if value is not None
        ]


class TaoGraphSettings(pydantic.BaseModel, extra="forbid", validate_assignment=True):
    """
    Per-graph settings for Tao.

    Intended for use with `tao.plot(..., settings=TaoGraphSettings())`.

    All attributes may be `None`. A value of `None` indicates that the setting
    should not be configured in Tao.

    Not all parameters are useful for PyTao plotting.  This class intends to
    support Tao's internal plotting mechanism as well for users who prefer
    to use it instead.

    Attributes
    ----------
    text_legend : Dict[int, str]
        Dictionary of text legend index to legend string.
        The text legend is a legend that can be setup by either the user or by
        Tao itself. Tao uses the text legend in conjunction with phase space
        plotting or histogram displays. The text legend is distinct from the
        curve legend.
    allow_wrap_around : bool
        If `plot%x_axis_type` is set to "s", and if the plotted data is from a
        lattice branch with a closed geometry, and if `graph%x%min` is negative,
        then the `graph%allow_wrap_around` parameter sets if the curves contained
        in the graph are “wrapped around” the beginning of the lattice so that
        the curves are not cut off at s = 0. The Tao default is True.
    box : Dict[int, int]
        The `graph%box parameter` sets the layout of the box which the graph is
        placed. The Tao default is 1,1,1,1 which scales a graph to cover the
        entire region the plot is placed in.
    commands : List[str]
        Custom commands to be sent to Tao for this graph.
        Each string will be formatted with the following keys:
        * `settings` - the `TaoGraphSettings` object itself
        * `region` - the region name (e.g., `r12`)
        * `graph_name` - the graph name (e.g., `g`)
        * `graph_type` - the graph type (e.g., `lat_layout`)
        * `graph` - the full graph name (e.g., `r12.g`)
    component : str
        Who to plot - applied to all curves. For example: `'meas - design'`
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
    clip : bool
        Clip curves at the boundary.
    curve_legend_origin : tuple[float, float, str] or QuickPlotPoint
        The curve legend displays which curves are associated with which of the
        plotted lines and symbols. This defines where the upper left hand
        corner of the legend is.
    draw_axes : bool
        Draw the graph axes.
    draw_title : bool
        Draw the graph title.
    draw_curve_legend : bool
        Draw the curve legend.
    draw_grid : bool
        Draw the graph grid.
    draw_only_good_user_data_or_vars : bool
        When plotting Tao data or variables, if
        `draw_only_good_user_data_or_vars` is set to True (the default), symbol
        points of curves in the graph associated with data or variables whose
        `good_user` parameter is set to False will be ignored. That is, data and
        variables that will not be used in an optimization will be ignored. If
        `draw_only_good_user_data_or_vars` is set to False, data or variables
        that have a valid value will be plotted.
    floor_plan : TaoFloorPlanSettings
        Settings only for floor plan graphs.
    ix_universe : int
        The default universe for curves of the graph.
    ix_branch : int
        The default branch for curves of the graph.
    margin : tuple[float, float, float, float, str] or QuickPlotRectangle
        Margin between the graph and the box: (x1, x2, y1, y2, units)
    scale_margin : Union[QuickPlotRectangle, QuickPlotRectangleTuple]
        (x1, x2, y1, y2, units)
        Used to set the minimum space between what is being drawn and the edges
        of the graph when a `scale`, `x_scale`, or an `xy_scale` command is
        issued. Normally this is zero but is useful for floor plan drawings.
    symbol_size_scale : float
        Symbol size scale.
    text_legend_origin : tuple[float, float, float, float, str] or QuickPlotRectangle
        (x1, x2, y1, y2, units)
        Text legend origin.
    title : str
        The `title` component is the string printed just above the graph
        box. The full string will also include information about what is being
        plotted and the horizontal axis type. To fully suppress the title leave
        it blank. Note: A graph also has a `title_suffix` which Tao uses to
        hold the string which is printed to the right of the `graph%title`.
        This string contains information like what curve%component is being
        plotted. The `graph%title_suffix` cannot be set by the user.
    type : str
        The type of graph. Tao knows about the following types:

        * `"data"`

        Lattice parameters, data and/or variable plots (default)

        With `type` set to `"data"`, data such as orbits and/or variable
        values such as quadrupole strengths are plotted. Here “data” can be
        data from a defined data structure or computed directly from the
        lattice, beam tracking, etc. A "data" graph type will contain a number
        of curves and multiple data and variable curves can be drawn in one
        graph.

        * `"floor_plan"`

        With `type` set to `"floor_plan"`, the two dimensional layout of the
        machine is drawn.

        * `"dynamic_aperture"`

        Dynamic aperture plot.

        * `"histogram"`

        With `type` set to `"histogram"`, such things such as beam
        densities can be his- togrammed.

        * `"lat_layout"`

        With `type` set to `"lat_layout"`, the elements of the
        lattice are symbolical drawn in a one dimensional line as a function of
        the longitudinal distance along the machine centerline.

        * `"phase_space"`

        With `type` set to `"phase_space"`, phase space plots are
        produced.

        * `"key_table"` - unsupported by PyTao plotting.

        With `type` set to `"key_table"`, the key bindings for use in single
        mode are displayed. Note: The "key_table" graph type does not have any
        associated curves.

    x : TaoAxisSettings
        Primary x-axis settings.
    x2 : TaoAxisSettings
        Secondary x-axis settings.
    y : TaoAxisSettings
        Primary y-axis settings.
    y2 : TaoAxisSettings
        Secondary y-axis settings.
    y2_mirrors_y : bool
        Keep y2 min/max the same as y-axis.
    x_axis_scale_factor : float
        Sets the horizontal x-axis scale factor. For a given datum value, the
        plotted value will be: `x(plotted) = scale_factor * x(datum)`
        The default value is 1.
        For example, a %x_axis_scale_factor of 1000 will draw a 1.0 mm phase
        space z value at the 1.0 mark on the horizontal scale. That is, the
        horizontal scale will be in millimeters. Also see
        `curve(N)%y_axis_scale_factor`.
    """

    text_legend: Dict[int, str] = pydantic.Field(default_factory=dict)
    allow_wrap_around: Optional[bool] = None
    box: Dict[int, int] = pydantic.Field(
        default_factory=dict,
        description="Defines which box the plot is put in.",
    )
    commands: Optional[List[str]] = None
    component: Optional[str] = None
    clip: Optional[bool] = None
    curve_legend_origin: Optional[Union[QuickPlotPoint, QuickPlotPointTuple]] = None
    draw_axes: Optional[bool] = None
    draw_title: Optional[bool] = None
    draw_curve_legend: Optional[bool] = None
    draw_grid: Optional[bool] = None
    draw_only_good_user_data_or_vars: Optional[bool] = None
    floor_plan: Optional[TaoFloorPlanSettings] = None
    ix_universe: Optional[int] = None
    ix_branch: Optional[int] = None
    margin: Optional[Union[QuickPlotRectangle, QuickPlotRectangleTuple]] = None
    name: Optional[str] = None
    scale_margin: Optional[Union[QuickPlotRectangle, QuickPlotRectangleTuple]] = None
    symbol_size_scale: Optional[float] = None
    text_legend_origin: Optional[Union[QuickPlotRectangle, QuickPlotRectangleTuple]] = None
    title: Optional[str] = None
    type: Optional[str] = None
    x: Optional[TaoAxisSettings] = None
    x2: Optional[TaoAxisSettings] = None
    y: Optional[TaoAxisSettings] = None
    y2: Optional[TaoAxisSettings] = None
    y2_mirrors_y: Optional[bool] = None
    x_axis_scale_factor: Optional[float] = None

    # 'set plot':
    n_curve_points: Optional[int] = None

    def get_commands(
        self,
        region_name: str,
        graph_name: str,
        graph_type: str,
    ) -> List[str]:
        """
        Get command strings to apply these settings with Tao.

        Parameters
        ----------
        region_name : str
            Region name.
        graph_name : str
            Graph name.
        graph_type : str
            Graph type.  Required to determine which commands to send - that is,
            `TaoFloorPlanSettings` will be skipped for non-floor plan graphs.

        Returns
        -------
        list of str
            Commands to send to Tao to apply these settings.
        """
        result = []
        for key in self.model_dump().keys():
            value = getattr(self, key)
            if value is None:
                continue

            if key == "commands":
                result.extend(
                    [
                        cmd.format(
                            settings=self,
                            region=region_name,
                            graph_name=graph_name,
                            graph_type=graph_type,
                            graph=f"{region_name}.{graph_name}",
                        )
                        for cmd in value
                        if cmd
                    ]
                )
                continue
            if key in ("curve_legend_origin",) and isinstance(value, tuple):
                value = QuickPlotPoint(*value)
            elif key in ("scale_margin", "margin", "text_legend_origin") and isinstance(
                value, tuple
            ):
                value = QuickPlotRectangle(*value)

            if isinstance(value, QuickPlotPoint):
                result.extend(value.get_commands(region_name, graph_name, key))
            elif isinstance(value, TaoFloorPlanSettings):
                result.extend(value.get_commands(region_name, graph_name))
            elif isinstance(value, QuickPlotRectangle):
                result.extend(value.get_commands(region_name, graph_name, key))
            elif isinstance(value, TaoAxisSettings):
                result.extend(value.get_commands(region_name, key))
            elif key == "n_curve_points":
                result.append(f"set plot {region_name} n_curve_pts = {value}")
            elif key == "text_legend":
                for legend_index, legend_value in value.items():
                    result.append(
                        f"set graph {region_name} text_legend({legend_index}) = {legend_value}"
                    )
            elif key == "box":
                for box_index, box_value in value.items():
                    result.append(f"set graph {region_name} box({box_index}) = {box_value}")
            elif isinstance(value, TaoFloorPlanSettings):
                if graph_type == "floor_plan":
                    result.extend(value.get_commands(region_name, graph_name))
            else:
                result.append(f"set graph {region_name}.{graph_name} {key} = {value}")
        return result

    @property
    def xlim(self) -> Optional[Limit]:
        if self.x is None:
            return None
        return self.x.scale

    @xlim.setter
    def xlim(self, xlim: Optional[Limit]):
        if self.x is None:
            self.x = TaoAxisSettings()
        self.x.scale = xlim

    @property
    def ylim(self) -> Optional[Limit]:
        if self.y is None:
            return None
        return self.y.scale

    @ylim.setter
    def ylim(self, ylim: Optional[Limit]):
        if self.y is None:
            self.y = TaoAxisSettings()
        self.y.scale = ylim
