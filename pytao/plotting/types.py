from typing import List, Tuple
from typing_extensions import NotRequired, TypedDict


FloorOrbitInfo = TypedDict(
    "FloorOrbitInfo",
    {
        "branch_index": int,
        "index": int,
        "ele_key": str,
        "axis": str,
        "orbits": List[float],
    },
)
BuildingWallGraphInfo = TypedDict(
    "BuildingWallGraphInfo",
    {
        "index": int,
        "point": int,
        "offset_x": float,
        "offset_y": float,
        "radius": float,
    },
)

BuildingWallInfo = TypedDict(
    "BuildingWallInfo",
    {
        "index": int,
        "name": str,
        "constraint": str,
        "shape": str,
        "color": str,
        "line_width": float,
    },
)
BuildingWallGlobalInfo = TypedDict(
    "BuildingWallGlobalInfo",
    {
        "index": int,
        "z": float,
        "x": float,
        "radius": float,
        "z_center": float,
        "x_center": float,
    },
)

WaveParams = TypedDict(
    "WaveParams",
    {
        "ix_a1": float,
        "ix_a2": float,
        "ix_b1": float,
        "ix_b2": float,
    },
)

PlotCurveLineInfo = TypedDict(
    "PlotCurveLineInfo",
    {
        "color": str,
        "line^pattern": str,
        "width": int,
    },
)

PlotCurveSymbolInfo = TypedDict(
    "PlotCurveSymbolInfo",
    {
        "color": str,
        "fill_pattern": str,
        "height": float,
        "line_width": int,
        "symbol^type": str,
    },
)
PlotCurveInfo = TypedDict(
    "PlotCurveInfo",
    {
        "-1^ix_branch": int,
        "-1^ix_bunch": int,
        "component": str,
        "data_source": str,
        "data_type": str,
        "data_type_x": str,
        "draw_error_bars": bool,
        "draw_line": bool,
        "draw_symbol_index": bool,
        "draw_symbols": bool,
        "ele_ref_name": str,
        "ix_ele_ref": int,
        "ix_ele_ref_track": int,
        "ix_universe": int,
        "legend_text": str,
        "line": PlotCurveLineInfo,
        "message_text": str,
        "name": str,
        "smooth_line_calc": bool,
        "symbol": PlotCurveSymbolInfo,
        "symbol_every": int,
        "symbol_line_width": int,
        "use_y2": bool,
        "valid": bool,
        "why_invalid": str,
        "y_axis_scale_factor": float,
        "z_color_autoscale": bool,
        "z_color_data_type": str,
        "z_color_is_on": bool,
        "z_color_max": float,
        "z_color_min": float,
    },
)


PlotGraphInfo = TypedDict(
    "PlotGraphInfo",
    {
        "-1^ix_branch": int,
        "clip": bool,
        "draw_axes": bool,
        "draw_curve_legend": bool,
        "draw_grid": bool,
        "draw_only_good_user_data_or_vars": bool,
        "floor_plan_correct_distortion": bool,
        "floor_plan_draw_building_wall": bool,
        "floor_plan_draw_only_first_pass": bool,
        "floor_plan_flip_label_side": bool,
        "floor_plan_orbit_color": str,
        "floor_plan_orbit_lattice": str,
        "floor_plan_orbit_pattern": str,
        "floor_plan_orbit_scale": float,
        "floor_plan_orbit_width": int,
        "floor_plan_rotation": float,
        "floor_plan_size_is_absolute": bool,
        "floor_plan_view": str,
        "graph^type": str,
        "is_valid": bool,
        "ix_universe": int,
        "limited": bool,
        "name": str,
        "num_curves": int,
        "symbol_size_scale": float,
        "title": str,
        "title_suffix": str,
        "why_invalid": str,
        "x_axis^type": str,
        "x_axis_scale_factor": float,
        "x_bounds": str,
        "x_draw_label": bool,
        "x_draw_numbers": bool,
        "x_label": str,
        "x_label_color": str,
        "x_label_offset": float,
        "x_major_div_nominal": int,
        "x_major_tick_len": float,
        "x_max": float,
        "x_min": float,
        "x_minor_div": int,
        "x_minor_div_max": int,
        "x_minor_tick_len": float,
        "x_number_offset": float,
        "x_number_side": int,
        "x_tick_side": int,
        "y2_axis^type": str,
        "y2_bounds": str,
        "y2_draw_label": bool,
        "y2_draw_numbers": bool,
        "y2_label": str,
        "y2_label_color": str,
        "y2_label_offset": float,
        "y2_major_div_nominal": int,
        "y2_major_tick_len": float,
        "y2_max": float,
        "y2_min": float,
        "y2_minor_div": int,
        "y2_minor_div_max": int,
        "y2_minor_tick_len": float,
        "y2_mirrors_y": bool,
        "y2_number_offset": float,
        "y2_number_side": int,
        "y2_tick_side": int,
        "y_axis^type": str,
        "y_bounds": str,
        "y_draw_label": bool,
        "y_draw_numbers": bool,
        "y_label": str,
        "y_label_color": str,
        "y_label_offset": float,
        "y_major_div_nominal": int,
        "y_major_tick_len": float,
        "y_max": float,
        "y_min": float,
        "y_minor_div": int,
        "y_minor_div_max": int,
        "y_minor_tick_len": float,
        "y_number_offset": float,
        "y_number_side": int,
        "y_tick_side": int,
        # "curve[1..N]": str,
    },
)


PlotHistogramInfo = TypedDict(
    "PlotHistogramInfo",
    {
        "center": float,
        "density_normalized": bool,
        "maximum": float,
        "minimum": float,
        "number": float,
        "weight_by_charge": bool,
        "width": float,
    },
)

PlotLatLayoutInfo = TypedDict(
    "PlotLatLayoutInfo",
    {
        "color": str,
        "ele_s": float,
        "ele_s_start": float,
        "index": int,
        "label_name": str,
        "line_width": float,
        "shape": str,
        "y1": float,
        "y2": float,
    },
)

FloorPlanElementInfo = TypedDict(
    "FloorPlanElementInfo",
    {
        "branch_index": int,
        "color": str,
        "ele_key": str,
        "end1_r1": float,
        "end1_r2": float,
        "end1_theta": float,
        "end2_r1": float,
        "end2_r2": float,
        "end2_theta": float,
        "index": int,
        "label_name": str,
        "line_width": float,
        "shape": str,
        "y1": float,
        "y2": float,
        # Only for sbend
        "ele_l": NotRequired[float],
        "ele_angle": NotRequired[float],
        "ele_e1": NotRequired[float],
        "ele_e": NotRequired[float],
    },
)


Point = Tuple[float, float]
