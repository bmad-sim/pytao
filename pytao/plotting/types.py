from typing import List, Optional, Tuple

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


PlotRegionInfo = TypedDict(
    "PlotRegionInfo",
    {
        "num_graphs": int,
        # "graph[1]": str,
        "name": str,
        "description": str,
        "x_axis_type": str,
        "autoscale_x": bool,
        "autoscale_y": bool,
        "autoscale_gang_x": bool,
        "autoscale_gang_y": bool,
        "n_curve_pts": int,
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
        # **{f"curve[{n}]": NotRequired[str] for n in range(1, 100)},
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
        "ix_branch": int,
        "ix_ele": int,
        "ele_s_start": float,
        "ele_s_end": float,
        "line_width": float,
        "shape": str,
        "y1": float,
        "y2": float,
        "color": str,
        "label_name": str,
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

PlotPage = TypedDict(
    "PlotPage",
    {
        "title_string": str,
        "title__x": float,
        "title_y": float,
        "title_units": str,
        "title_justify": str,
        "subtitle_string": str,
        "subtitle__x": float,
        "subtitle_y": float,
        "subtitle_units": str,
        "subtitle_justify": str,
        "size": list,
        "n_curve_pts": int,
        "border": list,
        "text_height": float,
        "main_title_text_scale": float,
        "graph_title_text_scale": float,
        "axis_number_text_scale": float,
        "axis_label_text_scale": float,
        "key_table_text_scale": float,
        "legend_text_scale": float,
        "floor_plan_shape_scale": float,
        "floor_plan_text_scale": float,
        "lat_layout_shape_scale": float,
        "lat_layout_text_scale": float,
        "delete_overlapping_plots": str,
        "draw_graph_title_suffix": str,
        "curve_legend_line_len": float,
        "curve_legend_text_offset": float,
    },
)


FloatVariableInfo = TypedDict(
    "FloatVariableInfo",
    {
        "model_value": float,
        "base_value": float,
        "ele_name": str,
        "attrib_name": str,
        "ix_v1": int,
        "ix_var": int,
        "ix_dvar": int,
        "ix_attrib": int,
        "ix_key_table": int,
        "design_value": float,
        "scratch_value": float,
        "old_value": float,
        "meas_value": float,
        "ref_value": float,
        "correction_value": float,
        "high_lim": float,
        "low_lim": float,
        "step": float,
        "weight": float,
        "delta_merit": float,
        "merit": float,
        "dmerit_dvar": float,
        "key_val0": float,
        "key_delta": float,
        "s": float,
        "var^merit_type": str,
        "exists": bool,
        "good_var": bool,
        "good_user": bool,
        "good_opt": bool,
        "good_plot": bool,
        "useit_opt": bool,
        "useit_plot": bool,
        "key_bound": bool,
    },
)


Point = Tuple[float, float]
Limit = Tuple[float, float]
OptionalLimit = Optional[Limit]


ShapeListInfo = TypedDict(
    "ShapeListInfo",
    {
        "shape_index": int,
        "ele_name": str,
        "shape": str,
        "color": str,
        "shape_size": float,
        "type_label": str,
        "shape_draw": bool,
        "multi_shape": bool,
        "line_width": int,
    },
)
