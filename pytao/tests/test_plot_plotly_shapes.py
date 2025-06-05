from __future__ import annotations

import logging
import math
from typing import Union

import pytest

from .. import SubprocessTao, Tao
from ..plotting.floor_plan_shapes import (
    BowTie,
    Box,
    Circle,
    Diamond,
    LetterX,
    SBend,
    Triangle,
    XBox,
)
from .conftest import test_artifacts

try:
    import plotly.graph_objects as go

    from ..plotting.plotly import _draw_floor_plan_shapes
    from .test_floor_plan_shape import make_shapes

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

AnyTao = Union[Tao, SubprocessTao]


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_floor_plan_shapes_plotly(request: pytest.FixtureRequest):
    """Test floor plan shapes rendering with Plotly."""
    from plotly.subplots import make_subplots

    # Create subplot figure with 2 rows
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Angles 0-90°", "Angles 90-180°"),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # Add shapes for first angle range to first subplot
    temp_fig1 = go.Figure()
    for shape in make_shapes(width=1, height=2, angle_low=0, angle_high=90):
        _draw_floor_plan_shapes(temp_fig1, [shape])

    # Copy traces from temp figure to subplot
    for trace in temp_fig1.data:
        fig.add_trace(trace, row=1, col=1)

    # Add shapes for second angle range to second subplot
    temp_fig2 = go.Figure()
    for shape in make_shapes(width=1, height=2, angle_low=90, angle_high=180):
        _draw_floor_plan_shapes(temp_fig2, [shape])

    # Copy traces from temp figure to subplot
    for trace in temp_fig2.data:
        fig.add_trace(trace, row=2, col=1)

    # Copy shapes from temp figures to main figure
    if temp_fig1.layout.shapes:
        for shape in temp_fig1.layout.shapes:
            fig.add_shape(shape, row=1, col=1)

    if temp_fig2.layout.shapes:
        for shape in temp_fig2.layout.shapes:
            fig.add_shape(shape, row=2, col=1)

    # Update layout
    fig.update_layout(
        title="Floor Plan Shapes Test - Plotly",
        showlegend=False,
        height=800,
        width=800,
    )

    # Set equal aspect ratio for both subplots
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_xaxes(scaleanchor="y2", scaleratio=1, row=2, col=1)

    # Set y-axis limits
    fig.update_yaxes(range=[-5, 85], row=1, col=1)
    fig.update_yaxes(range=[-5, 85], row=2, col=1)

    # Save to HTML file
    filename = test_artifacts / f"{request.node.name}.html"
    fig.write_html(filename)
    print(f"Saved Plotly floor plan shapes test to {filename}")


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_individual_shapes():
    """Test individual floor plan shapes with Plotly to ensure they render correctly."""
    width, height = 2.0, 1.0

    shapes_to_test = [
        ("Box", Box(x1=0, y1=0, x2=width, y2=0, off1=width, off2=height, angle_start=0)),
        ("Circle", Circle(x1=0, y1=3, x2=width, y2=3, off1=width, off2=height, angle_start=0)),
        (
            "Diamond",
            Diamond(x1=0, y1=6, x2=width, y2=6, off1=width, off2=height, angle_start=0),
        ),
        (
            "Triangle_up",
            Triangle(
                orientation="u",
                x1=0,
                y1=9,
                x2=width,
                y2=9,
                off1=width,
                off2=height,
                angle_start=0,
            ),
        ),
        ("XBox", XBox(x1=0, y1=12, x2=width, y2=12, off1=width, off2=height, angle_start=0)),
        (
            "BowTie",
            BowTie(x1=0, y1=15, x2=width, y2=15, off1=width, off2=height, angle_start=0),
        ),
    ]

    fig = go.Figure()

    for name, shape in shapes_to_test:
        _draw_floor_plan_shapes(fig, [shape])

        # Add text annotation to identify the shape
        fig.add_annotation(
            x=width / 2,
            y=shape.y1 + height / 2,
            text=name,
            showarrow=False,
            font=dict(size=10),
            xshift=width * 20,  # Offset text to the right
        )

    fig.update_layout(
        title="Individual Floor Plan Shapes - Plotly",
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            range=[-1, 10],
        ),
        yaxis=dict(range=[-2, 18]),
        showlegend=False,
        width=600,
        height=800,
    )

    # Check that we actually have some data
    assert len(fig.data) > 0 or (
        fig.layout.shapes and len(fig.layout.shapes) > 0
    ), "Figure should contain either traces or shapes"


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_rotated_shapes():
    """Test floor plan shapes with various rotations."""
    width, height = 1.0, 0.5

    fig = go.Figure()

    # Test a few shapes at different angles
    angles = [0, 30, 45, 90, 135, 180]

    for i, angle in enumerate(angles):
        y_offset = i * 3

        # Test Box shape at different angles
        shape = Box(
            x1=0,
            y1=y_offset,
            x2=width,
            y2=y_offset,
            off1=width,
            off2=height,
            angle_start=math.radians(angle),
        )
        _draw_floor_plan_shapes(fig, [shape])

        # Add annotation showing the angle
        fig.add_annotation(
            x=width + 1,
            y=y_offset,
            text=f"{angle}°",
            showarrow=False,
            font=dict(size=10),
        )

    fig.update_layout(
        title="Rotated Floor Plan Shapes - Plotly",
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            range=[-2, 5],
        ),
        yaxis=dict(range=[-1, len(angles) * 3]),
        showlegend=False,
        width=600,
        height=600,
    )

    # Verify we have rendered content
    has_content = len(fig.data) > 0 or (fig.layout.shapes and len(fig.layout.shapes) > 0)
    assert has_content, "Rotated shapes should produce visible content"


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_complex_shapes():
    """Test more complex floor plan shapes like SBend."""
    fig = go.Figure()

    # Test SBend shape which has complex geometry
    sbend = SBend(
        x1=0,
        y1=0,
        x2=2,
        y2=0,
        off1=2,
        off2=1,
        angle_start=0,
        angle_end=math.radians(30),
        rel_angle_start=0,
        rel_angle_end=0,
    )

    _draw_floor_plan_shapes(fig, [sbend])

    # Test LetterX shape
    letter_x = LetterX(
        x1=0,
        y1=3,
        x2=2,
        y2=3,
        off1=2,
        off2=1,
        angle_start=0,
    )

    _draw_floor_plan_shapes(fig, [letter_x])

    fig.update_layout(
        title="Complex Floor Plan Shapes - Plotly",
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            range=[-1, 4],
        ),
        yaxis=dict(range=[-1, 5]),
        showlegend=False,
        width=600,
        height=400,
    )

    # Should have some rendered content
    has_content = len(fig.data) > 0 or (fig.layout.shapes and len(fig.layout.shapes) > 0)
    assert has_content, "Complex shapes should produce visible content"


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_shape_properties():
    """Test that floor plan shapes maintain proper properties when rendered."""
    fig = go.Figure()

    # Create a simple box
    box = Box(
        x1=0,
        y1=0,
        x2=2,
        y2=0,
        off1=2,
        off2=1,
        angle_start=0,
        color="red",
        line_width=2.0,
    )

    _draw_floor_plan_shapes(fig, [box])

    # Check that figure has content
    has_traces = len(fig.data) > 0
    has_shapes = fig.layout.shapes and len(fig.layout.shapes) > 0

    assert has_traces or has_shapes, "Shape should produce either traces or layout shapes"

    if has_traces:
        # Check that traces have proper data
        for trace in fig.data:
            if hasattr(trace, "x") and hasattr(trace, "y"):
                assert len(trace.x) > 0, "Trace should have x data"
                assert len(trace.y) > 0, "Trace should have y data"


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_multiple_elements():
    """Test rendering multiple floor plan elements together."""
    fig = go.Figure()

    shapes = [
        Box(x1=0, y1=0, x2=1, y2=0, off1=1, off2=0.5, angle_start=0),
        Circle(x1=2, y1=0, x2=3, y2=0, off1=1, off2=0.5, angle_start=0),
        Diamond(x1=4, y1=0, x2=5, y2=0, off1=1, off2=0.5, angle_start=0),
    ]

    # Draw all elements at once
    _draw_floor_plan_shapes(fig, shapes)

    fig.update_layout(
        title="Multiple Floor Plan Elements - Plotly",
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            range=[-0.5, 6],
        ),
        yaxis=dict(range=[-1, 1]),
        showlegend=False,
        width=800,
        height=300,
    )

    # Should have content for multiple elements
    has_content = len(fig.data) > 0 or (fig.layout.shapes and len(fig.layout.shapes) > 0)
    assert has_content, "Multiple elements should produce visible content"
