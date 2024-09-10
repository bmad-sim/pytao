from typing import List, Set

import pytest

from ..tao_ctypes.util import TaoMessage, capture_messages_from_functions, filter_output_lines


@pytest.mark.parametrize(
    ("functions_to_filter", "lines", "expected_lines"),
    [
        pytest.param(
            {"foo"},
            [
                "test",
                "[ERROR ] foo:",
                "   bar.",
                "   bar.",
                "[ERROR ] foo:",
                "[ERROR ] foo:",
                "   bar.",
                "   bar.",
                "bar!",
            ],
            [
                "test",
                "bar!",
            ],
        ),
        pytest.param(
            set("bar"),
            [
                "test",
                "[ERROR ] bar:",
                "[ERROR ] foo:",
                "   bar.",
                "   bar.",
                "[ERROR ] bar:",
                "bar!",
            ],
            [
                "test",
                "[ERROR ] foo:",
                "   bar.",
                "   bar.",
                "bar!",
            ],
        ),
        pytest.param(
            set(),
            [
                "test",
                "[ERROR ] foo:",
                "   bar.",
                "   bar.",
                "bar!",
            ],
            [
                "test",
                "[ERROR ] foo:",
                "   bar.",
                "   bar.",
                "bar!",
            ],
        ),
    ],
)
def test_filter_lines(
    lines: List[str],
    functions_to_filter: Set[str],
    expected_lines: List[str],
) -> None:
    assert filter_output_lines(lines, functions_to_filter) == expected_lines


@pytest.mark.parametrize(
    ("lines", "expected_messages", "expected_lines"),
    [
        pytest.param(
            [
                "test",
                "[ERROR] foo1:",
                "   bar.",
                "   bar.",
                "[ERROR | date] foo2:",
                "[ERROR] foo3:",
                "   bar1.",
                "   bar2.",
                "bar!",
            ],
            [
                TaoMessage(level="ERROR", function="foo1", message="bar.\nbar."),
                TaoMessage(level="ERROR", function="foo2", message=""),
                TaoMessage(level="ERROR", function="foo3", message="bar1.\nbar2."),
            ],
            [
                "test",
                "bar!",
            ],
        ),
        pytest.param(
            [
                "start",
                "[INFO] foo1:",
                "   bar1.",
                "test",
                "[SUCCESS] foo2:",
                "   bar2.",
                "[WARNING] foo3:",
                "   bar3.",
                "[ERROR] foo4:",
                "   bar4.",
                "[FATAL ] foo5:",
                "   bar5.",
                "[ABORT ] foo6:",
                "   bar6.",
                "bar!",
                "[MESSAGE ] foo7:",
                "   bar7.",
            ],
            [
                TaoMessage(level="INFO", function="foo1", message="bar1."),
                TaoMessage(level="SUCCESS", function="foo2", message="bar2."),
                TaoMessage(level="WARNING", function="foo3", message="bar3."),
                TaoMessage(level="ERROR", function="foo4", message="bar4."),
                TaoMessage(level="FATAL", function="foo5", message="bar5."),
                TaoMessage(level="ABORT", function="foo6", message="bar6."),
                TaoMessage(level="MESSAGE", function="foo7", message="bar7."),
            ],
            [
                "start",
                "test",
                "bar!",
            ],
        ),
    ],
)
def test_capture_messages_from_functions(
    lines: List[str],
    expected_lines: List[str],
    expected_messages: List[TaoMessage],
) -> None:
    assert capture_messages_from_functions(lines) == (expected_lines, expected_messages)
