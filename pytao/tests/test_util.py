import pytest

from ..errors import TaoMessage, capture_messages_from_functions, filter_output_lines
from ..util.paths import set_design_lattice


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
    lines: list[str],
    functions_to_filter: set[str],
    expected_lines: list[str],
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
    lines: list[str],
    expected_lines: list[str],
    expected_messages: list[TaoMessage],
) -> None:
    assert capture_messages_from_functions(lines) == (expected_lines, expected_messages)


@pytest.mark.parametrize(
    ("line", "expected_line"),
    [
        pytest.param(
            '  design_lattice(1)%file = "old.bmad"',
            '  design_lattice(1)%file = "new.bmad"',
            id="typical",
        ),
        pytest.param(
            'design_lattice(1)%file="old.bmad"',
            'design_lattice(1)%file = "new.bmad"',
            id="no-whitespace",
        ),
        pytest.param(
            "\tdesign_lattice( 1 ) % file  =  'old.bmad'",
            '\tdesign_lattice( 1 ) % file = "new.bmad"',
            id="extra-whitespace",
        ),
        pytest.param(
            '  Design_Lattice(1)%File = "old.bmad"',
            '  Design_Lattice(1)%File = "new.bmad"',
            id="mixed-case",
        ),
    ],
)
def test_set_design_lattice(line: str, expected_line: str) -> None:
    init_contents = f"""\
&tao_design_lattice
  n_universes = 1
{line}
/
"""
    assert (
        set_design_lattice(init_contents, "new.bmad")
        == f"""\
&tao_design_lattice
  n_universes = 1
{expected_line}
/
"""
    )


def test_set_design_lattice_other_index() -> None:
    init_contents = """\
&tao_design_lattice
  n_universes = 2
  design_lattice(1)%file = "one.bmad"
  design_lattice(2)%file = "two.bmad"
/
"""
    assert (
        set_design_lattice(init_contents, "new.bmad", index=2)
        == """\
&tao_design_lattice
  n_universes = 2
  design_lattice(1)%file = "one.bmad"
  design_lattice(2)%file = "new.bmad"
/
"""
    )


def test_set_design_lattice_adds_namelist() -> None:
    assert (
        set_design_lattice("&tao_params\n/\n", "new.bmad")
        == """\
&tao_design_lattice
  n_universes = 1
  design_lattice(1)%file = "new.bmad"
/

&tao_params
/
"""
    )
