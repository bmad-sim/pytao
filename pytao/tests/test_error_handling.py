import pytest

from ..errors import (
    TaoCommandError,
    TaoMessage,
    capture_messages_from_functions,
    error_filter_context,
    filter_tao_messages,
    raise_for_error_messages,
)
from ..errors import (
    filter_tao_messages_context as filter_ctx,
)


def test_capture_global() -> None:
    ctx = filter_tao_messages(functions=["__foobar__"])
    assert ctx.functions == frozenset({"__foobar__"})
    assert error_filter_context.get().functions == frozenset({"__foobar__"})

    ctx = filter_tao_messages(functions=[])
    assert ctx.functions == frozenset()


@pytest.mark.parametrize(
    "err",
    [
        "ERROR",
        "FATAL",
        "ABORT",
    ],
)
def test_capture_context(err: str) -> None:
    with pytest.raises(TaoCommandError):
        with filter_ctx(functions=["tao_plo", "tao_plot2"]) as ctx:
            ctx.check_output("cmd", [f"[{err}] tao_plot:"])

    with filter_ctx(functions=["tao_plot"]) as ctx:
        ctx.check_output("cmd", [f"[{err}] tao_plot:"])


def test_capture_context_by_level() -> None:
    with filter_ctx(functions=[], by_level={"ERROR": ["tao_plot"]}) as ctx:
        ctx.check_output("cmd", ["[ERROR] tao_plot:"])

    with pytest.raises(TaoCommandError):
        with filter_ctx(functions=[], by_level={"ERROR": ["tao_plot"]}) as ctx:
            ctx.check_output("cmd", ["[ERROR] tao_plot:", "[FATAL] tao_plot:"])


def test_capture_context_by_command() -> None:
    with filter_ctx(functions=[], by_level={"ERROR": ["tao_plot"]}) as ctx:
        ctx.check_output("cmd", ["[ERROR] tao_plot:"])

    with pytest.raises(TaoCommandError):
        with filter_ctx(functions=[], by_level={"ERROR": ["tao_plot"]}) as ctx:
            ctx.check_output("cmd", ["[ERROR] tao_plot:", "[FATAL] tao_plot:"])


def test_raise_for_error_messages():
    lines = [
        "[FATAL] fatal_func:",
        "   func fatal",
        "[ERROR] error_func:",
        "   func error",
        "[ABORT] abort_func:",
        "   func abort",
        "[INFO] info_func:",
        "   func info",
        "actual output",
    ]

    expected_errors = [
        TaoMessage(level="FATAL", function="fatal_func", message="func fatal"),
        TaoMessage(level="ERROR", function="error_func", message="func error"),
        TaoMessage(level="ABORT", function="abort_func", message="func abort"),
    ]
    expected_messages = [
        *expected_errors,
        TaoMessage(level="INFO", function="info_func", message="func info"),
    ]
    _, messages = capture_messages_from_functions(lines)
    with pytest.raises(TaoCommandError) as cap:
        raise_for_error_messages(cmd="foo", lines=lines, errors=messages)
    ex = cap.value
    assert ex.messages == expected_messages
    assert ex.errors == expected_errors
    assert "\n".join(lines) == ex.tao_output
