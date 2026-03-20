import pytest

from ..errors import (
    TaoCommandError,
    error_filter_context,
    filter_tao_messages,
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
