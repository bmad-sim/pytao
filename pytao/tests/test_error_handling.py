import pytest

from ..tao_ctypes.util import (
    TaoCommandError,
    capture,
)


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
        with capture(functions=["tao_plo", "tao_plot2"]) as ctx:
            ctx.check_output("cmd", [f"[{err}] tao_plot:"])

    with capture(functions=["tao_plot"]) as ctx:
        ctx.check_output("cmd", [f"[{err}] tao_plot:"])


def test_capture_context_by_level() -> None:
    with capture(functions=[], by_level={"ERROR": ["tao_plot"]}) as ctx:
        ctx.check_output("cmd", ["[ERROR] tao_plot:"])

    with pytest.raises(TaoCommandError):
        with capture(functions=[], by_level={"ERROR": ["tao_plot"]}) as ctx:
            ctx.check_output("cmd", ["[ERROR] tao_plot:", "[FATAL] tao_plot:"])


def test_capture_context_by_command() -> None:
    with capture(functions=[], by_level={"ERROR": ["tao_plot"]}) as ctx:
        ctx.check_output("cmd", ["[ERROR] tao_plot:"])

    with pytest.raises(TaoCommandError):
        with capture(functions=[], by_level={"ERROR": ["tao_plot"]}) as ctx:
            ctx.check_output("cmd", ["[ERROR] tao_plot:", "[FATAL] tao_plot:"])
