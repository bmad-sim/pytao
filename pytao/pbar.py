from __future__ import annotations

import enum
import functools
import sys
import threading
import time
import typing
from contextlib import contextmanager

if typing.TYPE_CHECKING:
    from . import Tao


class OutputMode(enum.Enum):
    """Jupyter Notebook output support."""

    unknown = "unknown"
    plain = "plain"
    html = "html"


def active_beam_track_monitor(
    tao: Tao, cb, cancel_event: threading.Event, *, rate: float = 0.1
):
    last_idx = None

    while not cancel_event.is_set():
        idx = tao.get_active_beam_track_element()
        if idx != last_idx:
            cb(idx)
            last_idx = idx

        time.sleep(rate)


@functools.cache
def get_output_mode() -> OutputMode:
    """
    Get the output mode for lume-impact objects.

    This works by way of interacting with IPython display and seeing what
    choice it makes regarding reprs.

    Returns
    -------
    OutputMode
        The detected output mode.
    """
    if "IPython" not in sys.modules or "IPython.display" not in sys.modules:
        return OutputMode.plain

    from IPython.display import display

    class ReprCheck:
        mode: OutputMode = OutputMode.unknown

        def _repr_html_(self) -> str:
            self.mode = OutputMode.html
            return "<!-- lume-impact detected Jupyter and will use HTML for rendering. -->"

        def __repr__(self) -> str:
            self.mode = OutputMode.plain
            return ""

    check = ReprCheck()
    display(check)
    return check.mode


def is_jupyter() -> bool:
    """Is Jupyter detected?"""
    return get_output_mode() == OutputMode.html


@contextmanager
def maybe_progress_bar(enable: bool, **kwargs):
    if is_jupyter():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    if enable:
        with tqdm(**kwargs) as pbar:
            yield pbar
    else:
        yield None
