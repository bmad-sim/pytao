from __future__ import annotations

import contextlib
import enum
import functools
import sys
import threading
import time
import typing
from contextlib import contextmanager

import pydantic

if typing.TYPE_CHECKING:
    from tqdm import tqdm
    from . import Tao


class OutputMode(enum.Enum):
    """Jupyter Notebook output support."""

    unknown = "unknown"
    plain = "plain"
    html = "html"


BeamTrackUpdateCallback = typing.Callable[[int], None]


def active_beam_track_monitor(
    tao: Tao,
    cb: BeamTrackUpdateCallback,
    cancel_event: threading.Event,
    *,
    rate: float = 0.1,
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
    """Detect if we are running in a Jupyter notebook."""
    return get_output_mode() == OutputMode.html


@contextmanager
def maybe_progress_bar(enable: bool, jupyter: bool | None = False, **kwargs):
    """
    Context manager that conditionally yields a tqdm progress bar.

    Automatically detects if running in Jupyter notebook to use the appropriate
    tqdm version.

    Parameters
    ----------
    enable : bool
        Whether to create and yield a progress bar
    jupyter : bool or None, optional
        Specifically request a Jupyter progress bar (True), a terminal progress
        bar (False), or - the default - auto-detect (None).
    **kwargs : dict
        Keyword arguments passed to tqdm constructor

    Yields
    -------
    tqdm or None
        A tqdm progress bar instance if enabled=True, otherwise None

    Examples
    --------
    >>> with maybe_progress_bar(enable=True, total=100) as pbar:
    ...     for i in range(100):
    ...         if pbar:
    ...             pbar.update()
    """
    if is_jupyter():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    if enable:
        with tqdm(**kwargs) as pbar:
            yield pbar
    else:
        yield None


class TrackingInfo(pydantic.BaseModel):
    track_start: str
    track_end: str

    ix_to_name: dict[int, str]
    name_to_ix: dict[str, int]

    @classmethod
    def from_tao(cls, tao: Tao, *, ix_uni: str = "", ix_branch: str = "") -> TrackingInfo:
        ix_eles = typing.cast(
            list[int],
            tao.lat_list(
                "*",
                "ele.ix_ele",
                flags="-array_out -track_only",
                ix_uni=ix_uni,
                ix_branch=ix_branch,
            ),
        )
        ele_names = typing.cast(
            list[str],
            tao.lat_list(
                "*",
                "ele.name",
                flags="-array_out -track_only",
                ix_uni=ix_uni,
                ix_branch=ix_branch,
            ),
        )
        ele_names = [ele_name.upper() for ele_name in ele_names]

        ix_to_name = dict(zip(ix_eles, ele_names))
        name_to_ix = dict(zip(ele_names, ix_eles))

        beam = typing.cast(dict[str, str], tao.beam(ix_branch))
        return cls(
            ix_to_name=ix_to_name,
            name_to_ix=name_to_ix,
            track_start=beam["track_start"] or ele_names[0],
            track_end=beam["track_end"] or ele_names[-1],
        )


@contextlib.contextmanager
def track_beam_wrapper(
    tao: Tao,
    *,
    ix_branch: str = "",
    ix_uni: str = "",
    use_progress_bar: bool = True,
    jupyter: bool | None = None,
    leave: bool = False,
) -> typing.Generator[tqdm | None]:
    """
    A context manager that optionally adds a progress bar during beam tracking.

    Parameters
    ----------
    tao : Tao
        The Tao instance to track beam progress for.
    ix_branch : str, optional
        Branch index, by default ""
    ix_uni : str, optional
        Universe index, by default ""
    use_progress_bar : bool, optional
        Whether to show a progress bar, by default True
    jupyter : bool | None, optional
        Whether running in Jupyter environment. If None (default), auto-detects
        the presence of Jupyter.
    leave : bool, optional
        Leave the progress bar after completion. Defaults to False.

    Yields
    ------
    tqdm.tqdm
        Context manager yields nothing but provides progress bar functionality

    Notes
    -----
    Uses threading to monitor beam tracking progress without blocking the main process.
    Progress bar shows current element name and index while tracking.
    """
    track = TrackingInfo.from_tao(tao, ix_branch=ix_branch, ix_uni=ix_uni)
    start_idx = track.name_to_ix[track.track_start.upper()]
    end_idx = track.name_to_ix[track.track_end.upper()]

    def update_progress_bar(active_idx: int):
        if pbar is None:
            return

        if active_idx == -1:
            # Not yet started somehow
            active_idx = start_idx

        ele = track.ix_to_name.get(active_idx, "?")
        pbar.set_postfix({"Element": ele, "ix_ele": active_idx}, refresh=False)
        pbar.n = active_idx - start_idx
        pbar.refresh()

    cancel_event = threading.Event()
    pbar = None
    try:
        with maybe_progress_bar(
            use_progress_bar,
            total=end_idx - start_idx,
            leave=leave,
            jupyter=jupyter,
            unit="ele",
        ) as pbar:
            if use_progress_bar:
                thr = threading.Thread(
                    daemon=True,
                    target=active_beam_track_monitor,
                    kwargs=dict(tao=tao, cb=update_progress_bar, cancel_event=cancel_event),
                )
                thr.start()
            yield pbar
    finally:
        cancel_event.set()
        if pbar is not None:
            pbar.close()
