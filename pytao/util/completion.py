"""
Tab completion helpers backed by Tao's ``pipe complete`` command.

Tao's completion engine breaks tokens on blanks only, and every candidate it
returns is a full replacement for the token being completed. Front ends here
must be configured with matching word delimiters (space and tab).
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)


def file_candidates(prefix: str) -> list[str]:
    """
    Complete a file path prefix, marking directories with a trailing separator.

    Parameters
    ----------
    prefix : str
        The partial path to complete. ``~`` is expanded.

    Returns
    -------
    list of str
    """
    pattern = glob.escape(os.path.expanduser(prefix)) + "*"
    return [path + os.sep if os.path.isdir(path) else path for path in glob.glob(pattern)]


def line_candidates(tao, line: str) -> tuple[str, list[str]]:
    """
    Completion candidates for a partial Tao command line.

    Parameters
    ----------
    tao : Tao
        The Tao instance to query.
    line : str
        The command line up to the cursor.

    Returns
    -------
    word : str
        The token being completed (the trailing blank-delimited word of the line).
    matches : list of str
        Full replacement candidates for ``word``. File path contexts are
        completed locally with `file_candidates`.
    """
    result = tao.complete(line, raises=False)
    if result["context"] == "FILE":
        return result["word"], file_candidates(result["word"])
    return result["word"], result["matches"]


def make_readline_completer(tao) -> Callable[[str, int], str | None]:
    """
    Build a completer function for the standard ``readline`` module.

    The caller must also set the completer delimiters to blanks only::

        readline.set_completer(make_readline_completer(tao))
        readline.set_completer_delims(" \\t")
        readline.parse_and_bind("tab: complete")

    Parameters
    ----------
    tao : Tao
        The Tao instance to query.

    Returns
    -------
    callable
        A ``(text, state)`` completer per the readline protocol.
    """
    matches: list[str] = []

    def completer(text: str, state: int) -> str | None:
        nonlocal matches
        if state == 0:
            import readline

            line = readline.get_line_buffer()[: readline.get_endidx()]
            try:
                _, matches = line_candidates(tao, line)
            except Exception:
                logger.debug("Tao completion failed", exc_info=True)
                matches = []
        if state < len(matches):
            return matches[state]
        return None

    return completer
