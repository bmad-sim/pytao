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
from collections.abc import Callable

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


def make_prompt_toolkit_completer(tao):
    """
    Build a ``prompt_toolkit`` completer backed by Tao's completion engine.

    Unlike IPython's completer stack, this completes on a blank line too, so
    an empty prompt offers the full list of Tao commands.

    Parameters
    ----------
    tao : Tao
        The Tao instance to query.

    Returns
    -------
    prompt_toolkit.completion.Completer
    """
    from prompt_toolkit.completion import Completer, Completion

    class TaoCompleter(Completer):
        def get_completions(self, document, complete_event):
            try:
                word, matches = line_candidates(tao, document.text_before_cursor)
            except Exception:
                logger.debug("Tao completion failed", exc_info=True)
                return
            for match in matches:
                yield Completion(match, start_position=-len(word))

    return TaoCompleter()
