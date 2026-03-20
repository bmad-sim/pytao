from __future__ import annotations

import importlib
import sys


def import_by_name(clsname: str):
    """
    Import the given object by name.

    Parameters
    ----------
    clsname : str
        The module path to find the class e.g.
        ``"pytao.Tao"``
    """
    module, cls = clsname.rsplit(".", 1)
    if module not in sys.modules:
        importlib.import_module(module)

    mod = sys.modules[module]
    try:
        return getattr(mod, cls)
    except AttributeError:
        raise ImportError(f"Unable to import {clsname!r} from module {module!r}")
