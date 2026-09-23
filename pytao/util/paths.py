import os.path
import re
from pathlib import Path


def normalize_path(path: str | Path) -> Path:
    """
    Normalize a file path, expanding variables and user home (``~``) references.

    Parameters
    ----------
    path : str or Path
        The file path to normalize.

    Returns
    -------
    Path

    Notes
    -----
    - User and environment variables in the path will be expanded.
    - Backslashes in the path will be replaced with '_pass', based on Tao element naming schemes.
    - The path will be resolved to an absolute path.

    Examples
    --------
    >>> normalize_path("~/example/path", ".txt")
    PosixPath('/home/user/example/path.txt')

    >>> normalize_path("$HOME/example/path.txt")
    PosixPath('/home/user/example/path.txt')
    """
    path = os.path.expanduser(os.path.expandvars(str(path)))
    return Path(path).resolve()


def set_design_lattice(init_contents: str, lattice_file: str, index: int = 1) -> str:
    """
    Set the path of `design_lattice(index)%file` in the given `tao.init` contents.

    If the contents have no `tao_design_lattice` namelist, one is prepended.

    Parameters
    ----------
    init_contents : str
        The `tao.init`-format namelist file contents.
    lattice_file : str
        The lattice filename to use.
    index : int, default=1
        The design lattice (universe) index.

    Returns
    -------
    str
        The updated `tao.init` contents.
    """
    # TODO using nmlform would be better at some point
    if "tao_design_lattice" not in init_contents:
        return "\n".join(
            (
                "&tao_design_lattice",
                "  n_universes = 1",
                f'  design_lattice(1)%file = "{lattice_file}"',
                "/",
                "",
                init_contents,
            )
        )
    return re.sub(
        rf"^(\s*design_lattice\s*\(\s*{index}\s*\)\s*%\s*file)\s*=.*$",
        lambda match: f'{match.group(1)} = "{lattice_file}"',
        init_contents,
        flags=re.MULTILINE | re.IGNORECASE,
    )
