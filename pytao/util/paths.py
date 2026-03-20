import os.path
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
