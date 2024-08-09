import shlex


def make_tao_init(init: str, **kwargs) -> str:
    """
    Make Tao init string based on optional flags/command-line arguments.

    Parameters
    ----------
    init : str
        The user-specified init string.
    **kwargs :
        Command-line switches without the leading `-`,
        mapped to their respective values.
        Only added as an argument if not empty and not False.
    """
    result = shlex.split(init)
    for name, value in kwargs.items():
        switch = f"-{name}"
        if switch in result:
            continue
        if not value:
            continue
        result.append(switch)
        if value not in {True, False}:
            result.append(str(value))
    return shlex.join(result)
