import logging

logger = logging.getLogger(__name__)


def mpl_color(pgplot_color: str) -> str:
    """Pgplot color to matplotlib color."""
    return {
        "yellow_green": "greenyellow",
        "light_green": "limegreen",
        "navy_blue": "navy",
        "reddish_purple": "mediumvioletred",
        "dark_grey": "gray",
        "light_grey": "lightgray",
        "transparent": "none",
    }.get(pgplot_color.lower(), pgplot_color)


_pgplot_to_mpl_chars = [
    (" ", "\\ "),
    ("%", "\\%"),
    ("\\\\(2265)", "\\partial"),
    ("\\\\ga", "\\alpha"),
    ("\\\\gb", "\\beta"),
    ("\\\\gg", "\\gamma"),
    ("\\\\gd", "\\delta"),
    ("\\\\ge", "\\epsilon"),
    ("\\\\gz", "\\zeta"),
    ("\\\\gy", "\\eta"),
    ("\\\\gh", "\\theta"),
    ("\\\\gi", "\\iota"),
    ("\\\\gk", "\\kappa"),
    ("\\\\gl", "\\lambda"),
    ("\\\\gm", "\\mu"),
    ("\\\\gn", "\\nu"),
    ("\\\\gc", "\\xi"),
    ("\\\\go", "\\omicron"),
    ("\\\\gp", "\\pi"),
    ("\\\\gr", "\\rho"),
    ("\\\\gs", "\\sigma"),
    ("\\\\gt", "\\tau"),
    ("\\\\gu", "\\upsilon"),
    ("\\\\gf", "\\phi"),
    ("\\\\gx", "\\chi"),
    ("\\\\gq", "\\psi"),
    ("\\\\gw", "\\omega"),
    ("\\\\gA", "A"),
    ("\\\\gB", "B"),
    ("\\\\gG", "\\Gamma"),
    ("\\\\gD", "\\Delta"),
    ("\\\\gE", "E"),
    ("\\\\gZ", "Z"),
    ("\\\\gY", "H"),
    ("\\\\gH", "\\Theta"),
    ("\\\\gI", "I"),
    ("\\\\gK", "\\Kappa"),
    ("\\\\gL", "\\Lambda"),
    ("\\\\gM", "M"),
    ("\\\\gN", "N"),
    ("\\\\gC", "\\Xi"),
    ("\\\\gO", "O"),
    ("\\\\gP", "\\Pi"),
    ("\\\\gR", "P"),
    ("\\\\gS", "\\Sigma"),
    ("\\\\gT", "T"),
    ("\\\\gU", "\\Upsilon"),
    ("\\\\gF", "\\Phi"),
    ("\\\\gX", "X"),
    ("\\\\gQ", "\\Psi"),
    ("\\\\gW", "\\Omega"),
    ("\\\\fn", ""),
]


def mpl_string(value: str) -> str:
    """Takes string with pgplot characters and returns string with characters replaced with matplotlib equivalent.
    Raises NotImplementedError if an unknown pgplot character is used."""
    if "\\" not in value:
        return value

    value = value.replace("\\", "\\\\")

    result = f"${value}$"
    while "\\\\d" in result and "\\\\u" in result:
        d_pos = result.find("\\\\d")
        u_pos = result.find("\\\\u")
        if d_pos < u_pos:
            sx = result[d_pos : u_pos + 3]
            result = result.replace(sx, "_" + sx[3:-3])
        else:
            sx = result[u_pos : d_pos + 3]
            result = result.replace(sx, "^" + sx[3:-3])

    for from_, to in _pgplot_to_mpl_chars:
        result = result.replace(from_, to)

    if "\\\\" in result:
        logger.debug(f"Unknown pgplot character in string: {result}")

    return result


styles = {
    "solid": "solid",
    "dashed": "dashed",
    "dash_dot": "dashdot",
    "dotted": "dotted",
    "dash_dot3": "dashdot",  # currently the same as dashdot
    "1": "solid",
    "2": "dashed",
    "3": "dashdot",
    "4": "dotted",
    "5": "dashdot",
}

fills = {
    "solid_fill": "full",
    "no_fill": "none",
    "hatched": "full",
    "cross_hatched": "full",
    "1": "full",
    "2": "none",
    "3": "full",
    "4": "full",
}

# Dictionary with pgplot symbol strings as keys and corresponding matplotlib symbol strings as values
symbols = {
    "do_not_draw": "",
    "square": "s",  # no fill
    "dot": ".",
    "plus": "+",
    "times": (6, 2, 0),
    "circle": "$\\circ$",
    "x": "x",
    "x_symbol": "x",
    "triangle": "^",  # no fill
    "circle_plus": "$\\oplus$",
    "circle_dot": "$\\odot$",
    "square_concave": (4, 1, 45),
    "diamond": "d",  # no fill
    "star5": "*",  # no fill
    "triangle_filled": "^",  # fill
    "red_cross": "P",
    "star_of_david": (6, 1, 0),
    "square_filled": "s",  # fill
    "circle_filled": "o",  # fill
    "star5_filled": "*",  # fill
    "0": "s",  # no fill
    "1": ".",
    "2": "+",
    "3": (6, 2, 0),
    "4": "$\\circ$",
    "5": "x",
    "6": "s",
    "7": "^",  # no fill
    "8": "$\\oplus$",
    "9": "$\\odot$",
    "10": (4, 1, 45),
    "11": "d",  # no fill
    "12": "*",  # no fill
    "13": "^",  # fill
    "14": "P",
    "15": (6, 1, 0),
    "16": "s",  # fill
    "17": "o",  # fill
    "18": "*",  # fill
    "-1": ",",
    "-2": ",",
    "-3": "(3,0,0)",
    "-4": "(4,0,0)",
    "-5": "(5,0,0)",
    "-6": "(6,0,0)",
    "-7": "(7,0,0)",
    "-8": "(8,0,0)",
    "-9": "(9,0,0)",
    "-10": "(10,0,0)",
    "-11": "(11,0,0)",
    "-12": "(12,0,0)",
}
