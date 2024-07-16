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
    (" ", r"\ "),
    ("%", r"\%"),
    (r"\\(2265)", r"\partial"),
    (r"\\ga", r"\alpha"),
    (r"\\gb", r"\beta"),
    (r"\\gg", r"\gamma"),
    (r"\\gd", r"\delta"),
    (r"\\ge", r"\epsilon"),
    (r"\\gz", r"\zeta"),
    (r"\\gy", r"\eta"),
    (r"\\gh", r"\theta"),
    (r"\\gi", r"\iota"),
    (r"\\gk", r"\kappa"),
    (r"\\gl", r"\lambda"),
    (r"\\gm", r"\mu"),
    (r"\\gn", r"\nu"),
    (r"\\gc", r"\xi"),
    (r"\\go", r"\omicron"),
    (r"\\gp", r"\pi"),
    (r"\\gr", r"\rho"),
    (r"\\gs", r"\sigma"),
    (r"\\gt", r"\tau"),
    (r"\\gu", r"\upsilon"),
    (r"\\gf", r"\phi"),
    (r"\\gx", r"\chi"),
    (r"\\gq", r"\psi"),
    (r"\\gw", r"\omega"),
    (r"\\gA", "A"),
    (r"\\gB", "B"),
    (r"\\gG", r"\Gamma"),
    (r"\\gD", r"\Delta"),
    (r"\\gE", "E"),
    (r"\\gZ", "Z"),
    (r"\\gY", "H"),
    (r"\\gH", r"\Theta"),
    (r"\\gI", "I"),
    (r"\\gK", r"\Kappa"),
    (r"\\gL", r"\Lambda"),
    (r"\\gM", "M"),
    (r"\\gN", "N"),
    (r"\\gC", r"\Xi"),
    (r"\\gO", "O"),
    (r"\\gP", r"\Pi"),
    (r"\\gR", "P"),
    (r"\\gS", r"\Sigma"),
    (r"\\gT", "T"),
    (r"\\gU", r"\Upsilon"),
    (r"\\gF", r"\Phi"),
    (r"\\gX", "X"),
    (r"\\gQ", r"\Psi"),
    (r"\\gW", r"\Omega"),
    (r"\\fn", ""),
]


def mpl_string(value: str) -> str:
    """Takes string with pgplot characters and returns string with characters replaced with matplotlib equivalent."""
    backslash = "\\"
    if backslash not in value:
        return value

    value = value.replace(backslash, r"\\")

    result = f"${value}$"
    while r"\\d" in result and r"\\u" in result:
        d_pos = result.find(r"\\d")
        u_pos = result.find(r"\\u")
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


def mathjax_string(value: str) -> str:
    """
    Takes string with pgplot characters and returns string with characters replaced with MathJax equivalent.
    """
    res = mpl_string(value)
    if res.startswith("$") and res.endswith("$"):
        # MathJAX strings are $$ ... $$ instead of just $ ... $
        return f"${res}$"
    return res


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
