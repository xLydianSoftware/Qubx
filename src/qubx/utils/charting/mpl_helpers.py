"""
Misc graphics handy utilitites to be used in interactive analysis
"""

import colorsys
import datetime

import matplotlib
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as st
from cycler import cycler
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.dates import date2num, num2date
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from qubx.utils.misc import Struct
from qubx.utils.time import infer_series_frequency

DARK_MATLPLOT_THEME = [
    ("backend", "module://matplotlib_inline.backend_inline"),
    ("interactive", True),
    ("lines.color", "#5050f0"),
    ("text.color", "#d0d0d0"),
    ("axes.facecolor", "#000000"),
    ("axes.edgecolor", "#404040"),
    ("axes.grid", True),
    ("axes.labelsize", "large"),
    ("axes.labelcolor", "green"),
    (
        "axes.prop_cycle",
        cycler(
            "color",
            [
                "#08F7FE",
                "#00ff41",
                "#FE53BB",
                "#F5D300",
                "#449AcD",
                "g",
                "#f62841",
                "y",
                "#088487",
                "#E24A33",
                "#f01010",
            ],
        ),
    ),
    ("legend.fontsize", "small"),
    ("legend.fancybox", False),
    ("legend.edgecolor", "#305030"),
    ("legend.shadow", False),
    ("lines.antialiased", True),
    ("lines.linewidth", 0.8),  # reduced line width
    ("patch.linewidth", 0.5),
    ("patch.antialiased", True),
    ("xtick.color", "#909090"),
    ("ytick.color", "#909090"),
    ("xtick.labelsize", "large"),
    ("ytick.labelsize", "large"),
    ("grid.color", "#404040"),
    ("grid.linestyle", "--"),
    ("grid.linewidth", 0.5),
    ("grid.alpha", 0.8),
    ("figure.figsize", [12.0, 5.0]),
    ("figure.dpi", 80.0),
    ("figure.facecolor", "#050505"),
    ("figure.edgecolor", (1, 1, 1, 0)),
    ("figure.subplot.bottom", 0.125),
    ("savefig.facecolor", "#000000"),
]

LIGHT_MATPLOT_THEME = [
    ("backend", "module://matplotlib_inline.backend_inline"),
    ("interactive", True),
    ("lines.color", "#101010"),
    ("text.color", "#303030"),
    ("lines.antialiased", True),
    ("lines.linewidth", 1),
    ("patch.linewidth", 0.5),
    ("patch.facecolor", "#348ABD"),
    ("patch.edgecolor", "#eeeeee"),
    ("patch.antialiased", True),
    ("axes.facecolor", "#fafafa"),
    ("axes.edgecolor", "#d0d0d0"),
    ("axes.linewidth", 1),
    ("axes.titlesize", "x-large"),
    ("axes.labelsize", "large"),
    ("axes.labelcolor", "#555555"),
    ("axes.axisbelow", True),
    ("axes.grid", True),
    (
        "axes.prop_cycle",
        cycler(
            "color",
            [
                "#6792E0",
                "#27ae60",
                "#c44e52",
                "#975CC3",
                "#ff914d",
                "#77BEDB",
                "#303030",
                "#4168B7",
                "#93B851",
                "#e74c3c",
                "#bc89e0",
                "#ff711a",
                "#3498db",
                "#6C7A89",
            ],
        ),
    ),
    ("legend.fontsize", "small"),
    ("legend.fancybox", False),
    ("xtick.color", "#707070"),
    ("ytick.color", "#707070"),
    ("grid.color", "#606060"),
    ("grid.linestyle", "--"),
    ("grid.linewidth", 0.5),
    ("grid.alpha", 0.3),
    ("figure.figsize", [8.0, 5.0]),
    ("figure.dpi", 80.0),
    ("figure.facecolor", "#ffffff"),
    ("figure.edgecolor", "#ffffff"),
    ("figure.subplot.bottom", 0.1),
]


def fig(w=16, h=5, dpi=96, facecolor=None, edgecolor=None, num=None):
    """
    Simple helper for creating figure
    """
    return plt.figure(num=num, figsize=(w, h), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)


def subplot(shape, loc, rowspan=2, colspan=1):
    """
    Some handy grid splitting for plots. Example for 2x2:

    >>> subplot(22, 1); plt.plot([-1,2,-3])
    >>> subplot(22, 2); plt.plot([1,2,3])
    >>> subplot(22, 3); plt.plot([1,2,3])
    >>> subplot(22, 4); plt.plot([3,-2,1])

    same as following

    >>> subplot((2,2), (0,0)); plt.plot([-1,2,-3])
    >>> subplot((2,2), (0,1)); plt.plot([1,2,3])
    >>> subplot((2,2), (1,0)); plt.plot([1,2,3])
    >>> subplot((2,2), (1,1)); plt.plot([3,-2,1])

    :param shape: scalar (like matlab subplot) or tuple
    :param loc: scalar (like matlab subplot) or tuple
    :param rowspan: rows spanned
    :param colspan: columns spanned
    """
    isscalar = lambda x: not isinstance(x, (list, tuple, dict, np.ndarray))

    if isscalar(shape):
        if 0 < shape < 100:
            shape = (max(shape // 10, 1), max(shape % 10, 1))
        else:
            raise ValueError("Wrong scalar value for shape. It should be in range (1...99)")

    if isscalar(loc):
        nm = max(shape[0], 1) * max(shape[1], 1)
        if 0 < loc <= nm:
            x = (loc - 1) // shape[1]
            y = loc - 1 - shape[1] * x
            loc = (x, y)
        else:
            raise ValueError("Wrong scalar value for location. It should be in range (1...%d)" % nm)

    return plt.subplot2grid(shape, loc=loc, rowspan=rowspan, colspan=colspan)


def sbp(shape, loc, r=1, c=1):
    """
    Just shortcut for subplot(...) function

    :param shape: scalar (like matlab subplot) or tuple
    :param loc: scalar (like matlab subplot) or tuple
    :param r: rows spanned
    :param c: columns spanned
    :return:
    """
    return subplot(shape, loc, rowspan=r, colspan=c)


def vline(ax, x, c, lw=1, ls="--"):
    x = pd.to_datetime(x) if isinstance(x, str) else x
    if not isinstance(ax, (list, tuple)):
        ax = [ax]
    for a in ax:
        a.axvline(x, 0, 1, c=c, lw=1, linestyle=ls)


def hline(*zs, mirror=True):
    [plt.axhline(z, ls="--", c="r", lw=0.5) for z in zs]
    if mirror:
        [plt.axhline(-z, ls="--", c="r", lw=0.5) for z in zs]


def ellips(ax, x, y, c="r", r=2.5, lw=2, ls="-"):
    """
    Draw ellips annotation on specified plot at (x,y) point
    """
    from matplotlib.patches import Ellipse

    x = pd.to_datetime(x) if isinstance(x, str) else x
    w, h = (r, r) if np.isscalar(r) else (r[0], r[1])
    ax.add_artist(Ellipse(xy=[x, y], width=w, height=h, angle=0, fill=False, color=c, lw=lw, ls=ls))


def set_mpl_theme(theme: str):
    import plotly.io as pio

    if "dark" in theme.lower():
        pio.templates.default = "plotly_dark"

        # fmt: off
        pio.templates[pio.templates.default]["layout"]["plot_bgcolor"] = "rgb(5,5,5)"
        pio.templates[pio.templates.default]["layout"]["paper_bgcolor"] = "rgb(5,5,5)"
        pio.templates[pio.templates.default]["layout"]["xaxis"]["gridcolor"] = "#171717"
        pio.templates[pio.templates.default]["layout"]["yaxis"]["gridcolor"] = "#171717"
        pio.templates[pio.templates.default]["layout"]["font"]["color"] = "#d0d0d0"
        pio.templates[pio.templates.default]["layout"]["colorway"] = [
            "#08F7FE", "#00ff41", "#FE53BB", "#F5D300", "#449AcD", 
            "green", "#f62841", "yellow", "#088487", "#E24A33", "#f01010",
        ]
        # fmt: on

        for k, v in DARK_MATLPLOT_THEME:
            matplotlib.rcParams[k] = v

    elif "light" in theme.lower():
        pio.templates.default = "plotly_white"
        for k, v in LIGHT_MATPLOT_THEME:
            matplotlib.rcParams[k] = v


def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:  # noqa: E722
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_day_summary_oclh(ax, quotes, ticksize=3, colorup="k", colordown="r"):
    """Plots day summary

        Represent the time, open, close, high, low as a vertical line
        ranging from low to high.  The left tick is the open and the right
        tick is the close.

    Parameters
    ----------
    ax : `Axes`
        an `Axes` instance to plot to
    quotes : sequence of (time, open, close, high, low, ...) sequences
        data to plot.  time must be in float date format - see date2num
    ticksize : int
        open/close tick marker in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open

    Returns
    -------
    lines : list
        list of tuples of the lines added (one tuple per quote)
    """
    return _plot_day_summary(ax, quotes, ticksize=ticksize, colorup=colorup, colordown=colordown, ochl=True)


def plot_day_summary_ohlc(ax, quotes, ticksize=3, colorup="k", colordown="r"):
    """Plots day summary

        Represent the time, open, high, low, close as a vertical line
        ranging from low to high.  The left tick is the open and the right
        tick is the close.

    Parameters
    ----------
    ax : `Axes`
        an `Axes` instance to plot to
    quotes : sequence of (time, open, high, low, close, ...) sequences
        data to plot.  time must be in float date format - see date2num
    ticksize : int
        open/close tick marker in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open

    Returns
    -------
    lines : list
        list of tuples of the lines added (one tuple per quote)
    """
    return _plot_day_summary(ax, quotes, ticksize=ticksize, colorup=colorup, colordown=colordown, ochl=False)


def _plot_day_summary(ax, quotes, ticksize=3, colorup="k", colordown="r", ochl=True):
    """Plots day summary


        Represent the time, open, high, low, close as a vertical line
        ranging from low to high.  The left tick is the open and the right
        tick is the close.

    Parameters
    ----------
    ax : `Axes`
        an `Axes` instance to plot to
    quotes : sequence of quote sequences
        data to plot.  time must be in float date format - see date2num
        (time, open, high, low, close, ...) vs
        (time, open, close, high, low, ...)
        set by `ochl`
    ticksize : int
        open/close tick marker in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    ochl: bool
        argument to select between ochl and ohlc ordering of quotes

    Returns
    -------
    lines : list
        list of tuples of the lines added (one tuple per quote)
    """
    # unfortunately this has a different return type than plot_day_summary2_*
    lines = []
    for q in quotes:
        if ochl:
            t, open, close, high, low = q[:5]
        else:
            t, open, high, low, close = q[:5]

        if close >= open:
            color = colorup
        else:
            color = colordown

        vline = Line2D(
            xdata=(t, t),
            ydata=(low, high),
            color=color,
            antialiased=False,  # no need to antialias vert lines
        )

        oline = Line2D(
            xdata=(t, t),
            ydata=(open, open),
            color=color,
            antialiased=False,
            marker=TICKLEFT,
            markersize=ticksize,
        )

        cline = Line2D(
            xdata=(t, t), ydata=(close, close), color=color, antialiased=False, markersize=ticksize, marker=TICKRIGHT
        )

        lines.extend((vline, oline, cline))
        ax.add_line(vline)
        ax.add_line(oline)
        ax.add_line(cline)

    ax.autoscale_view()

    return lines


def candlestick_ochl(ax, quotes, width=0.2, colorup="k", colordown="r", alpha=1.0):
    """
    Plot the time, open, close, high, low as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    quotes : sequence of (time, open, close, high, low, ...) sequences
        As long as the first 5 elements are these values,
        the record can be as long as you want (e.g., it may store volume).

        time must be in float days format - see date2num

    width : float
        fraction of a day for the rectangle width
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level

    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added

    """
    return _candlestick(ax, quotes, width=width, colorup=colorup, colordown=colordown, alpha=alpha, ochl=True)


def candlestick_ohlc(ax, quotes, width=0.2, colorup="k", colordown="r", alpha=1.0):
    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    quotes : sequence of (time, open, high, low, close, ...) sequences
        As long as the first 5 elements are these values,
        the record can be as long as you want (e.g., it may store volume).

        time must be in float days format - see date2num

    width : float
        fraction of a day for the rectangle width
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level

    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added

    """
    return _candlestick(ax, quotes, width=width, colorup=colorup, colordown=colordown, alpha=alpha, ochl=False)


def _candlestick(ax, quotes, width=0.2, colorup="k", colordown="r", alpha=1.0, ochl=True):
    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    quotes : sequence of quote sequences
        data to plot.  time must be in float date format - see date2num
        (time, open, high, low, close, ...) vs
        (time, open, close, high, low, ...)
        set by `ochl`
    width : float
        fraction of a day for the rectangle width
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level
    ochl: bool
        argument to select between ochl and ohlc ordering of quotes

    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added

    """

    OFFSET = width / 2.0

    lines = []
    patches = []
    for q in quotes:
        if ochl:
            t, open, close, high, low = q[:5]
        else:
            t, open, high, low, close = q[:5]

        if close >= open:
            color = colorup
            edgecolor = adjust_lightness(color, 1.5)
            lower = open
            height = close - open
        else:
            color = colordown
            edgecolor = adjust_lightness(color, 1.2)
            lower = close
            height = open - close

        vline = Line2D(
            xdata=(t, t),
            ydata=(low, high),
            # color=adjust_lightness(color, 1.3),
            color=color,
            linewidth=1,
            antialiased=True,
        )

        rect = Rectangle(
            xy=(t - OFFSET, lower),
            width=width,
            height=height,
            facecolor=color,
            # edgecolor=color,
            # facecolor=adjust_lightness(color, 1.3),
            edgecolor=edgecolor,
            lw=0.75,
        )
        rect.set_alpha(alpha)

        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
    ax.autoscale_view()

    return lines, patches


def _check_input(opens, closes, highs, lows, miss=-1):
    """Checks that *opens*, *highs*, *lows* and *closes* have the same length.
    NOTE: this code assumes if any value open, high, low, close is
    missing (*-1*) they all are missing

    Parameters
    ----------
    ax : `Axes` an Axes instance to plot to
    opens : sequence of opening values
    highs : sequence of high values
    lows : sequence of low values
    closes : sequence of closing values
    miss : identifier of the missing data

    Raises
    ------
    ValueError
        if the input sequences don't have the same length
    """

    def _missing(sequence, miss=-1):
        """Returns the index in *sequence* of the missing data, identified by
        *miss*

        Parameters
        ----------
        sequence :
            sequence to evaluate
        miss :
            identifier of the missing data

        Returns
        -------
        where_miss: numpy.ndarray
            indices of the missing data
        """
        return np.where(np.array(sequence) == miss)[0]

    same_length = len(opens) == len(highs) == len(lows) == len(closes)
    _missopens = _missing(opens)
    same_missing = (
        (_missopens == _missing(highs)).all()
        and (_missopens == _missing(lows)).all()
        and (_missopens == _missing(closes)).all()
    )

    if not (same_length and same_missing):
        msg = (
            "*opens*, *highs*, *lows* and *closes* must have the same"
            " length. NOTE: this code assumes if any value open, high,"
            " low, close is missing (*-1*) they all must be missing."
        )
        raise ValueError(msg)


def plot_day_summary2_ochl(ax, opens, closes, highs, lows, ticksize=4, colorup="k", colordown="r"):
    """Represent the time, open, close, high, low,  as a vertical line
    ranging from low to high.  The left tick is the open and the right
    tick is the close.

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence of opening values
    closes : sequence of closing values
    highs : sequence of high values
    lows : sequence of low values
    ticksize : int
        size of open and close ticks in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
         the color of the lines where close <  open

    Returns
    -------
    ret : list
        a list of lines added to the axes
    """

    return plot_day_summary2_ohlc(ax, opens, highs, lows, closes, ticksize, colorup, colordown)


def plot_day_summary2_ohlc(ax, opens, highs, lows, closes, ticksize=4, colorup="k", colordown="r"):
    """Represent the time, open, high, low, close as a vertical line
    ranging from low to high.  The left tick is the open and the right
    tick is the close.
    *opens*, *highs*, *lows* and *closes* must have the same length.
    NOTE: this code assumes if any value open, high, low, close is
    missing (*-1*) they all are missing

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        sequence of opening values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    closes : sequence
        sequence of closing values
    ticksize : int
        size of open and close ticks in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
         the color of the lines where close <  open

    Returns
    -------
    ret : list
        a list of lines added to the axes
    """

    _check_input(opens, highs, lows, closes)

    rangeSegments = [((i, low), (i, high)) for i, low, high in zip(range(len(lows)), lows, highs) if low != -1]

    # the ticks will be from ticksize to 0 in points at the origin and
    # we'll translate these to the i, close location
    openSegments = [((-ticksize, 0), (0, 0))]

    # the ticks will be from 0 to ticksize in points at the origin and
    # we'll translate these to the i, close location
    closeSegments = [((0, 0), (ticksize, 0))]

    offsetsOpen = [(i, open) for i, open in zip(range(len(opens)), opens) if open != -1]

    offsetsClose = [(i, close) for i, close in zip(range(len(closes)), closes) if close != -1]

    scale = ax.figure.dpi * (1.0 / 72.0)

    tickTransform = Affine2D().scale(scale, 0.0)

    colorup = mcolors.to_rgba(colorup)
    colordown = mcolors.to_rgba(colordown)
    colord = {True: colorup, False: colordown}
    colors = [colord[open < close] for open, close in zip(opens, closes) if open != -1 and close != -1]

    useAA = (0,)  # use tuple here
    lw = (1,)  # and here
    rangeCollection = LineCollection(
        rangeSegments,
        colors=colors,
        linewidths=lw,
        antialiaseds=useAA,
    )

    openCollection = LineCollection(
        openSegments,
        colors=colors,
        antialiaseds=useAA,
        linewidths=lw,
        offsets=offsetsOpen,
        transOffset=ax.transData,
    )
    openCollection.set_transform(tickTransform)

    closeCollection = LineCollection(
        closeSegments,
        colors=colors,
        antialiaseds=useAA,
        linewidths=lw,
        offsets=offsetsClose,
        transOffset=ax.transData,
    )
    closeCollection.set_transform(tickTransform)

    minpy, maxx = (0, len(rangeSegments))
    miny = min([low for low in lows if low != -1])
    maxy = max([high for high in highs if high != -1])
    corners = (minpy, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()

    # add these last
    ax.add_collection(rangeCollection)
    ax.add_collection(openCollection)
    ax.add_collection(closeCollection)
    return rangeCollection, openCollection, closeCollection


def candlestick2_ochl(ax, opens, closes, highs, lows, width=4, colorup="k", colordown="r", alpha=0.75):
    """Represent the open, close as a bar line and high low range as a
    vertical line.

    Preserves the original argument order.


    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        sequence of opening values
    closes : sequence
        sequence of closing values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    width : int
        size of open and close ticks in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency

    Returns
    -------
    ret : tuple
        (lineCollection, barCollection)
    """

    return candlestick2_ohlc(
        ax, opens, highs, lows, closes, width=width, colorup=colorup, colordown=colordown, alpha=alpha
    )


def candlestick2_ohlc(ax, opens, highs, lows, closes, width=4, colorup="k", colordown="r", alpha=0.75):
    """Represent the open, close as a bar line and high low range as a
    vertical line.

    NOTE: this code assumes if any value open, low, high, close is
    missing they all are missing


    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        sequence of opening values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    closes : sequence
        sequence of closing values
    width : int
        size of open and close ticks in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency

    Returns
    -------
    ret : tuple
        (lineCollection, barCollection)
    """

    _check_input(opens, highs, lows, closes)

    delta = width / 2.0
    barVerts = [
        ((i - delta, open), (i - delta, close), (i + delta, close), (i + delta, open))
        for i, open, close in zip(range(len(opens)), opens, closes)
        if open != -1 and close != -1
    ]

    rangeSegments = [((i, low), (i, high)) for i, low, high in zip(range(len(lows)), lows, highs) if low != -1]

    colorup = mcolors.to_rgba(colorup, alpha)
    colordown = mcolors.to_rgba(colordown, alpha)
    colord = {True: colorup, False: colordown}
    colors = [colord[open < close] for open, close in zip(opens, closes) if open != -1 and close != -1]

    useAA = (0,)  # use tuple here
    lw = (0.5,)  # and here
    rangeCollection = LineCollection(
        rangeSegments,
        colors=colors,
        linewidths=lw,
        antialiaseds=useAA,
    )

    barCollection = PolyCollection(
        barVerts,
        facecolors=colors,
        edgecolors=colors,
        antialiaseds=useAA,
        linewidths=lw,
    )

    minx, maxx = 0, len(rangeSegments)
    miny = min([low for low in lows if low != -1])
    maxy = max([high for high in highs if high != -1])

    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()

    # add these last
    ax.add_collection(rangeCollection)
    ax.add_collection(barCollection)
    return rangeCollection, barCollection


def volume_overlay(ax, opens, closes, volumes, colorup="g", colordown="r", width=4, alpha=1.0):
    """Add a volume overlay to the current axes.  The opens and closes
    are used to determine the color of the bar.  -1 is missing.  If a
    value is missing on one it must be missing on all

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        a sequence of opens
    closes : sequence
        a sequence of closes
    volumes : sequence
        a sequence of volumes
    width : int
        the bar width in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency

    Returns
    -------
    ret : `barCollection`
        The `barrCollection` added to the axes

    """

    colorup = mcolors.to_rgba(colorup, alpha)
    colordown = mcolors.to_rgba(colordown, alpha)
    colord = {True: colorup, False: colordown}
    colors = [colord[open < close] for open, close in zip(opens, closes) if open != -1 and close != -1]

    delta = width / 2.0
    bars = [((i - delta, 0), (i - delta, v), (i + delta, v), (i + delta, 0)) for i, v in enumerate(volumes) if v != -1]

    barCollection = PolyCollection(
        bars,
        facecolors=colors,
        edgecolors=((0, 0, 0, 1),),
        antialiaseds=(0,),
        linewidths=(0.5,),
    )

    ax.add_collection(barCollection)
    corners = (0, 0), (len(bars), max(volumes))
    ax.update_datalim(corners)
    ax.autoscale_view()

    # add these last
    return barCollection


def volume_overlay2(ax, closes, volumes, colorup="g", colordown="r", width=4, alpha=1.0):
    """
    Add a volume overlay to the current axes.  The closes are used to
    determine the color of the bar.  -1 is missing.  If a value is
    missing on one it must be missing on all

    nb: first point is not displayed - it is used only for choosing the
    right color


    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    closes : sequence
        a sequence of closes
    volumes : sequence
        a sequence of volumes
    width : int
        the bar width in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency

    Returns
    -------
    ret : `barCollection`
        The `barrCollection` added to the axes

    """

    return volume_overlay(ax, closes[:-1], closes[1:], volumes[1:], colorup, colordown, width, alpha)


def volume_overlay3(ax, quotes, colorup="g", colordown="r", width=4, alpha=1.0):
    """Add a volume overlay to the current axes.  quotes is a list of (d,
    open, high, low, close, volume) and close-open is used to
    determine the color of the bar

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    quotes : sequence of (time, open, high, low, close, ...) sequences
        data to plot.  time must be in float date format - see date2num
    width : int
        the bar width in points
    colorup : color
        the color of the lines where close1 >= close0
    colordown : color
        the color of the lines where close1 <  close0
    alpha : float
         bar transparency

    Returns
    -------
    ret : `barCollection`
        The `barrCollection` added to the axes


    """

    colorup = mcolors.to_rgba(colorup, alpha)
    colordown = mcolors.to_rgba(colordown, alpha)
    colord = {True: colorup, False: colordown}

    dates, opens, highs, lows, closes, volumes = list(zip(*quotes))
    colors = [
        colord[close1 >= close0] for close0, close1 in zip(closes[:-1], closes[1:]) if close0 != -1 and close1 != -1
    ]
    colors.insert(0, colord[closes[0] >= opens[0]])

    right = width / 2.0
    left = -width / 2.0

    bars = [((left, 0), (left, volume), (right, volume), (right, 0)) for d, open, high, low, close, volume in quotes]

    sx = ax.figure.dpi * (1.0 / 72.0)  # scale for points
    sy = ax.bbox.height / ax.viewLim.height

    barTransform = Affine2D().scale(sx, sy)

    dates = [d for d, open, high, low, close, volume in quotes]
    offsetsBars = [(d, 0) for d in dates]

    useAA = (0,)  # use tuple here
    lw = (0.5,)  # and here
    barCollection = PolyCollection(
        bars,
        facecolors=colors,
        edgecolors=((0, 0, 0, 1),),
        antialiaseds=useAA,
        linewidths=lw,
        offsets=offsetsBars,
        transOffset=ax.transData,
    )
    barCollection.set_transform(barTransform)

    minpy, maxx = (min(dates), max(dates))
    miny = 0
    maxy = max([volume for d, open, high, low, close, volume in quotes])
    corners = (minpy, miny), (maxx, maxy)
    ax.update_datalim(corners)
    # print 'datalim', ax.dataLim.bounds
    # print 'viewlim', ax.viewLim.bounds

    ax.add_collection(barCollection)
    ax.autoscale_view()

    return barCollection


def index_bar(ax, vals, facecolor="b", edgecolor="l", width=4, alpha=1.0):
    """Add a bar collection graph with height vals (-1 is missing).

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    vals : sequence
        a sequence of values
    facecolor : color
        the color of the bar face
    edgecolor : color
        the color of the bar edges
    width : int
        the bar width in points
    alpha : float
       bar transparency

    Returns
    -------
    ret : `barCollection`
        The `barrCollection` added to the axes

    """

    facecolors = (mcolors.to_rgba(facecolor, alpha),)
    edgecolors = (mcolors.to_rgba(edgecolor, alpha),)

    right = width / 2.0
    left = -width / 2.0

    bars = [((left, 0), (left, v), (right, v), (right, 0)) for v in vals if v != -1]

    sx = ax.figure.dpi * (1.0 / 72.0)  # scale for points
    sy = ax.bbox.height / ax.viewLim.height

    barTransform = Affine2D().scale(sx, sy)

    offsetsBars = [(i, 0) for i, v in enumerate(vals) if v != -1]

    barCollection = PolyCollection(
        bars,
        facecolors=facecolors,
        edgecolors=edgecolors,
        antialiaseds=(0,),
        linewidths=(0.5,),
        offsets=offsetsBars,
        transOffset=ax.transData,
    )
    barCollection.set_transform(barTransform)

    minpy, maxx = (0, len(offsetsBars))
    miny = 0
    maxy = max([v for v in vals if v != -1])
    corners = (minpy, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()

    # add these last
    ax.add_collection(barCollection)
    return barCollection


def _to_mdate_index(x):
    return date2num(x.index.to_pydatetime()), x


def ohlc_plot(ohlc: pd.DataFrame, width=0, colorup="#209040", colordown="#e02020", fmt=None, autofmt=False):
    """
    Plot OHLC data frame

    :param ohlc: index, 'open', 'high', 'low', 'close' columns
    :param width: used bar width
    :param colorup: color of growing bar
    :param colordown: color of declining bar
    :param autofmt: true if needed to aoutoformatting time labels
    :param fmt: format string for timescale
    :return: axis
    """
    ohlc_f = ohlc.filter(["open", "high", "low", "close"])
    if ohlc_f.shape[1] != 4:
        raise ValueError("DataFrame ohlc must contain 'open', 'high', 'low', 'close' columns !")

    _freq = pd.Timedelta(infer_series_frequency(ohlc_f))

    # customization of the axis
    f = plt.gcf()
    ax = plt.gca()
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="both", direction="out", width=2, length=8, labelsize=8, pad=8)

    tms, _ = _to_mdate_index(ohlc_f)
    t_data = np.vstack((np.array(tms), ohlc_f.values.T)).T

    # auto width
    if width <= 0:
        width = max(1, _freq.total_seconds() * 0.7) / 24 / 60 / 60

    reshaped_data = np.hstack((np.reshape(t_data[:, 0], (-1, 1)), t_data[:, 1:]))
    candlestick_ohlc(ax, reshaped_data, width=width, colorup=colorup, colordown=colordown)

    is_eod = _freq >= datetime.timedelta(1)
    if is_eod:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(15))
        fmt = "%d-%b-%y" if fmt is None else fmt
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(20))
        fmt = "%H:%M" if fmt is None else fmt

    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels([datetime.date.strftime(num2date(x), fmt) for x in ax.get_xticks()])

    if autofmt:
        f.autofmt_xdate()

    # - don't want to see grid lines on the top
    ax.set_axisbelow(True)

    return ax


def plot_trends(trends: pd.DataFrame | Struct, uc="w--", dc="m--", lw=2, ms=6, fmt="%H:%M"):
    """
    Plot find_movements function output as trend lines on chart

    >>> from qube.quantitative.ta.swings.swings_splitter import find_movements
    >>>
    >>> tx = pd.Series(np.random.randn(500).cumsum() + 100, index=pd.date_range('2000-01-01', periods=500))
    >>> trends = find_movements(tx, np.inf, use_prev_movement_size_for_percentage=False,
    >>>                    pcntg=0.02,
    >>>                    t_window=np.inf, drop_weekends_crossings=False,
    >>>                    drop_out_of_market=False, result_as_frame=True, silent=True)
    >>> plot_trends(trends)

    :param trends: find_movements output
    :param uc: up trends line spec ('w--')
    :param dc: down trends line spec ('c--')
    :param lw: line weight (0.7)
    :param ms: trends reversals marker size (5)
    :param fmt: time format (default is '%H:%M')
    """
    if isinstance(trends, Struct) and hasattr(trends, "trends"):
        trends = trends.trends

    if isinstance(trends, pd.DataFrame):
        if not trends.empty:
            u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
            plt.plot([u.index, u.end], [u.start_price, u.end_price], uc, lw=lw, marker="o", markersize=ms)
            plt.plot([d.index, d.end], [d.start_price, d.end_price], dc, lw=lw, marker="o", markersize=ms)

            import datetime

            from matplotlib.dates import num2date

            ax = plt.gca()
            ax.set_xticks(ax.get_xticks(), labels=[datetime.date.strftime(num2date(x), fmt) for x in ax.get_xticks()])
    else:
        raise ValueError("trends must be a DataFrame or Struct with 'trends' attribute")


def plot_quantiles(data, top_n=10, bottom_n=10, title=None, ylabel=None, xlabel="Items", 
                   figsize=(16, 10), positive_color='#2E8B57', negative_color='#DC143C',
                   value_formatter=None, label_transformer=None, show_values=True,
                   zero_line=True, positive_label=None, negative_label=None):
    """
    Plot top and bottom quantiles of data as a bar chart with customizable styling.
    
    Parameters:
    -----------
    data : pd.Series
        Series of values to plot, should be sorted in descending order
    top_n : int, default 10
        Number of highest values to show
    bottom_n : int, default 10
        Number of lowest values to show
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    xlabel : str, default "Items"
        X-axis label
    figsize : tuple, default (16, 10)
        Figure size
    positive_color : str, default '#2E8B57'
        Color for positive values (sea green)
    negative_color : str, default '#DC143C'
        Color for negative values (crimson)
    value_formatter : callable, optional
        Function to format values for display (e.g., lambda x: f'{x:.1f}%')
    label_transformer : callable, optional
        Function to transform index labels (e.g., lambda x: x.replace('USDC', ''))
    show_values : bool, default True
        Whether to show values on bars
    zero_line : bool, default True
        Whether to show horizontal line at zero
    positive_label : str, optional
        Legend label for positive values
    negative_label : str, optional
        Legend label for negative values
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    
    Examples:
    ---------
    # Basic usage
    plot_quantiles(data_series)
    
    # Funding rates example
    plot_quantiles(funding_rates, 
                   title="Annualized Funding Rates",
                   ylabel="Rate (%)", 
                   value_formatter=lambda x: f'{x:.1f}%',
                   label_transformer=lambda x: x.replace('USDC', ''),
                   positive_label="Longs pay Shorts",
                   negative_label="Shorts pay Longs")
    """
    import pandas as pd
    from matplotlib.patches import Patch
    
    # Get top and bottom quantiles
    top_data = data.head(top_n)
    bottom_data = data.tail(bottom_n)
    
    # Combine them
    combined_data = pd.concat([top_data, bottom_data])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colors: positive_color for positive, negative_color for negative
    colors = [positive_color if value >= 0 else negative_color for value in combined_data.values]
    
    # Create bar plot
    bars = ax.bar(range(len(combined_data)), combined_data.values, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=0.5)
    
    # Customize the plot
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    if label_transformer:
        labels = [label_transformer(str(label)) for label in combined_data.index]
    else:
        labels = [str(label) for label in combined_data.index]
    
    ax.set_xticks(range(len(combined_data)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add horizontal line at zero
    if zero_line:
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1.5)
    
    # Add grid
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, color='#666666')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    if show_values:
        if value_formatter is None:
            value_formatter = lambda x: f'{x:.1f}'
            
        for bar, value in zip(bars, combined_data.values):
            height = bar.get_height()
            label_offset = max(abs(height) * 0.02, 5)  # Dynamic offset based on value magnitude
            
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (label_offset if height >= 0 else -label_offset),
                   value_formatter(value), 
                   ha='center', 
                   va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold', 
                   color='white' if abs(height) > max(abs(combined_data.min()), abs(combined_data.max())) * 0.3 else 'black')
    
    # Add legend if labels provided
    if positive_label or negative_label:
        legend_elements = []
        if positive_label:
            legend_elements.append(Patch(facecolor=positive_color, label=positive_label))
        if negative_label:
            legend_elements.append(Patch(facecolor=negative_color, label=negative_label))
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    return fig, ax
