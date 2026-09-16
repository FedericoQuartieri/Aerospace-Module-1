"""The look shared by the figures of the report.

Fonts match the LaTeX body (Latin Modern), the marks are thin, the grid
recessive.  Two palettes: SERIES for distinct things (one backend against
the other, one component against another), RAMP for an ordered parameter
such as the permeability, light to dark.  Both were validated for
colour-blind readers on a white page.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
MARKERS = ["o", "s", "D", "^", "v"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlesize": 9,
        "axes.titlecolor": INK,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_SECONDARY,
        "pdf.fonttype": 42,
    })


def tidy(ax, title=None):
    if title:
        ax.set_title(title, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def line(ax, x, y, index, label, palette=SERIES, marker=None):
    ax.plot(x, y, color=palette[index], lw=1.3,
            marker=marker or MARKERS[index], ms=4.2, mec="white", mew=0.7,
            label=label, zorder=3)


def slope_guide(ax, x0, y0, order, span=2.5, text=None):
    """A short reference line of the given log-log slope from (x0, y0)."""
    x = [x0, x0 * span]
    y = [y0, y0 * span ** order]
    ax.plot(x, y, color=MUTED, lw=0.8, zorder=2)
    ax.annotate(text or f"slope {order}", xy=(x[1], y[1]), xytext=(4, 0),
                textcoords="offset points", color=MUTED, fontsize=8,
                va="center", ha="left")


def save(fig, path, preview=None):
    """The PDF at `path`, and a PNG to look at: next to it, or at `preview`
    when the PDF goes somewhere a stray image should not end up."""
    preview = preview or path.with_suffix(".png")
    for target in (path, preview):
        target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(preview, dpi=200)
    print(f"written {path} (preview {preview})")
