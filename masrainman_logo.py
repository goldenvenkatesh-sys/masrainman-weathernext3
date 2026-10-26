"""
MasRainman logo on every figure render.py saves: the model maps (WeatherNext 3, ECMWF IFS, AIFS), the
rain charts, and the GIF frames made from them.

Importing this module wraps matplotlib's Figure.savefig. Before a figure is saved, the rainbow-cloud
logo (assets/masrainman_logo.png) and the name "MasRainman" are added at its top right. With
bbox_inches="tight" (all of render.py's figures) they sit in a slim strip just above the figure, so they
never cover the map, the titles or the colour bars. Otherwise they go inside the top-right corner.

A figure can instead reserve a header band for them: set fig.masrainman_header = (y_centre, x_right) in
figure fractions and the logo, then the name, are placed inside the figure, ending at x_right and vertically
centred on y_centre, so they sit on the same line as the chart title (used by the MSLP maps).
"""
import os

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = {False: os.path.join(HERE, "assets", "masrainman_logo.png"),        # for light backgrounds
        True: os.path.join(HERE, "assets", "masrainman_logo_dark.png")}    # white lettering, dark backgrounds
HEIGHT_IN = 0.95          # logo height in inches (about 170 px at dpi 180)
TAG = "masrainman_logo"
_images = {}
_savefig = Figure.savefig


def _image(dark, height_px):
    """The logo resized (Lanczos) to the pixel height it will have in the saved file, so it stays sharp."""
    key = (dark, height_px)
    if key not in _images:
        im = Image.open(LOGO[dark]).convert("RGBA")
        im = im.resize((round(im.width * height_px / im.height), height_px), Image.LANCZOS)
        _images[key] = np.asarray(im)
    return _images[key]


def add_logo(fig, outside=True, dpi=None):
    """Add the logo and name to `fig` once (again after fig.clf())."""
    if any(a.get_label() == TAG for a in fig.axes):
        return
    r, g, b = fig.get_facecolor()[:3]
    dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
    img = _image(dark, max(24, round(HEIGHT_IN * (dpi or fig.dpi))))
    fw, fh = fig.get_size_inches()
    h = HEIGHT_IN / fh
    w = h * img.shape[1] / img.shape[0] * fh / fw
    header = getattr(fig, "masrainman_header", None)
    if header:                                   # same line as the title, inside the header band
        y_mid, x_right = header
        x0, y0 = x_right - w, y_mid - h * 0.5
    else:
        x0 = 1 - w - 0.004
        y0 = 1.004 if outside else 1 - h - 0.004
    colour = "white" if dark else "#1d63ba"
    if header:                                   # logo first, then the name, ending at x_right
        name = fig.text(x_right, y_mid, "MasRainman", ha="right", va="center", fontsize=19,
                        fontweight="bold", color=colour, gid=TAG)
        name_w = name.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
        x0 = x_right - name_w - 0.008 - w
        ax = fig.add_axes([x0, y0, w, h], label=TAG, zorder=100)
        ax.imshow(img, interpolation="nearest")
        ax.set_axis_off()
        return
    ax = fig.add_axes([x0, y0, w, h], label=TAG, zorder=100)
    ax.imshow(img, interpolation="nearest")
    ax.set_axis_off()
    fig.text(x0 - 0.006, y0 + h * 0.5, "MasRainman", ha="right", va="center", fontsize=17, fontweight="bold",
             color=colour, gid=TAG)


def _savefig_with_logo(self, *args, **kwargs):
    try:
        dpi = kwargs.get("dpi")
        add_logo(self, outside=kwargs.get("bbox_inches") == "tight", dpi=dpi if isinstance(dpi, (int, float)) else None)
    except Exception as exc:          # never lose a map because of the logo
        print(f"(logo not added: {exc})")
    return _savefig(self, *args, **kwargs)


Figure.savefig = _savefig_with_logo
