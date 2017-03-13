"""Visualization tools independent of OpenCV and available headless."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def apply_colormap(image, vmin=None, vmax=None, cmap='viridis'):
    """Apply a matplotlib colormap to an image."""
    image = image.astype("float64")  # Returns a copy.
    # Normalization.
    imin, imax = np.min(image), np.max(image)
    if vmin is not None:
        imin = float(vmin)
        image = np.clip(image, vmin, sys.float_info.max)
    if vmax is not None:
        imax = float(vmax)
        image = np.clip(image, -sys.float_info.max, vmax)
    image -= imin
    image /= (imax - imin)
    # Visualization.
    cmap_ = plt.get_cmap(cmap)
    vis = cmap_(image, bytes=True)
    return vis
