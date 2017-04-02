"""Visualization tools independent of OpenCV and available headless."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn  # For dynamic colormap generation.


def apply_colormap(image, vmin=None, vmax=None, cmap='viridis', cmap_seed=1):
    """
    Apply a matplotlib colormap to an image.

    This method will preserve the exact image size. `cmap` can be either a
    matplotlib colormap name, a discrete number, or a colormap instance. If it
    is a number, a discrete colormap will be generated based on the HSV
    colorspace. The permutation of colors is random and can be controlled with
    the `cmap_seed`. The state of the RNG is preserved.
    """
    image = image.astype("float64")  # Returns a copy.
    # Normalization.
    if vmin is not None:
        imin = float(vmin)
        image = np.clip(image, vmin, sys.float_info.max)
    else:
        imin = np.min(image)
    if vmax is not None:
        imax = float(vmax)
        image = np.clip(image, -sys.float_info.max, vmax)
    else:
        imax = np.max(image)
    image -= imin
    image /= (imax - imin)
    # Visualization.
    if isinstance(cmap, str):
        cmap_ = plt.get_cmap(cmap)
    elif isinstance(cmap, int):
        # Construct a classification colormap.
        # Preserve RNG state.
        rng_state = np.random.get_state()
        np.random.seed(cmap_seed)
        palette = np.random.permutation(seaborn.husl_palette(n_colors=cmap,
                                                             s=0.9,
                                                             l=0.7
        ))
        np.random.set_state(rng_state)
        # Fix the darkest color for background (can look nasty otherwise).
        darkidx = np.argsort(np.sum(np.square(palette), axis=1))[0]
        darkc = palette[darkidx, :].copy()
        palette[darkidx, :] = palette[0, :]
        palette[0, :] = darkc[:]
        cmap_ = matplotlib.colors.ListedColormap(palette)
    else:
        cmap_ = cmap
    vis = cmap_(image, bytes=True)
    return vis
