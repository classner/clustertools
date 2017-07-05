"""Visualization tools independent of OpenCV and available headless."""
import sys
import colorsys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
try:
    import seaborn  # noqa: E402
except ImportError:
    seaborn = None
import skimage.draw as skdraw
import skimage.transform as sktransform
import PIL
import PIL.ImageDraw as pildraw


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
        if seaborn is not None:
            palette = np.random.permutation(seaborn.husl_palette(n_colors=cmap,
                                                                 s=0.9,
                                                                 l=0.7))
        else:
            palette = np.random.uniform(0., 1., (cmap, 3))
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


def draw_transparent(image, drawing_function, params, opacity):
    """Applies a drawing function in place on the provided image."""
    assert opacity >= 0. and opacity <= 1.
    assert image.ndim in [2, 3]
    assert image.dtype == np.uint8
    overlay = image.copy()
    full_plist = [overlay] + list(params)
    drawing_function(*full_plist)
    image[...] = (image.astype(np.float32) * (1. - opacity) +
                  overlay.astype(np.float32) * opacity).astype(np.uint8)

def draw_line(image,  # pylint: disable=too-many-locals, too-many-arguments
              pt1,
              pt2,
              color,
              thickness,
              dash_length=0):
    """
    Draw a dashed line on the given image.

    Points are length 2 containers with x, y coordinates.
    """
    pt1 = np.asarray(pt1).astype(np.int32)
    pt2 = np.asarray(pt2).astype(np.int32)
    assert image.ndim in [2, 3]
    if image.ndim == 2:
        assert len(color) == 1
    else:
        assert len(color) == image.shape[2]
    assert thickness > 0
    assert dash_length >= 0
    beginpts = []
    endpts = []
    if dash_length > 0:
        rr, cc = skdraw.line(pt1[1], pt1[0],  # pylint: disable=invalid-name
                             pt2[1], pt2[0])
        beginpts.append(np.asarray(pt1))
        drawing = True
        lastp = np.asarray(pt1)
        # Determine beginning and endpoints of line segments.
        for yc, xc in zip(rr, cc):  # pylint: disable=invalid-name
            if np.linalg.norm(lastp - [xc, yc]) >= dash_length:
                currp = np.asarray([xc, yc])
                if drawing:
                    endpts.append(currp)
                else:
                    beginpts.append(currp)
                lastp = currp
                drawing = not drawing
        if len(endpts) < len(beginpts):
            endpts.append(np.asarray([cc[-1], rr[-1]]))
    else:
        beginpts.append(np.asarray(pt1))
        endpts.append(np.asarray(pt2))
    assert len(beginpts) == len(endpts)
    # Draw.
    pilim = PIL.Image.fromarray(image)
    draw = pildraw.Draw(pilim)
    for beginpt, endpt in zip(beginpts, endpts):
        draw.line([tuple(beginpt.astype(np.int32)),
                   tuple(endpt.astype(np.int32))],
                  color,
                  thickness)
    image[...] = pilim

def draw_circle(image,  # pylint: disable=invalid-name
                pt,
                radius,
                color):
    """Draw a circle on an image."""
    assert image.ndim in [2, 3]
    if image.ndim == 2:
        assert len(color) == 1
    else:
        assert len(color) == image.shape[2]
    assert radius > 0
    rr, cc = skdraw.circle(pt[1], pt[0], radius)  # pylint: disable=invalid-name
    image[rr, cc] = color


def visualize_pose(image,  # pylint: disable=dangerous-default-value, too-many-arguments, too-many-locals, too-many-branches
                   pose,
                   line_thickness=3,
                   dash_length=15,
                   opacity=0.6,
                   circle_color=(255, 255, 255),
                   connections=[],
                   lm_region_mapping=None,
                   skip_unconnected_joints=True,
                   scale=1.):
    """Draw a pose on the given image.

    The pose is an numpy array of size R, C, where R>2 and N is the number of
    joints. The first two entries in R are x and y coordinates.
    """
    assert len(circle_color) == 3 or len(circle_color) == pose.shape[1], (
        "You must provide either one 3-tuple as a color or a list of 3-tuples "
        "with the length of the pose joints. Is %d!" % (len(circle_color)))
    if (len(circle_color) == pose.shape[1] and
            isinstance(circle_color[0], tuple)):
        for col in circle_color:
            assert isinstance(col, tuple)
            assert len(col) == 3
        multi_col_mode = True
    else:
        multi_col_mode = False
    image = sktransform.rescale(image, float(scale), preserve_range=True,
                                mode='reflect').astype(np.uint8)
    pose = (pose * scale).astype(np.int32)[:2, :]
    for connection in connections:
        if len(connection) > 3:
            if connection[4] == 2:
                ccol = list(colorsys.rgb_to_hsv(*list(
                    np.array(connection[2]) / 255.)))
                ccol[2] = np.clip(ccol[2] - 0.2, 0., 1.)
                ccol = tuple((np.array(colorsys.hsv_to_rgb(*ccol)) *
                              255.).astype('int'))
            else:
                ccol = connection[2]
        if np.all(pose[0:2, connection[0]] <= 0) or\
           np.all(pose[0:2, connection[1]] <= 0):
            print "Warning: omitting connection with endpoints <= 0!"
            continue
        this_dash_length = int(connection[3] and
                               np.linalg.norm(pose[0:2, connection[0]] -
                                              pose[0:2, connection[1]]) >
                               dash_length * 2) * dash_length
        draw_transparent(image,
                         draw_line,
                         (pose[0:2, connection[0]],
                          pose[0:2, connection[1]],
                          connection[2],
                          line_thickness,
                          this_dash_length),
                         opacity)
    # Draw circles.
    for joint_idx in range(pose.shape[1]):
        if connections is not None and skip_unconnected_joints:
            # Skip joints that are unconnected.
            lmfound = False
            for connection in connections:
                if joint_idx in connection[:2]:
                    lmfound = True
                    this_line_thickness = int(np.ceil(np.sqrt(connection[4]) *
                                                      line_thickness * 0.6))
                    break
            if not lmfound:
                continue
        else:
            this_line_thickness = int(np.ceil(line_thickness * 0.6))
        if multi_col_mode:
            ccol = circle_color[joint_idx]
        elif lm_region_mapping is not None:
            if joint_idx in lm_region_mapping.keys():
                ccol = lm_region_mapping[joint_idx]
            else:
                ccol = circle_color
        else:
            ccol = circle_color
        draw_transparent(image,
                         draw_circle,
                         (tuple(pose[0:2, joint_idx]),
                          this_line_thickness,
                          ccol),
                         opacity)
    return image
