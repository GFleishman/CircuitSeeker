import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage.measurements import label, labeled_comprehension
import morphsnakes

from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree


def estimate_background(image, rad=5):
    """
    """

    a, b = slice(0, rad), slice(-rad, None)
    corners = [[a,a,a], [a,a,b], [a,b,a], [a,b,b],
               [b,a,a], [b,a,b], [b,b,a], [b,b,b]]
    return np.median([np.mean(image[c]) for c in corners])


def segment(
    image,
    lambda2,
    iterations,
    smoothing=1,
    threshold=None,
    init=None
    ):
    """
    """

    if threshold is not None:
        image[image < threshold] = 0
    if init is None:
        init = np.zeros_like(image, dtype=np.uint8)
        bounds = np.ceil(np.array(image.shape) * 0.1).astype(int)
        init[[slice(b, -b) for b in bounds]] = 1
    return morphsnakes.morphological_chan_vese(
        image,
        iterations,
        init_level_set=init,
        smoothing=smoothing,
        lambda2=lambda2,
    ).astype(np.uint8)


def largest_connected_component(mask):
    lbls, nlbls = label(mask)
    vols = labeled_comprehension(mask, lbls, range(1, nlbls+1), np.sum, float, 0)
    mask[lbls != np.argmax(vols)+1] = 0
    return mask


def brain_detection(
    image,
    voxel_spacing,
    iterations=[40,8,2],
    shrink_factors=[4,2,1],
    smooth_sigmas=[8,4,2],
    lambda2=20,
    background=None,
    mask=None,
    mask_smoothing=1,
    ):
    """
    """

    # segment
    if background is None:
        background = estimate_background(image)
    for its, sf, ss in zip(iterations, shrink_factors, smooth_sigmas):
        image_small = zoom(gaussian_filter(image, ss/voxel_spacing), 1./sf, order=1)
        if mask is not None:
            zoom_factors = [x/y for x, y in zip(image_small.shape, mask.shape)]
            mask = zoom(mask, zoom_factors, order=0)
        mask = segment(
            image_small,
            lambda2=lambda2,
            iterations=its,
            smoothing=mask_smoothing,
            threshold=background,
            init=mask,
        )

    # basic topological correction
    mask = binary_erosion(mask, iterations=2)
    mask = largest_connected_component(mask)
    mask = binary_dilation(mask, iterations=2)
    mask = binary_fill_holes(mask).astype(np.uint8)

    # ensure output is on correct grid
    if mask.shape != image.shape:
        zoom_factors = [x/y for x, y in zip(image.shape, mask.shape)]
        mask = zoom(mask, zoom_factors, order=0)
    return mask


def difference_of_gaussians_3d(image, big_sigma, small_sigma):
    """
    """

    g1 = gaussian_filter(image, small_sigma)
    g2 = gaussian_filter(image, big_sigma)
    return g1 - g2


def cells_point_cloud(
    image, spacing,
    big_sigma,
    small_sigma,
    threshold,
    erode_filter,
    mask=None,
):
    """
    """

    dog = difference_of_gaussians_3d(image, big_sigma/spacing, small_sigma/spacing)
    dog = (dog >= threshold).astype(np.uint16)
    points = binary_erosion(dog, structure=erode_filter).astype(np.uint16)
    if mask is not None:
        points = points * mask
    point_labels, npoints = label(points)
    points = center_of_mass(points, labels=point_labels, index=range(1, npoints))
    return np.array(points) * spacing, dog


def get_context(image, position, radius):
    """
    """

    position = np.round(position).astype(int)
    radius = np.array(radius)
    low = position - radius
    high = position + radius + 1
    x = image[low[0]:high[0], low[1]:high[1], low[2]:high[2]].flatten()
    if len(x) != np.prod(2*radius + 1):
        x = None
    return x


def correlations(fix_window, mov_window):
    """
    """

    # ensure enough precision for sensitive calculations
    fix_window = fix_window.astype(np.float64)  # W
    mov_window = mov_window.astype(np.float64)  # NxW

    # get statistics
    fix_mean = np.mean(fix_window)  # scalar
    fix_std  = np.std(fix_window)  # scalar
    mov_mean = np.mean(mov_window, axis=1)  # N
    mov_std  = np.std(mov_window, axis=1)  # N

    # center arrays
    fix_center = fix_window - fix_mean  # W
    mov_center = mov_window - mov_mean[..., None]  # NxW

    # compute correlations
    correlations = np.sum(fix_center[None, ...] * mov_center, axis=1) / len(fix_window)  # N
    return correlations / mov_std / fix_std  # N



def point_correspondence(
    fixed, moving,
    fixed_spacing, moving_spacing,
    fix_points, mov_points,
    search_radius,
    threshold,
    context_radius,
):
    """
    """

    # get trees for each set of points
    fix_tree = cKDTree(fix_points)
    mov_tree = cKDTree(mov_points)

    # search fix tree with mov tree points
    potential_pairs = fix_tree.query_ball_tree(mov_tree, search_radius)

    # loop over point matches
    pairs = []
    for i, matches in enumerate(potential_pairs):

        # get fixed point context
        fix_window = get_context(fixed, fix_points[i]/fixed_spacing, context_radius)
        if fix_window is None:
            continue
 
        # get moving point contexts
        mov_window = []
        for j, match in enumerate(matches):
            w = get_context(moving, mov_points[match]/moving_spacing, context_radius)
            if w is None:
                matches.pop(j)
                continue
            mov_window.append(w)
        mov_window = np.array(mov_window)

        # get best correlation match
        if mov_window.shape[0] > 0:
            corr = correlations(fix_window, mov_window)
            argmx = np.argmax(corr)
            best, best_corr = matches[argmx], corr[argmx]

            # if above threshold, add to correspondence list
            if best_corr >= threshold:
                pairs.append((i, best))

        # TEMP
        if i%100 == 0:
            print(i, ": ", len(pairs))

    return pairs

