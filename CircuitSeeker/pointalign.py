import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.measurements import label, center_of_mass
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, gaussian_filter1d


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

    # TODO: get rid of threshold, instead take a "topk" parameter, retain top k matches
    #    return correlations with the actual point matches

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


def force_to_displacement(
    force,
    score_function,
    step=0.01,
    field_sigma=1.0,
):
    """
    """

    # integrate force with smoothing
    deform = np.zeros_like(force)
    score = score_function(deform)
    previous_score = score + 1
    while score - previous_score < 0:
        deform = deform + step*force
        for i in range(3):
            deform[..., i] = gaussian_filter1d(deform[..., i], field_sigma, axis=-1)
        previous_score = score
        score = score_function(deform)
    return deform


def align_point_correspondence(
    fixed, moving,
    fixed_spacing, moving_spacing,
    fix_points, mov_points,
    pairs,
    force_sigma=32.0,
):
    """
    """

    # TODO
    None


