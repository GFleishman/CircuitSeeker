import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import rotate


def raw_moment(image, spacing, iord, jord, kord):
    """
    """

    nrows, ncols, nstacks = image.shape
    x, y, z = np.mgrid[:nrows, :ncols, :nstacks]
    x = x * spacing[0]
    y = y * spacing[1]
    z = z * spacing[2]
    return (image * x**iord * y**jord * z**kord).sum()


def principal_axes(image, spacing):
    """
    """

    # mean
    image_sum = image.sum()
    x_bar = raw_moment(image, spacing, 1, 0, 0)
    y_bar = raw_moment(image, spacing, 0, 1, 0)
    z_bar = raw_moment(image, spacing, 0, 0, 1)
    mean = np.array([x_bar, y_bar, z_bar]) / image_sum

    # covariance
    c00 = raw_moment(image, spacing, 2, 0, 0) - x_bar * mean[0]
    c11 = raw_moment(image, spacing, 0, 2, 0) - y_bar * mean[1]
    c22 = raw_moment(image, spacing, 0, 0, 2) - z_bar * mean[2]
    c01 = raw_moment(image, spacing, 1, 1, 0) - x_bar * mean[1]
    c02 = raw_moment(image, spacing, 1, 0, 1) - x_bar * mean[2]
    c12 = raw_moment(image, spacing, 0, 1, 1) - y_bar * mean[2]
    covariance = np.array([[c00, c01, c02], [c01, c11, c12],[c02, c12, c22]]) / image_sum

    # principal axes
    eigvals, eigvecs = np.linalg.eigh(covariance)

    # ensure positive axis orientation and consistent axis ordering
    for col in range(eigvecs.shape[-1]):
        idx = np.argmax(np.abs(eigvecs[:, col]))
        if eigvecs[idx, col] < 0:
            eigvecs[:, col] *= -1
    sort_idx = np.argmax(eigvecs, axis=1)
    return mean, eigvals[sort_idx], eigvecs[:, sort_idx]


def align_modes(mean1, cov1, mean2, cov2):
    """
    """

    # align centers of mass
    translation = np.eye(4)
    translation[:3, -1] = mean2 - mean1

    # rotate principal axes about center of mass
    com_right = np.eye(4)
    com_right[:3, -1] = -mean2
    com_left = np.abs(com_right)
    rot = np.eye(4)
    rot[:3, :3] = np.matmul(cov2, cov1.T)

    # compose transforms
    rotation = np.matmul(com_left, np.matmul(rot, com_right))
    return np.matmul(rotation, translation)


def sagittal_medial_polynomial(image, spacing, cor_axis, ax_axis, order=2):
    """
    """

    # get center of mass for each slice along axis
    orders = np.eye(3)
    points = np.empty((image.shape[cor_axis], 3))
    for idx in range(image.shape[cor_axis]):
        sli = [slice(None, None, None),]*3
        sli[cor_axis] = slice(idx, idx+1)
        image_slice = image[sli]
        norm = image_slice.sum()
        for ax in range(3):
            if ax != cor_axis:
                points[idx, ax] = raw_moment(image_slice, spacing, *orders[ax]) / norm
            else:
                points[idx, ax] = idx * spacing[cor_axis]

    # remove slices that had no data
    points = points[~np.isnan(points).any(axis=1)]

    # fit a smooth curve through those centers
    return np.polyfit(points[:, cor_axis], points[:, ax_axis], order)


def compute_arc_lengths(mask, smp, spacing, cor_axis, ax_axis, sag_plane):
    """
    """

    arc = 0
    result = []
    previous_slice = None
    ant_mark = 0
    post_mark = mask.shape[cor_axis] - 1
    for i in range(mask.shape[cor_axis]):
        # compute coordinates
        x = i * spacing[cor_axis]
        y = smp[0]*x**2 + smp[1]*x + smp[2]
        j = int(round( y / spacing[ax_axis] ))

        # accumulate arc length
        arc += (1 + (2*smp[0]*x + smp[1])**2)**0.5
        result.append([x, y, arc])

        # if entering or exiting foreground, mark index
        sli = [slice(sag_plane, sag_plane+1),]*3
        sli[cor_axis] = slice(i, i+1)
        sli[ax_axis] = slice(j, j+1)
        if previous_slice is not None:
            if mask[previous_slice] == 0 and mask[sli] == 1:
                ant_mark = i
            if mask[previous_slice] == 1 and mask[sli] == 0:
                post_mark = i-1
        previous_slice = sli

    # correspondence defined by anterior/posterior landmarks set
    # to 0/1 respectively
    result = np.array(result)
    result[:, 2] = result[:, 2] - result[ant_mark, 2]
    result[:, 2] = result[:, 2] / result[post_mark, 2]
    return result


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


def align_sagittal_medial_polynomials(
    fixed, moving,
    fixed_smp, moving_smp,
    fixed_spacing, moving_spacing,
    cor_axis, ax_axis, sag_plane,
    force_sigma=64.0,
    **kwargs,
):
    """
    """

    # get coordinates and arc lengths
    fix_arc = compute_arc_lengths(
        fixed, fixed_smp, fixed_spacing,
        cor_axis, ax_axis, sag_plane,
    )
    mov_arc = compute_arc_lengths(
        moving, moving_smp, moving_spacing,
        cor_axis, ax_axis, sag_plane,
    )

    # establish correspondences
    delta_x = np.interp(fix_arc[:, -1], mov_arc[:, -1], mov_arc[:, 0]) - fix_arc[:, 0]
    delta_y = np.interp(fix_arc[:, -1], mov_arc[:, -1], mov_arc[:, 1]) - fix_arc[:, 1]
    delta_z = np.zeros_like(delta_x)
    delta = np.hstack((delta_z[..., None], delta_x[..., None], delta_y[..., None]))

    # get voxel coordinates of smp curves
    x_as_i = np.round(fix_arc[:, 0] / fixed_spacing[cor_axis]).astype(np.uint16)
    y_as_j = np.round(fix_arc[:, 1] / fixed_spacing[ax_axis]).astype(np.uint16)
    ant_mark = np.argwhere(fix_arc[:, -1] == 0.)[0, 0]
    post_mark = np.argwhere(fix_arc[:, -1] == 1.)[0, 0]
    x_as_i_fg = x_as_i[ant_mark:post_mark+1]  # within foreground
    y_as_j_fg = y_as_j[ant_mark:post_mark+1]

    # define force
    force = np.zeros(fixed.shape + (3,), dtype=np.float32)
    force[:, x_as_i, y_as_j] = delta
    for i in range(3):
        force[..., i] = gaussian_filter1d(force[..., i], force_sigma, axis=-1)
    scale = delta[x_as_i_fg, 1:] / force[sag_plane, x_as_i_fg, y_as_j_fg, 1:]
    force = force * np.mean(scale)

    # define integration convergence criteria
    def score_function(x):
        dist = x[sag_plane, x_as_i_fg, y_as_j_fg] - delta[x_as_i_fg]
        return np.max( np.linalg.norm(dist, axis=-1) )

    # integrate force to smooth displacement and return
    return force_to_displacement(force, score_function, **kwargs)


def symmetric_padding(image, center):
    """
    """

    pads = []
    for i in range(len(image.shape)):
        left = center[i]
        right = image.shape[i] - center[i] - 1
        if right - left > 0:
            pads.append((right-left, 0))
        elif left - right > 0:
            pads.append((0, left-right))
        else:
            pads.append((0, 0))
    return pads


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


def brute_force_rotation_2d(
    fixed, moving, moving_mask,
    min_angle, max_angle, step,
    threshold=0.4,
):
    """
    """

    # only attempt to align planes with some coincident data
    if not np.any( fixed*moving ):
        return 0, np.zeros(2)

    # get moving center of mass in pixel units
    denom = np.sum(moving_mask)
    x_bar = raw_moment(moving_mask[..., None], np.ones(3), 1, 0, 0)
    y_bar = raw_moment(moving_mask[..., None], np.ones(3), 0, 1, 0)
    center_pix = np.round( np.array([x_bar, y_bar]) / denom ).astype(int)

    # pad moving image for symmetry about center
    pads = symmetric_padding(moving_mask, center_pix)
    sli = []
    for p in pads:
        if p[0] != 0:
            sli.append(slice(p[0], None))
        elif p[1] != 0:
            sli.append(slice(None, -p[1]))
        else:
            sli.append(slice(None, None))
    moving_pad = np.pad(moving, pads)

    # make all rotations
    fixed_flat = fixed.flatten()
    angles = np.arange(min_angle, max_angle+step/2., step)
    rotated = np.empty((len(angles), len(fixed_flat)))
    for i, angle in enumerate(angles):
        rotated[i] = rotate(moving_pad, angle, reshape=False, order=1)[sli].flatten()

    # find best correlation
    corrs = correlations(fixed_flat, rotated)
    best_idx = np.argmax(corrs)
    best_corr, best_angle = corrs[best_idx], angles[best_idx]
    if best_corr < threshold:
        best_corr, best_angle = 0, 0

    # return rigid transform
    return best_angle, center_pix


def rigid_matrix(angle, center):
    """
    """

    # construct rotation matrix
    rot = np.eye(3)
    rot[0, 0] =  np.cos(angle * np.pi/180)
    rot[1, 1] =  np.cos(angle * np.pi/180)
    rot[0, 1] =  np.sin(angle * np.pi/180)
    rot[1, 0] = -np.sin(angle * np.pi/180)

    # construct translations to/from center of rotation
    trans_right = np.eye(3)
    trans_right[:2, -1] = -center
    trans_left = np.abs(trans_right)

    return np.matmul(trans_left, np.matmul(rot, trans_right))


def align_twist(
    fixed, moving, moving_mask,
    spacing, axis,
    min_angle, max_angle, angle_step,
    smooth_sigma=12.0,
):
    """
    """

    # get rotation for each slice
    angles, centers = [], []
    for i in range(fixed.shape[axis]):
        slc = [slice(None, None),]*3
        slc[axis] = slice(i, i+1)
        f = fixed[slc].squeeze()
        m = moving[slc].squeeze()
        mm = moving_mask[slc].squeeze()
        angle, center = brute_force_rotation_2d(
            f, m, mm, min_angle, max_angle, angle_step,
        )
        angles.append(angle)
        centers.append(center)

    # smooth angles and centers
    angles, centers = np.array(angles), np.array(centers)
    angles = gaussian_filter1d(angles, smooth_sigma)
    centers = gaussian_filter1d(centers, smooth_sigma, axis=0)

    # convert to vector field
    deform = np.empty(fixed.shape + (3,))
    nrows, ncols, nstacks = fixed.shape
    grid = np.array(np.mgrid[:nrows, :stacks]).transpose(1,2,0)
    for i in range(deform.shape[axis]):

        # get slice
        slc = [slice(None, None),]*4
        slc[axis] = slice(i, i+1)
        if axis == 0:
            slc[-1] = slice(1, 3)
        elif axis == 1:
            slc[-1] = slice(0, 3, 2)
        elif axis == 2:
            slc[-1] = slice(0, 2)

        # transform grid
        rigid = rigid_matrix(angles[i], centers[i])
        mm, tt = rigid[:2, :2], rigid[:2, -1]
        res = (np.einsum('...ij,...j->...i', mm, grid) + tt) - grid
        deform[slc] = np.expand_dims(res, axis)

    return deform * spacing


