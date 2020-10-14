import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


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
    for i in range(mask.shape[cor_axis]):
        # compute coordinates
        x = i * spacing[cor_axis]
        y = smp[0]*x**2 + smp[1]*x + smp[2]
        j = int(round( y / spacing[ax_axis] ))

        # if inside mask, accumulate arc length
        sli = [slice(sag_plane, sag_plane+1),]*3
        sli[cor_axis] = slice(i, i+1)
        sli[ax_axis] = slice(j, j+1)
        if mask[sli] == 1:
            arc += (1 + (2*smp[0]*x + smp[1])**2)**0.5
            result.append([x, y, arc])

    # correspondence defined by percentage of arc length
    result = np.array(result)
    result[:, -1] /= arc
    return result


def force_to_displacement(
    force,
    stop_function,
    step=0.1,
    field_sigma=1.0,
):
    """
    """

    # integrate force with smoothing
    deform = np.zeros_like(force)
    while stop_function(deform):
        deform = deform + step*force
        for i in range(3):
            deform[..., i] = gaussian_filter1d(deform[..., i], field_sigma, axis=-1)
    return deform


def align_sagittal_medial_polynomials(
    fixed, moving,
    fixed_smp, moving_smp,
    fixed_spacing, moving_spacing,
    cor_axis, ax_axis, sag_plane,
    force_sigma=64.0,
    threshold=None,
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

    # define force
    force = np.zeros(fixed.shape + (3,), dtype=np.float32)
    force[:, x_as_i, y_as_j] = delta
    for i in range(3):
        force[..., i] = gaussian_filter1d(force[..., i], force_sigma, axis=-1)
    force = force * abs(np.mean( force[sag_plane, x_as_i, y_as_j] - delta ))

    # define integration convergence criteria
    if threshold is None:
        threshold = np.min(fixed_spacing)
    stop_function = lambda x: np.mean( (x[sag_plane, x_as_i, y_as_j] - delta)**2 ) > threshold

    # integrate force to smooth displacement and return
    return force_to_displacement(force, stop_function, **kwargs)



