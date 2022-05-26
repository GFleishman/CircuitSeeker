import numpy as np
import SimpleITK as sitk
import CircuitSeeker.utility as ut
import os, psutil
from scipy.ndimage import map_coordinates


def apply_transform(
    fix, mov,
    fix_spacing, mov_spacing,
    transform_list,
    transform_spacing=None,
    transform_origin=None,
    fix_origin=None,
    mov_origin=None,
    interpolate_with_nn=False,
    extrapolate_with_nn=False,
):
    """
    """

    # set global number of threads
    if "LSB_DJOB_NUMPROC" in os.environ:
        ncores = int(os.environ["LSB_DJOB_NUMPROC"])
    else:
        ncores = psutil.cpu_count(logical=False)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2*ncores)

    # convert images to sitk objects
    dtype = fix.dtype
    fix = sitk.Cast(ut.numpy_to_sitk(fix, fix_spacing, fix_origin), sitk.sitkFloat32)
    mov = sitk.Cast(ut.numpy_to_sitk(mov, mov_spacing, mov_origin), sitk.sitkFloat32)

    # construct transform
    fix_spacing = np.array(fix_spacing)
    if transform_spacing is None: transform_spacing = fix_spacing
    transform = ut.transform_list_to_composite_transform(
        transform_list, transform_spacing, transform_origin,
    )

    # set up resampler object
    resampler = sitk.ResampleImageFilter()
    resampler.SetNumberOfThreads(2*ncores)
    resampler.SetReferenceImage(fix)
    resampler.SetTransform(transform)

    # check for NN interpolation
    if interpolate_with_nn:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # check for NN extrapolation
    if extrapolate_with_nn:
        resampler.SetUseNearestNeighborExtrapolator(True)

    # execute, return as numpy array
    resampled = resampler.Execute(mov)
    return sitk.GetArrayFromImage(resampled).astype(dtype)


def apply_transform_to_coordinates(
    coordinates,
    transform_list,
    transform_spacing=None,
    transform_origin=None,
):
    """
    """

    for iii, transform in enumerate(transform_list):

        # if transform is an affine matrix
        if transform.shape == (4, 4):

            # matrix vector multiply
            mm, tt = transform[:3, :3], transform[:3, -1]
            coordinates = np.einsum('...ij,...j->...i', mm, coordinates) + tt

        # if transform is a deformation vector field
        else:

            # transform_spacing must be given
            error_message = "If transform is a displacement vector field, "
            error_message += "transform_spacing must be given."
            assert (transform_spacing is not None), error_message

            # handle multiple spacings
            spacing = transform_spacing
            if isinstance(spacing, tuple): spacing = spacing[iii]

            # get coordinates in transform voxel units, reformat for map_coordinates
            if transform_origin is not None: coordinates -= transform_origin
            coordinates = ( coordinates / spacing ).transpose()
    
            # interpolate position field at coordinates, reformat, return
            interp = lambda x: map_coordinates(x, coordinates, order=1, mode='nearest')
            dX = np.array([interp(transform[..., i]) for i in range(3)]).transpose()
            coordinates = coordinates.transpose() * spacing + dX
            if transform_origin is not None: coordinates += transform_origin

    return coordinates


def compose_displacement_vector_fields(
    first_field,
    second_field,
    spacing,
):
    """
    """

    # container for warped first field
    first_field_warped = np.empty_like(first_field)

    # loop over components
    for iii in range(3):

        # warp first field with second
        first_field_warped[..., iii] = apply_transform(
            first_field[..., iii], first_field[..., iii],
            spacing, spacing,
            transform_list=[second_field,],
            extrapolate_with_nn=True,
        )

    # combine warped first field and second field
    return first_field_warped + second_field


def compose_transforms(transform_one, transform_two, spacing):
    """
    """

    # two affines
    if transform_one.shape == (4, 4) and transform_two.shape == (4, 4):
        return np.matmul(transform_one, transform_two)

    # one affine, two field
    elif transform_one.shape == (4, 4):
        transform_one = ut.matrix_to_displacement_field(
            transform_one, transform_two.shape[:-1], spacing,
        )

    # one field, two affine
    elif transform_two.shape == (4, 4):
        transform_two = ut.matrix_to_displacement_field(
            transform_two, transform_one.shape[:-1], spacing,
        )

    # compose fields
    return compose_displacement_vector_fields(
        transform_one, transform_two, spacing,
    )


def invert_displacement_vector_field(
    field,
    spacing,
    iterations=10,
    order=2,
    sqrt_iterations=5,
):
    """
    """

    # initialize inverse as negative root
    root = _displacement_field_composition_nth_square_root(
        field, spacing, order, sqrt_iterations,
    )
    inv = - np.copy(root)

    # iterate to invert
    for i in range(iterations):
        inv -= compose_displacement_vector_fields(root, inv, spacing)

    # square-compose inv order times
    for i in range(order):
        inv = compose_displacement_vector_fields(inv, inv, spacing)

    # return result
    return inv


def _displacement_field_composition_nth_square_root(
    field,
    spacing,
    order,
    sqrt_iterations=5,
):
    """
    """

    # initialize with given field
    root = np.copy(field)

    # iterate taking square roots
    for i in range(order):
        root = _displacement_field_composition_square_root(
            root, spacing, iterations=sqrt_iterations,
        )

    # return result
    return root


def _displacement_field_composition_square_root(
    field,
    spacing,
    iterations=5,
):
    """
    """

    # container to hold root
    root = 0.5 * field

    # iterate
    for i in range(iterations):
        residual = (field - compose_displacement_vector_fields(root, root, spacing))
        root += 0.5 * residual

    # return result
    return root


