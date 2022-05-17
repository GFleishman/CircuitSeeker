import numpy as np
from itertools import product
import SimpleITK as sitk
import CircuitSeeker.utility as ut
import os, psutil, shutil
from scipy.ndimage import map_coordinates
from ClusterWrap.decorator import cluster
import dask.array as da
import zarr


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


@cluster
def distributed_apply_transform(
    fix_zarr, mov_zarr,
    fix_spacing, mov_spacing,
    transform_list,
    blocksize,
    write_path,
    dataset_path=None,
    temporary_directory=None,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # TODO: generalize
    affine, deform = transform_list
    blocksize = np.array(blocksize)

    # ensure temporary directory exists
    if temporary_directory is None:
        temporary_directory = os.getcwd()
    temporary_directory += '/distributed_apply_transform_temp'
    os.makedirs(temporary_directory)
    zarr_path = temporary_directory + '/deform.zarr'
    zarr_blocks = (128,)*len(blocksize) + (3,)
    deform_zarr = ut.numpy_to_zarr(deform, zarr_blocks, zarr_path)

    # get overlap and number of blocks
    overlap = np.round(blocksize * 0.5).astype(int)
    nblocks = np.ceil(np.array(fix_zarr.shape) / blocksize).astype(int)

    # store block coordinates in a dask array
    block_coords = np.empty(nblocks, dtype=tuple)
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k) - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(fix_zarr.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))
        block_coords[i, j, k] = coords
    block_coords = da.from_array(block_coords, chunks=(1,)*block_coords.ndim)

    # pipeline to run on each block
    def transform_single_block(coords):

        # fetch fixed image slices and read fix
        fix_slices = coords.item()
        fix = fix_zarr[fix_slices]

        # get deform slices and read deform
        deform_slices = fix_slices
        if fix_zarr.shape != deform_zarr.shape[:-1]:
            ratio = np.array(deform_zarr.shape[:-1]) / fix_zarr.shape
            start = np.round([s.start * r for s, r in zip(fix_slices, ratio)]).astype(int)
            stop = np.round([s.stop * r for s, r in zip(fix_slices, ratio)]).astype(int)
            deform_slices = tuple(slice(a, b) for a, b in zip(start, stop))
        deform = deform_zarr[deform_slices]

        # get fixed block corners in physical units
        fix_block_coords = []
        for corner in list(product([0, 1], repeat=3)):
            a = [x.stop if y else x.start for x, y in zip(fix_slices, corner)]
            fix_block_coords.append(a)
        fix_block_coords = np.array(fix_block_coords)
        fix_block_coords_phys = fix_block_coords * fix_spacing

        # transform fix block coordinates
        transform_spacing = ut.relative_spacing(deform, fix, fix_spacing)
        transform_origin = transform_spacing * [s.start for s in deform_slices]
        mov_block_coords_phys = apply_transform_to_coordinates(
            fix_block_coords_phys, deform, transform_spacing, transform_spacing,
        )
        mov_block_coords_phys = apply_transform_to_coordinates(
            mov_block_coords_phys, affine,
        )

        # get moving block slices
        mov_block_coords = np.round(mov_block_coords_phys / mov_spacing).astype(int)
        mov_block_coords = np.maximum(0, mov_block_coords)
        mov_block_coords = np.minimum(np.array(mov_zarr.shape)-1, mov_block_coords)
        mov_start = np.min(mov_block_coords, axis=0)
        mov_stop = np.max(mov_block_coords, axis=0)
        mov_slices = tuple(slice(a, b) for a, b in zip(mov_start, mov_stop))
        mov = mov_zarr[mov_slices]

        # determine origins
        fix_origin = fix_spacing * [s.start for s in fix_slices]
        mov_origin = mov_spacing * [s.start for s in mov_slices]

        # resample
        aligned = apply_transform(
            fix, mov, fix_spacing, mov_spacing,
            transform_list=[affine, deform],
            transform_spacing=transform_spacing,
            transform_origin=fix_origin,
            fix_origin=fix_origin,
            mov_origin=mov_origin,
        )

        # crop out overlap
        for axis in range(aligned.ndim):

            # left side
            slc = [slice(None),]*aligned.ndim
            if fix_slices[axis].start != 0:
                slc[axis] = slice(overlap[axis], None)
                aligned = aligned[tuple(slc)]

            # right side
            slc = [slice(None),]*aligned.ndim
            if aligned.shape[axis] > blocksize[axis]:
                slc[axis] = slice(None, blocksize[axis])
                aligned = aligned[tuple(slc)]

        # return result
        return aligned

    # align all blocks
    aligned = da.map_blocks(
        transform_single_block,
        block_coords,
        dtype=fix_zarr.dtype,
        chunks=blocksize,
    )

    # crop to original size
    aligned = aligned[tuple(slice(0, s) for s in fix_zarr.shape)]

    # compute result, write to zarr array
    da.to_zarr(aligned, write_path, component=dataset_path)

    # remove temporary directory
    shutil.rmtree(temporary_directory)

    # return reference to result
    return zarr.open(write_path, 'r+')


# TODO: this function should take a list of transforms
def apply_transform_to_coordinates(
    coordinates,
    transform,
    transform_spacing=None,
    transform_origin=None,
):
    """
    """

    # if transform is an affine matrix
    if transform.shape == (4, 4):

        # matrix vector multiply
        mm, tt = transform[:3, :3], transform[:3, -1]
        return np.einsum('...ij,...j->...i', mm, coordinates) + tt

    # if transform is a deformation vector field
    else:

        # transform_spacing must be given
        error_message = "If transform is a displacement vector field, "
        error_message += "transform_spacing must be given."
        assert (transform_spacing is not None), error_message

        # get coordinates in transform voxel units, reformat for map_coordinates
        if transform_origin is not None: coordinates -= transform_origin
        coordinates = ( coordinates / transform_spacing ).transpose()

        # interpolate position field at coordinates, reformat, return
        interp = lambda x: map_coordinates(x, coordinates, order=1, mode='nearest')
        dX = np.array([interp(transform[..., i]) for i in range(3)]).transpose()
        return coordinates.transpose() * transform_spacing + dX


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


