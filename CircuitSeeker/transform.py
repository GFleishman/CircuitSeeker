import numpy as np
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
    transform = ut.transform_list_to_composite_transform(
        transform_list, transform_spacing or fix_spacing, transform_origin,
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
    transform_spacing=None,
    dataset_path=None,
    temporary_directory=None,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # TODO: generalize
    affine, deform = transform_list

    # blocksize should be an array
    blocksize = np.array(blocksize)

    # ensure temporary directory exists
    if temporary_directory is None:
        temporary_directory = os.getcwd()
    temporary_directory += '/distributed_apply_transform_temp'
    os.makedirs(temporary_directory)

    # write deform as zarr file
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

        # fetch coords slice and deform
        coords = coords.item()

        # read relevant part of transform
        # TODO: double check transform origin for example, make sure no error being introduced!
        deform_coords = coords
        if fix_zarr.shape != deform_zarr.shape[:-1]:
            ratio = np.array(deform_zarr.shape[:-1]) / fix_zarr.shape
            starts = np.round([s.start * r for s, r in zip(coords, ratio)]).astype(int)
            stops = np.round([s.stop * r for s, r in zip(coords, ratio)]).astype(int)
            deform_coords = tuple(slice(a, b) for a, b in zip(starts, stops))
        deform = deform_zarr[deform_coords]

        # determine bounding box around moving image region
        # get initial voxel coords
        start_ijk = tuple(s.start for s in coords)
        stop_ijk = tuple(s.stop for s in coords)

        # convert to physical units, add the displacements
        start_xyz = np.array(start_ijk) * fix_spacing + deform[(0,)*len(coords)]
        stop_xyz = np.array(stop_ijk) * fix_spacing + deform[(-1,)*len(coords)]

        # apply the affine
        start_mov_xyz = np.matmul(affine, np.concatenate((start_xyz, (1,))))
        stop_mov_xyz = np.matmul(affine, np.concatenate((stop_xyz, (1,))))

        # convert to voxel units, take outer box, ensure coords within range
        start_mov = np.floor(start_mov_xyz[:-1] / mov_spacing).astype(int)
        start_mov = np.maximum(0, start_mov)
        stop_mov = np.ceil(stop_mov_xyz[:-1] / mov_spacing).astype(int)
        stop_mov = np.minimum(mov_zarr.shape, stop_mov)

        # convert back to tuple of slice
        mov_coords = tuple(slice(a, b) for a, b in zip(start_mov, stop_mov))

        # read the data
        fix = fix_zarr[coords]
        mov = mov_zarr[mov_coords]

        # determine origin
        fix_origin = fix_spacing * [s.start for s in coords]
        mov_origin = mov_spacing * [s.start for s in mov_coords]

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
            if coords[axis].start != 0:
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


def apply_transform_to_coordinates(
    coordinates,
    transform,
    transform_spacing=None,
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


