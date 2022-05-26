import numpy as np
from itertools import product
import CircuitSeeker.utility as ut
import os, shutil
from ClusterWrap.decorator import cluster
import dask.array as da
import zarr
import CircuitSeeker.transform as transform


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
        mov_block_coords_phys = transform.apply_transform_to_coordinates(
            fix_block_coords_phys, [deform, affine,], transform_spacing, transform_origin,
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
        aligned = transform.apply_transform(
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


@cluster
def distributed_apply_transform_to_coordinates(
    coordinates,
    transform_list,
    partition_size=30.,
    transform_spacing=None,
    transform_origin=None,
    temporary_directory=None,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # ensure temporary directory exists
    if temporary_directory is None:
        temporary_directory = os.getcwd()
    temporary_directory += '/distributed_apply_transform_temp'
    os.makedirs(temporary_directory)

    # ensure all deforms are zarr
    new_list = []
    zarr_blocks = (128,)*3 + (3,)  # TODO: generalize
    for iii, transform in enumerate(transform_list):
        if transform.shape != (4, 4):
            zarr_path = temporary_directory + f'/deform{iii}.zarr'
            transform = ut.numpy_to_zarr(transform, zarr_blocks, zarr_path)
        new_list.append(transform)
    transform_list = new_list

    # determine partitions of coordinates
    origin = np.min(coordinates, axis=0)
    nblocks = np.max(coordinates, axis=0) - origin
    nblocks = np.ceil(nblocks / partition_size).astype(int)
    partitions = []
    for (i, j, k) in np.ndindex(*nblocks):
        lower_bound = origin + partition_size * np.array((i, j, k))
        upper_bound = lower_bound + partition_size
        not_too_low = np.all(coordinates >= lower_bound, axis=1)
        not_too_high = np.all(coordinates < upper_bound, axis=1)
        coords = coordinates[ not_too_low * not_too_high ]
        if coords.size != 0: partitions.append(coords)

    def transform_partition(coordinates, transform_list):

        # read relevant region of transform
        a = np.min(coordinates, axis=0)
        b = np.max(coordinates, axis=0)
        new_list = []
        for iii, transform in enumerate(transform_list):
            if transform.shape != (4, 4):
                spacing = transform_spacing
                if isinstance(spacing, tuple): spacing = spacing[iii]
                start = np.floor(a / spacing).astype(int)
                stop = np.ceil(b / spacing).astype(int) + 1
                crop = tuple(slice(x, y) for x, y in zip(start, stop))
                transform = transform[crop]
            new_list.append(transform)
        transform_list = new_list

        # apply transforms
        return transform.apply_transform_to_coordinates(
            coordinates, transform_list,
            transform_spacing,
            transform_origin=a,
        )

    # transform all partitions and return
    futures = cluster.client.map(
        transform_partition, partitions,
        transform_list=transform_list,
    )
    results = cluster.client.gather(futures)

    # remove temp directory
    shutil.rmtree(temporary_directory)

    return np.concatenate(results, axis=0)


@cluster
def distributed_invert_displacement_vector_field(
    field_zarr,
    spacing,
    blocksize,
    write_path,
    iterations=10,
    order=2,
    sqrt_iterations=5,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # get overlap and number of blocks
    blocksize = np.array(blocksize)
    overlap = np.round(blocksize * 0.25).astype(int)
    nblocks = np.ceil(np.array(field_zarr.shape[:-1]) / blocksize).astype(int)

    # store block coordinates in a dask array
    block_coords = np.empty(nblocks, dtype=tuple)
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k) - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(field_zarr.shape[:-1], stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))
        block_coords[i, j, k] = coords
    block_coords = da.from_array(block_coords, chunks=(1,)*block_coords.ndim)


    def invert_block(slices):

        slices = slices.item()
        field = field_zarr[slices]
        inverse = transform.invert_displacement_vector_field(
            field, spacing, iterations, order, sqrt_iterations,
        )
        
        # crop out overlap
        for axis in range(inverse.ndim - 1):

            # left side
            slc = [slice(None),]*(inverse.ndim - 1)
            if slices[axis].start != 0:
                slc[axis] = slice(overlap[axis], None)
                inverse = inverse[tuple(slc)]

            # right side
            slc = [slice(None),]*(inverse.ndim - 1)
            if inverse.shape[axis] > blocksize[axis]:
                slc[axis] = slice(None, blocksize[axis])
                inverse = inverse[tuple(slc)]

        # return result
        return inverse

    # invert all blocks
    inverse = da.map_blocks(
        invert_block,
        block_coords,
        dtype=field_zarr.dtype,
        new_axis=[3,],
        chunks=tuple(blocksize) + (3,),
    )

    # crop to original size
    inverse = inverse[tuple(slice(0, s) for s in field_zarr.shape[:-1])]

    # compute result, write to zarr array
    da.to_zarr(inverse, write_path)

    # return reference to result
    return zarr.open(write_path, 'r+')

