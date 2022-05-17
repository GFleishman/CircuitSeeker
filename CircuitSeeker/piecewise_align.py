import os, shutil
import numpy as np
from itertools import product
from dask.distributed import as_completed, wait
from ClusterWrap.decorator import cluster
import CircuitSeeker.utility as ut
from CircuitSeeker.align import alignment_pipeline
from CircuitSeeker.transform import apply_transform
from CircuitSeeker.transform import apply_transform_to_coordinates
from CircuitSeeker.transform import compose_transforms


@cluster
def distributed_piecewise_alignment_pipeline(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    steps,
    nblocks,
    overlap=0.5,
    fix_mask=None,
    mov_mask=None,
    static_transform_list=[],
    random_kwargs={},
    rigid_kwargs={},
    affine_kwargs={},
    deform_kwargs={},
    cluster=None,
    cluster_kwargs={},
    temporary_directory=None,
    write_path=None,
    n_serial_partitions=1,
    **kwargs,
):
    """
    Piecewise alignment of moving to fixed image.
    Overlapping blocks are given to `alignment_pipeline` in parallel
    on distributed hardware. Can include random, rigid, affine, and
    deformable alignment. Inputs can be numpy or zarr arrays. Output
    is a single displacement vector field for the entire domain.
    Output can be returned to main process memory as a numpy array
    or written to disk as a zarr array.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.shape` must equal `mov.shape`
        I.e. typically piecewise affine alignment is done after
        a global affine alignment wherein the moving image has
        been resampled onto the fixed image voxel grid.

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    steps : list of strings
        steps argument of alignment_pipeline function

    nblocks : iterable
        The number of blocks to use along each axis.
        Length should be equal to `fix.ndim`

    overlap : float in range [0, 1] (default: 0.5)
        Block overlap size as a percentage of block size

    fix_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the fixed image
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    mov_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the moving image
        Assumed to have the same domain as the moving image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    random_kwargs : dict (default: {})
        Arguments passed to `random_affine_search`
        Note - some arguments are required. See documentation for `random_affine_search`

    rigid_kwargs : dict (default: {})
        Arguments passed to `affine_align` during rigid step
        Note - some arguments are required. See documentation for `affine_align`

    affine_kwargs : dict (default: {})
        Arguments passed to `affine_align` during affine step
        Note - some arguments are required. See documentation for `affine_align`

    deform_kwargs : dict (default: {})
        Arguments passed to `deformable_align`
        Note - some arguments are required. See documentation for `deformable_align`

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, then close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    temporary_directory : string (default: None)
        Temporary files are created during alignment. The temporary files will be
        in their own folder within the `temporary_directory`. The default is the
        current directory. Temporary files are removed if the function completes
        successfully.

    write_path : string (default: None)
        If the transform found by this function is too large to fit into main
        process memory, set this parameter to a location where the transform
        can be written to disk as a zarr file.

    n_serial_partitions : int (default: 1)
        The number of sets of parallel computations to run. Very large alignments
        can overwhelm the resources of even a large cluster. This can be solved
        by ordering the computations in a way that fits into the requested
        resources. 

    kwargs : any additional arguments
        Arguments that will apply to all alignment steps. These are overruled by
        arguments for specific steps e.g. `random_kwargs` etc.

    Returns
    -------
    field : nd array or zarr.core.Array
        Local affines stitched together into a displacement field
        Shape is `fix.shape` + (3,) as the last dimension contains
        the displacement vector.
    """

    # temporary file paths and create zarr images
    if temporary_directory is None:
        temporary_directory = os.getcwd()
    temporary_directory += '/distributed_alignment_temp'
    os.makedirs(temporary_directory)
    fix_zarr_path = temporary_directory + '/fix.zarr'
    mov_zarr_path = temporary_directory + '/mov.zarr'
    fix_mask_zarr_path = temporary_directory + '/fix_mask.zarr'
    mov_mask_zarr_path = temporary_directory + '/mov_mask.zarr'
    zarr_blocks = (128,)*fix.ndim
    fix_zarr = ut.numpy_to_zarr(fix, zarr_blocks, fix_zarr_path)
    mov_zarr = ut.numpy_to_zarr(mov, zarr_blocks, mov_zarr_path)
    if fix_mask is not None: fix_mask_zarr = ut.numpy_to_zarr(fix_mask, zarr_blocks, fix_mask_zarr_path)
    if mov_mask is not None: mov_mask_zarr = ut.numpy_to_zarr(mov_mask, zarr_blocks, mov_mask_zarr_path)

    # zarr files for initial deformations, get initial deform spacings
    new_list = []
    for iii, transform in enumerate(static_transform_list):
        if transform.shape != (4, 4) and len(transform.shape) != 1:
            path = temporary_directory + f'/deform{iii}.zarr'
            transform = ut.numpy_to_zarr(transform, zarr_blocks + (transform.shape[-1],), path)
        new_list.append(transform)
    static_transform_list = new_list

    # determine fixed image slices for blocking
    blocksize = np.ceil( np.array(fix.shape) / nblocks ).astype(int)
    overlaps = np.round(blocksize * overlap).astype(int)
    indices = []
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k) - overlaps
        stop = start + blocksize + 2 * overlaps
        start = np.maximum(0, start)
        stop = np.minimum(fix.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))
        indices.append((i, j, k, coords))

    # establish all keyword arguments
    random_kwargs = {**kwargs, **random_kwargs}
    rigid_kwargs = {**kwargs, **rigid_kwargs}
    affine_kwargs = {**kwargs, **affine_kwargs}
    deform_kwargs = {**kwargs, **deform_kwargs}

    # closure for alignment pipeline
    def align_single_block(
        indices,
        static_transform_list,
    ):

        # get the coordinates, read fixed data
        block_index, fix_slices = indices[:3], indices[3]
        fix = fix_zarr[fix_slices]

        # get fixed image block corners in physical units
        fix_block_coords = []
        for corner in list(product([0, 1], repeat=3)):
            a = [x.stop if y else x.start for x, y in zip(fix_slices, corner)]
            fix_block_coords.append(a)
        fix_block_coords = np.array(fix_block_coords)
        fix_block_coords_phys = fix_block_coords * fix_spacing

        # parse initial transforms
        # recenter affines, read deforms, apply transforms to crop coordinates
        new_list = []
        mov_block_coords_phys = np.copy(fix_block_coords_phys)
        for transform in static_transform_list[::-1]:
            if transform.shape == (4, 4):
                mov_block_coords_phys = apply_transform_to_coordinates(
                    mov_block_coords_phys, transform,
                )
                transform = ut.change_affine_matrix_origin(transform, -fix_block_coords_phys[0])
            else:
                ratio = np.array(transform.shape[:-1]) / fix_zarr.shape
                start = np.round( ratio * fix_block_coords[0] ).astype(int)
                stop = np.round( ratio * fix_block_coords[-1] ).astype(int)
                transform_slices = tuple(slice(a, b) for a, b in zip(start, stop))
                transform = transform[transform_slices]
                spacing = ut.relative_spacing(transform, fix, fix_spacing)
                origin = spacing * [s.start for s in transform_slices]
                mov_block_coords_phys = apply_transform_to_coordinates(
                    mov_block_coords_phys, transform, spacing, origin
                )
            new_list.append(transform)
        static_transform_list = new_list[::-1]

        # get moving image crop, read moving data 
        mov_block_coords = np.round(mov_block_coords_phys / mov_spacing).astype(int)
        mov_block_coords = np.maximum(0, mov_block_coords)
        mov_block_coords = np.minimum(np.array(mov_zarr.shape)-1, mov_block_coords)
        mov_start = np.min(mov_block_coords, axis=0)
        mov_stop = np.max(mov_block_coords, axis=0)
        mov_slices = tuple(slice(a, b) for a, b in zip(mov_start, mov_stop))
        mov = mov_zarr[mov_slices]

        # read masks
        fix_mask, mov_mask = None, None
        if os.path.isdir(fix_mask_zarr_path):
            ratio = np.array(fix_mask_zarr.shape) / fix_zarr.shape
            start = np.round( ratio * fix_block_coords[0] ).astype(int)
            stop = np.round( ratio * fix_block_coords[-1] ).astype(int)
            fix_mask_slices = tuple(slice(a, b) for a, b in zip(start, stop))
            fix_mask = fix_mask_zarr[fix_mask_slices]
        if os.path.isdir(mov_mask_zarr_path):
            ratio = np.array(mov_mask_zarr.shape) / mov_zarr.shape
            start = np.round( ratio * mov_start ).astype(int)
            stop = np.round( ratio * mov_stop ).astype(int)
            mov_mask_slices = tuple(slice(a, b) for a, b in zip(start, stop))
            mov_mask = mov_mask_zarr[mov_mask_slices]

        # get moving image origin
        mov_origin = mov_start * mov_spacing - fix_block_coords_phys[0]

        # run alignment pipeline
        transform = alignment_pipeline(
            fix, mov, fix_spacing, mov_spacing, steps,
            fix_mask=fix_mask, mov_mask=mov_mask,
            mov_origin=mov_origin,
            static_transform_list=static_transform_list,
            random_kwargs=random_kwargs,
            rigid_kwargs=rigid_kwargs,
            affine_kwargs=affine_kwargs,
            deform_kwargs=deform_kwargs,
        )

        # convert to single vector field
        if isinstance(transform, tuple):
            affine, deform = transform[0], transform[1]
            transform = compose_transforms(affine, deform, fix_spacing)
        else:
            transform = ut.matrix_to_displacement_field(transform, fix.shape, fix_spacing)

        # create weights array
        core, pad_ones, pad_linear = [], [], []
        for i in range(3):
            # get core shape and pad sizes
            o = max(0, 2*overlaps[i]-1)
            p_ones, p_linear = [0, 0], [o, o]
            if block_index[i] == 0:
                p_ones[0], p_linear[0] = o//2, 0
            if block_index[i] == nblocks[i] - 1:
                p_ones[1], p_linear[1] = o//2, 0
            core.append( blocksize[i] - o + 1 )
            pad_ones.append(tuple(p_ones))
            pad_linear.append(tuple(p_linear))
        # create weights
        weights = np.ones(core, dtype=np.float32)
        weights = np.pad(weights, pad_ones, mode='constant', constant_values=1)
        weights = np.pad(weights, pad_linear, mode='linear_ramp', end_values=0)

        # crop any incomplete blocks (on the ends)
        if np.any( weights.shape != transform.shape[:-1] ):
            crop = tuple(slice(0, s) for s in transform.shape[:-1])
            weights = weights[crop]

        # return the weighted transform
        return transform * weights[..., None]
    # END CLOSURE

    # for small alignments
    if not write_path:

        # submit all alignments to cluster
        futures = cluster.client.map(
            align_single_block, indices,
            static_transform_list=static_transform_list,
        )
        future_keys = [f.key for f in futures]

        # initialize container, monitor progress, write blocks when finished
        transform = np.zeros(fix.shape + (fix.ndim,), dtype=np.float32)
        for batch in as_completed(futures, with_results=True).batches():
            for future, result in batch:
                iii = future_keys.index(future.key)
                transform[indices[iii][3]] += result

    # for large alignments
    else:

        # initialize container and define how to write to it
        shape = fix.shape + (fix.ndim,)
        transform = ut.create_zarr(write_path, shape, zarr_blocks + (fix.ndim,), np.float32)
        def write_block(coords, block):
            transform[coords] = transform[coords] + block

        # split into serial partitions
        a = len(indices) // n_serial_partitions
        b = len(indices) % n_serial_partitions
        partitions = [indices[0:a+b],]
        partitions += [indices[b+i*a:b+(i+1)*a] for i in range(1, n_serial_partitions)]

        # functions for getting neighbor indices
        neighbor_offsets = np.array(list(product([-1, 0, 1], repeat=3)))
        def original_neighbor_indices(index):
            lock_indices = index + neighbor_offsets
            not_too_low = lock_indices.min(axis=1) >= 0
            not_too_high = np.all(lock_indices < nblocks, axis=1)
            lock_indices = lock_indices[not_too_low * not_too_high, :]
            return np.ravel_multi_index(lock_indices.transpose(), nblocks)
        def partition_neighbor_indices(indices, partition_number):
            offset = int(np.sum([len(x) for x in partitions[:partition_number]]))
            indices = original_neighbor_indices(indices) - offset
            not_too_low = indices >= 0
            not_too_high = indices < len(partitions[partition_number])
            return indices[not_too_low * not_too_high]

        # submit partitions in series
        for iii, partition in enumerate(partitions):

            # submit partition to cluster
            futures = cluster.client.map(
                align_single_block, partition,
                static_transform_list=static_transform_list,
            )
            future_keys = [f.key for f in futures]

            # write blocks as parallel as possible
            written = np.zeros(len(futures), dtype=bool)
            while not np.all(written):
                writing_futures = []
                locked = np.zeros(len(futures), dtype=bool)
                for future in futures:
                    jjj = future_keys.index(future.key)
                    if future.done() and not written[jjj] and not locked[jjj]:
                        f = cluster.client.submit(write_block, partition[jjj][3], future)
                        locked[partition_neighbor_indices(partition[jjj][:3], iii)] = True
                        writing_futures.append(f)
                        written[jjj] = True
                wait(writing_futures)

    # remove temporary files and return
    shutil.rmtree(temporary_directory)
    return transform


# TODO: this function not yet refactored
@cluster
def nested_distributed_piecewise_alignment_pipeline(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    block_schedule,
    parameter_schedule=None,
    initial_transform_list=None,
    fix_mask=None,
    mov_mask=None,
    intermediates_path=None,
    cluster=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    Nested piecewise affine alignments.
    Two levels of nesting: outer levels and inner levels.
    Transforms are averaged over inner levels and composed
    across outer levels. See the `block_schedule` parameter
    for more details.

    This method is good at capturing large bends and twists that
    cannot be captured with global rigid and affine alignment.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; if `initial_transform_list` is None then
        `fix.shape` must equal `mov.shape`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    block_schedule : list of lists of tuples of ints.
        Block structure for outer and inner levels.
        Tuples must all be of length `fix.ndim`

        Example:
            [ [(2, 1, 1), (1, 2, 1),],
              [(3, 1, 1), (1, 1, 2),],
              [(4, 1, 1), (2, 2, 1), (2, 2, 2),], ]

            This block schedule specifies three outer levels:
            1) This outer level contains two inner levels:
                1.1) Piecewise rigid+affine with 2 blocks along first axis
                1.2) Piecewise rigid+affine with 2 blocks along second axis
            2) This outer level contains two inner levels:
                2.1) Piecewise rigid+affine with 3 blocks along first axis
                2.2) Piecewise rigid+affine with 2 blocks along third axis
            3) This outer level contains three inner levels:
                3.1) Piecewise rigid+affine with 4 blocks along first axis
                3.2) Piecewise rigid+affine with 4 blocks total: the first
                     and second axis are each cut into 2 blocks
                3.3) Piecewise rigid+affine with 8 blocks total: all axes
                     are cut into 2 blocks

            1.1 and 1.2 are computed (serially) then averaged. This result
            is stored. 2.1 and 2.2 are computed (serially) then averaged.
            This is then composed with the result from the first level.
            This process proceeds for as many levels that are specified.

            Each instance of a piecewise rigid+affine alignment is handled
            by `distributed_piecewise_affine_alignment` and is therefore
            parallelized over blocks on distributed hardware.

    parameter_schedule : list of type dict (default: None)
        Overrides the general parameter `distributed_piecewise_affine_align`
        parameter settings for individual instances. Length of the list
        (total number of dictionaries) must equal the total number of
        tuples in `block_schedule`.

    initial_transform_list : list of ndarrays (default: None)
        A list of transforms to apply to the moving image before running
        twist alignment. If `fix.shape` does not equal `mov.shape`
        then an `initial_transform_list` must be given.

    fix_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the fixed image

    mov_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the moving image

    intermediates_path : string (default: None)
        Path to folder where intermediate results are written.
        The deform, transformed moving image, and transformed
        moving image mask (if given) are stored on disk as npy files.
    
    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    kwargs : any additional arguments
        Passed to `distributed_piecewise_affine_align`

    Returns
    -------
    field : ndarray
        Composition of all outer level transforms. A displacement vector
        field of the shape `fix.shape` + (3,) where the last dimension
        is the vector dimension.
    """

    # set working copies of moving data
    if initial_transform_list is not None:
        current_moving = apply_transform(
            fix, mov, fix_spacing, mov_spacing,
            transform_list=initial_transform_list,
        )
        current_moving_mask = None
        if mov_mask is not None:
            current_moving_mask = apply_transform(
                fix, mov_mask, fix_spacing, mov_spacing,
                transform_list=initial_transform_list,
            )
            current_moving_mask = (current_moving_mask > 0).astype(np.uint8)
    else:
        current_moving = np.copy(mov)
        current_moving_mask = None if mov_mask is None else np.copy(mov_mask)

    # initialize container and Loop over outer levels
    counter = 0  # count each call to distributed_piecewise_affine_align
    deform = np.zeros(fix.shape + (3,), dtype=np.float32)
    for outer_level, inner_list in enumerate(block_schedule):

        # initialize inner container and Loop over inner levels
        ddd = np.zeros_like(deform)
        for inner_level, nblocks in enumerate(inner_list):

            # determine parameter settings
            if parameter_schedule is not None:
                instance_kwargs = {**kwargs, **parameter_schedule[counter]}
            else:
                instance_kwargs = kwargs

            # align
            ddd += distributed_piecewise_alignment_pipeline(
                fix, current_moving,
                fix_spacing, fix_spacing,  # images should be on same grid
                nblocks=nblocks,
                fix_mask=fix_mask,
                mov_mask=current_moving_mask,
                cluster=cluster,
                cluster_kwargs=cluster_kwargs,
                **instance_kwargs,
            )

            # increment counter
            counter += 1

        # take mean
        ddd = ddd / len(inner_list)

        # if not first iteration, compose with existing deform
        if outer_level > 0:
            deform = compose_transforms(deform, ddd, fix_spacing,)
        else:
            deform = ddd

        # combine with initial transforms if given
        if initial_transform_list is not None:
            transform_list = initial_transform_list + [deform,]
        else:
            transform_list = [deform,]

        # update working copy of image
        current_moving = apply_transform(
            fix, mov, fix_spacing, mov_spacing,
            transform_list=transform_list,
        )
        # update working copy of mask
        if mov_mask is not None:
            current_moving_mask = apply_transform(
                fix, mov_mask, fix_spacing, mov_spacing,
                transform_list=transform_list,
            )
            current_moving_mask = (current_moving_mask > 0).astype(np.uint8)

        # write intermediates
        if intermediates_path is not None:
            ois = str(outer_level)
            deform_path = (intermediates_path + '/twist_deform_{}.npy').format(ois)
            image_path = (intermediates_path + '/twist_image_{}.npy').format(ois)
            mask_path = (intermediates_path + '/twist_mask_{}.npy').format(ois)
            np.save(deform_path, deform)
            np.save(image_path, current_moving)
            if mov_mask is not None:
                np.save(mask_path, current_moving_mask)

    # return deform
    return deform
    

