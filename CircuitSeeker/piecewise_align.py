import time, os, shutil
import numpy as np
import dask.array as da
from dask.distributed import as_completed
from ClusterWrap.decorator import cluster
import CircuitSeeker.utility as ut
from CircuitSeeker.transform import apply_transform
from CircuitSeeker.transform import compose_transforms
import zarr
from itertools import product


@cluster
def distributed_piecewise_alignment_pipeline(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    nblocks,
    overlap=0.5,
    fix_mask=None,
    mov_mask=None,
    steps=['rigid', 'affine'],
    random_kwargs={},
    rigid_kwargs={},
    affine_kwargs={},
    deform_kwargs={},
    cluster=None,
    cluster_kwargs={},
    temporary_directory=None,
    write_path=None,
    **kwargs,
):
    """
    Piecewise affine alignment of moving to fixed image.
    Overlapping blocks are given to `affine_align` in parallel
    on distributed hardware. Can include random initialization,
    rigid alignment, and affine alignment.

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

    nblocks : iterable
        The number of blocks to use along each axis.
        Length should be equal to `fix.ndim`

    overlap : float in range [0, 1] (default: 0.5)
        Block overlap size as a percentage of block size

    fix_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the fixed image

    mov_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the moving image
        Due to the distribution aspect, if a mov_mask is provided
        you must also provide a fix_mask. A reasonable choice if
        no fix_mask exists is an array of all ones.

    steps : list of type string (default: ['rigid', 'affine'])
        Flags to indicate which steps to run. An empty list will guarantee
        all affines are the identity. Any of the following may be in the list:
            'random': run `random_affine_search` first
            'rigid': run `affine_align` with rigid=True
            'affine': run `affine_align` with rigid=False
        If all steps are present they are run in the order given above.
        Steps share parameters given to kwargs. Parameters for individual
        steps override general settings with `random_kwargs`, `rigid_kwargs`,
        and `affine_kwargs`. If `random` is in the list, `random_kwargs`
        must be defined.

    random_kwargs : dict (default: {})
        Keyword arguments to pass to `random_affine_search`. This is only
        necessary if 'random' is in `steps`. If so, the following keys must
        be given:
                'max_translation'
                'max_rotation'
                'max_scale'
                'max_shear'
                'random_iterations'
        However any argument to `random_affine_search` may be defined. See
        documentation for `random_affine_search` for descriptions of these
        parameters. If 'random' and 'rigid' are both in `steps` then
        'max_scale' and 'max_shear' must both be 0.

    rigid_kwargs : dict (default: {})
        If 'rigid' is in `steps`, these keyword arguments are passed
        to `affine_align` during the rigid=True step. They override
        any common general kwargs.

    affine_kwargs : dict (default: {})
        If 'affine' is in `steps`, these keyword arguments are passed
        to `affine_align` during the rigid=False (affine) step. They
        override any common general kwargs.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    kwargs : any additional arguments
        Passed to calls `random_affine_search` and `affine_align` calls

    Returns
    -------
    affines : nd array
        Affine matrix for each block. Shape is (X, Y, ..., 4, 4)
        for X blocks along first axis and so on.

    field : nd array
        Local affines stitched together into a displacement field
        Shape is `fix.shape` + (3,) as the last dimension contains
        the displacement vector.
    """

    # compute block size and overlaps
    blocksize = np.array(fix.shape).astype(np.float32) / nblocks
    blocksize = np.ceil(blocksize).astype(np.int16)
    overlaps = np.round(blocksize * overlap).astype(np.int16)

    # ensure temporary directory exists
    if temporary_directory is None:
        temporary_directory = os.getcwd()
    temporary_directory += '/distributed_alignment_temp'
    os.makedirs(temporary_directory)

    # define zarr paths
    fix_zarr_path = temporary_directory + '/fix.zarr'
    mov_zarr_path = temporary_directory + '/mov.zarr'
    fix_mask_zarr_path = temporary_directory + '/fix_mask.zarr'
    mov_mask_zarr_path = temporary_directory + '/mov_mask.zarr'

    # create zarr files
    zarr_blocks = (128,)*fix.ndim
    fix_zarr = ut.numpy_to_zarr(fix, zarr_blocks, fix_zarr_path)
    mov_zarr = ut.numpy_to_zarr(mov, zarr_blocks, mov_zarr_path)
    if fix_mask is not None:
        fix_mask_zarr = ut.numpy_to_zarr(fix_mask, zarr_blocks, fix_mask_zarr_path)
    if mov_mask is not None:
        mov_mask_zarr = ut.numpy_to_zarr(mov_mask, zarr_blocks, mov_mask_zarr_path)

    # determine indices for blocking
    indices = []
    for (i, j, k) in product(*[range(x) for x in nblocks]):
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
    def align_single_block(indices):

        # squeeze the coords and convert to tuple of slices
        block_index = indices[:3]
        coords = indices[3]

        # read the chunks
        fix = fix_zarr[coords]
        mov = mov_zarr[coords]
        fix_mask, mov_mask = None, None
        if os.path.isdir(fix_mask_zarr_path):
            fix_mask = fix_mask_zarr[coords]
        if os.path.isdir(mov_mask_zarr_path):
            mov_mask = mov_mask_zarr[coords]

        # run alignment pipeline
        transform = alignment_pipeline(
            fix, mov, fix_spacing, mov_spacing, steps,
            fix_mask=fix_mask, mov_mask=mov_mask,
            random_kwargs=random_kwargs,
            rigid_kwargs=rigid_kwargs,
            affine_kwargs=affine_kwargs,
            deform_kwargs=deform_kwargs,
        )

        # convert to single vector field
        if isinstance(transform, tuple):
            affine, deform = transform[0], transform[1][1]
            transform = compose_transforms(affine, deform, fix_spacing)
        else:
            transform = ut.matrix_to_displacement_field(
                fix, transform, fix_spacing,
            )

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

        # crop for incomplete blocks (on the ends)
        if np.any( weights.shape != transform.shape[:-1] ):
            crop = tuple(slice(0, s) for s in transform.shape[:-1])
            weights = weights[crop]

        # return the weighted transform
        return transform * weights[..., None]
    # END CLOSURE

    # wait for at least one worker to be fully instantiated
    while ((cluster.client.status == "running") and
           (len(cluster.client.scheduler_info()["workers"]) < 1)):
        time.sleep(1.0)

    # submit all alignments to cluster
    futures = cluster.client.map(align_single_block, indices)
    future_keys = [f.key for f in futures]

    # for small alignments
    if write_path is None:
        # initialize container, monitor progress, write blocks when finished
        transform = np.zeros(fix.shape + (fix.ndim,), dtype=np.float32)
        for batch in as_completed(futures, with_results=True).batches():
            for future, result in batch:
                iii = future_keys.index(future.key)
                transform[indices[iii][3]] += result

    # for large alignments
    # TODO: time is probably going to be an issue here
    #       need to write some of the data in parallel directly from workers
    else:
        # initialize container
        shape = fix.shape + (fix.ndim,)
        zarr_blocks = (128,)*fix.ndim + (fix.ndim,)
        transform = ut.create_zarr(write_path, shape, zarr_blocks, np.float32)
        for future, result in as_completed(futures, with_results=True):
            iii = future_keys.index(future.key)
            transform[indices[iii][3]] = transform[indices[iii][3]] + result

    # remove temporary files
    shutil.rmtree(temporary_directory)

    # return transform
    return transform


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
    

