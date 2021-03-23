import numpy as np
import os, sys
import CircuitSeeker.stitch as stitch
import dask.array as da
import greedypy.greedypy_registration_method as grm
import SimpleITK as sitk
from scipy.ndimage import find_objects, zoom
from scipy.ndimage import minimum_filter, gaussian_filter
import ClusterWrap


def skip_sample(image, spacing, ss_spacing):
    """
    """

    ss = np.maximum(np.round(ss_spacing / spacing), 1).astype(np.int)
    image = image[::ss[0], ::ss[1], ::ss[2]]
    spacing = spacing * ss
    return image, spacing


def numpy_to_sitk(image, spacing, origin=None, vector=False):
    """
    """

    image = sitk.GetImageFromArray(image.copy(), isVector=vector)
    image.SetSpacing(spacing[::-1])
    if origin is None:
        origin = np.zeros(len(spacing))
    image.SetOrigin(origin[::-1])
    return image


def invert_matrix_axes(matrix):
    """
    """

    corrected = np.eye(4)
    corrected[:3, :3] = matrix[:3, :3][::-1, ::-1]
    corrected[:3, -1] = matrix[:3, -1][::-1]
    return corrected


def affine_transform_to_matrix(transform):
    """
    """

    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape((3,3))
    matrix[:3, -1] = np.array(transform.GetTranslation())
    return invert_matrix_axes(matrix)


def matrix_to_affine_transform(matrix):
    """
    """

    matrix_sitk = invert_matrix_axes(matrix)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix_sitk[:3, :3].flatten())
    transform.SetTranslation(matrix_sitk[:3, -1].squeeze())
    return transform


def matrix_to_displacement_field(reference, matrix, spacing):
    """
    """

    nrows, ncols, nstacks = reference.shape
    grid = np.array(np.mgrid[:nrows, :ncols, :nstacks]).transpose(1,2,3,0)
    grid = grid * spacing
    mm, tt = matrix[:3, :3], matrix[:3, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt - grid


def field_to_displacement_field_transform(field, spacing):
    """
    """

    field = field.astype(np.float64)
    transform = numpy_to_sitk(field, spacing, vector=True)
    return sitk.DisplacementFieldTransform(transform)


def configure_irm(
    metric='MI', bins=128,
    sampling='regular', sampling_percentage=1.0,
    optimizer='GD', iterations=200, learning_rate=1.0,
    min_step=1.0, max_step=1.0,
    shrink_factors=[2,1],
    smooth_sigmas=[2,1],
    num_steps=[2, 2, 2],
    step_sizes=[1., 1., 1.],
    callback=None,
):
    """
    """

    # set up registration object
    ncores = int(os.environ["LSB_DJOB_NUMPROC"])  # TODO: LSF specific!
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2*ncores)
    irm = sitk.ImageRegistrationMethod()
    irm.SetNumberOfThreads(2*ncores)

    # set interpolator
    irm.SetInterpolator(sitk.sitkLinear)

    # set metric
    if metric == 'MI':
        irm.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=bins,
        )
    elif metric == 'CC':
        irm.SetMetricAsCorrelation()
    elif metric == 'MS':
        irm.SetMetricAsMeanSquares()

    # set metric sampling
    if sampling == 'regular':
        irm.SetMetricSamplingStrategy(irm.REGULAR)
    elif sampling == 'random':
        irm.SetMetricSamplingStrategy(irm.RANDOM)
    irm.SetMetricSamplingPercentage(sampling_percentage)

    # set optimizer
    if optimizer == 'GD':
        irm.SetOptimizerAsGradientDescent(
            numberOfIterations=iterations,
            learningRate=learning_rate,
        )
        irm.SetOptimizerScalesFromPhysicalShift()
    elif optimizer == 'RGD':
        irm.SetOptimizerAsRegularStepGradientDescent(
            minStep=min_step, learningRate=learning_rate,
            numberOfIterations=iterations,
            maximumStepSizeInPhysicalUnits=max_step,
        )
        irm.SetOptimizerScalesFromPhysicalShift()
    elif optimizer == 'EX':
        irm.SetOptimizerAsExhaustive(num_steps[::-1])
        irm.SetOptimizerScales(step_sizes[::-1])

    # set pyramid
    irm.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    irm.SetSmoothingSigmasPerLevel(smoothingSigmas=smooth_sigmas)
    irm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # set callback function
    if callback is None:
        def callback(irm):
            level = irm.GetCurrentLevel()
            iteration = irm.GetOptimizerIteration()
            metric = irm.GetMetricValue()
            print("LEVEL: ", level, " ITERATION: ", iteration, " METRIC: ", metric)
    irm.AddCommand(sitk.sitkIterationEvent, lambda: callback(irm))

    # return configured irm
    return irm


def affine_align(
    fix, mov,
    fix_spacing, mov_spacing,
    rigid=True,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    default=np.eye(4),
    **kwargs,
):
    """
    """

    # update default if rigid is provided
    if isinstance(rigid, np.ndarray) and np.all(default == np.eye(4)):
        default = rigid

    # if using masks, ensure there is sufficient foreground
    FOREGROUND_PERCENTAGE_THRESHOLD = 0.2
    if fix_mask is not None:
        foreground_percentage = np.sum(fix_mask) / np.prod(fix_mask.shape)
        if foreground_percentage < FOREGROUND_PERCENTAGE_THRESHOLD:
            print("Too little foreground data in fixed image")
            print("Returning default")
            sys.stdout.flush()
            return default
    if mov_mask is not None:
        foreground_percentage = np.sum(mov_mask) / np.prod(mov_mask.shape)
        if foreground_percentage < FOREGROUND_PERCENTAGE_THRESHOLD:
            print("Too little foreground data in moving image")
            print("Returning default")
            sys.stdout.flush()
            return default

    # skip sample to alignment spacing
    if alignment_spacing is not None:
        fix, fix_spacing_ss = skip_sample(fix, fix_spacing, alignment_spacing)
        mov, mov_spacing_ss = skip_sample(mov, mov_spacing, alignment_spacing)
        if fix_mask is not None:
            fix_mask, _ = skip_sample(fix_mask, fix_vox, alignment_spacing)
        if mov_mask is not None:
            mov_mask, _ = skip_sample(mov_mask, mov_vox, alignment_spacing)
        fix_spacing = fix_spacing_ss
        mov_spacing = mov_spacing_ss

    # convert to sitk images
    fix = numpy_to_sitk(fix, fix_spacing, origin=fix_origin)
    mov = numpy_to_sitk(mov, mov_spacing, origin=mov_origin)

    # set up registration object
    irm = configure_irm(**kwargs)

    # set initial transform
    if isinstance(rigid, np.ndarray):
        transform = matrix_to_affine_transform(rigid)
    elif not rigid:
        transform = sitk.AffineTransform(3)
    else:
        transform = sitk.Euler3DTransform()
    irm.SetInitialTransform(transform, inPlace=True)

    # set masks
    if fix_mask is not None:
        fix_mask = numpy_to_sitk(fix_mask, fix_spacing, origin=fix_origin)
        irm.SetMetricFixedMask(fix_mask)
    if mov_mask is not None:
        mov_mask = numpy_to_sitk(mov_mask, mov_spacing, origin=mov_origin)
        irm.SetMetricMovingMask(mov_mask)

    # execute alignment
    irm.Execute(
        sitk.Cast(fix, sitk.sitkFloat32),
        sitk.Cast(mov, sitk.sitkFloat32),
    )

    # get initial and final metric values
    initial_metric_value = irm.MetricEvaluate(
        sitk.Cast(fix, sitk.sitkFloat32),
        sitk.Cast(mov, sitk.sitkFloat32),
    )
    final_metric_value = irm.GetMetricValue()

    # if registration improved metric return result
    # otherwise return default
    if final_metric_value < initial_metric_value:
        sys.stdout.flush()
        return affine_transform_to_matrix(transform)
    else:
        print("Optimization failed to improve metric")
        print("Returning default")
        sys.stdout.flush()
        return default


def piecewise_affine_align(
    fix, mov,
    fix_spacing, mov_spacing,
    nblocks,
    pad=16,
    fix_mask=None,
    mov_mask=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    """

    # get default masks
    joint_mask = np.ones(fix.shape, dtype=np.uint8)
    if fix_mask is not None:
        joint_mask = np.logical_and(joint_mask, fix_mask).astype(np.uint8)
    if mov_mask is not None:
        joint_mask = np.logical_and(joint_mask, mov_mask).astype(np.uint8)

    # get mask bounds and crop inputs
    bounds = find_objects(joint_mask, max_label=1)[0]
    starts = [max(bounds[ax].start - pad, 0) for ax in range(3)]
    stops = [min(bounds[ax].stop + pad, fix.shape[ax]) for ax in range(3)]
    slc = tuple([slice(x, y) for x, y in zip(starts, stops)])
    fix_c = fix[slc]
    mov_c = mov[slc]
    fm_c = fix_mask[slc] if fix_mask is not None else joint_mask[slc]
    mm_c = mov_mask[slc] if mov_mask is not None else joint_mask[slc]

    # compute block size and overlaps
    blocksize = np.array(fix_c.shape).astype(np.float32) / nblocks
    blocksize = np.ceil(blocksize).astype(np.int16)
    overlaps = blocksize // 2

    # set cluster defaults
    if 'cores' not in cluster_kwargs.keys():
        cluster_kwargs['cores'] = 2

    # set up cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:
        cluster.scale_cluster(np.prod(nblocks)+1)

        # construct dask array versions of objects
        fix_da = da.from_array(fix_c)
        mov_da = da.from_array(mov_c)
        fm_da = da.from_array(fm_c)
        mm_da = da.from_array(mm_c)

        # pad the ends to fill in the last blocks
        # blocks must all be exact for stitch to work correctly
        orig_sh = fix_da.shape
        pads = [(0, y - x % y) if x % y > 0
            else (0, 0) for x, y in zip(orig_sh, blocksize)]
        fix_da = da.pad(fix_da, pads).rechunk(tuple(blocksize))
        mov_da = da.pad(mov_da, pads).rechunk(tuple(blocksize))
        fm_da = da.pad(fm_da, pads).rechunk(tuple(blocksize))
        mm_da = da.pad(mm_da, pads).rechunk(tuple(blocksize))

        # closure for rigid align function
        def single_rigid_align(fix, mov, fm, mm):
            rigid = affine_align(
                fix, mov, fix_spacing, mov_spacing,
                fix_mask=fm, mov_mask=mm,
                **kwargs,
            )
            return rigid.reshape((1,1,1,4,4))

        # rigid align all chunks
        rigids = da.map_overlap(
            single_rigid_align, fix_da, mov_da, fm_da, mm_da,
            depth=tuple(overlaps),
            dtype=np.float32,
            boundary='none',
            trim=False,
            align_arrays=False,
            new_axis=[3,4],
            chunks=[1,1,1,4,4],
        ).compute()

        # closure for affine align function
        def single_affine_align(fix, mov, fm, mm, block_info=None):
            block_idx = block_info[0]['chunk-location']
            rigid = rigids[block_idx[0], block_idx[1], block_idx[2]]
            affine = affine_align(
                fix, mov, fix_spacing, mov_spacing,
                fix_mask=fm, mov_mask=mm, rigid=rigid,
                **kwargs,
            )
            return affine.reshape((1,1,1,4,4))

        # affine align all chunks
        affines = da.map_overlap(
            single_affine_align, fix_da, mov_da, fm_da, mm_da,
            depth=tuple(overlaps),
            dtype=np.float32,
            boundary='none',
            trim=False,
            align_arrays=False,
            new_axis=[3,4],
            chunks=[1,1,1,4,4],
        ).compute()

        # stitching is very memory intensive, use smaller field size for stitching
        stitch_sh = np.array(orig_sh).astype(np.int16) // 2
        stitch_blocksize = blocksize // 2
        stitch_spacing = fix_spacing * 2.

        # convert local affines to displacement vector field
        stitch_field = stitch.local_affine_to_displacement(
            stitch_sh, stitch_spacing, affines, stitch_blocksize,
        ).compute()

        # resample field back to correct size
        field = np.empty(orig_sh + (3,), dtype=np.float32)
        stitch_sh = np.array(stitch_field.shape[:-1])
        for i in range(3):
            field[..., i] = zoom(stitch_field[..., i], orig_sh/stitch_sh, order=1)

        # pad back to original shape
        pads = [(x, z-y) for x, y, z in zip(starts, stops, fix.shape)]
        pads += [(0, 0),]  # vector dimension

        return np.pad(field, pads)


def exhaustive_translation(
    fix, mov,
    fix_spacing, mov_spacing,
    num_steps, step_sizes,
    fix_origin=None,
    mov_origin=None,
    peak_ratio=1.2,
    **kwargs,
):
    """
    """

    # squeeze any negligible dimensions
    fix = fix.squeeze()
    mov = mov.squeeze()

    # convert to sitk images
    fix_itk = numpy_to_sitk(fix, fix_spacing, origin=fix_origin)
    mov_itk = numpy_to_sitk(mov, mov_spacing, origin=mov_origin)

    # define callback: keep track of alignment scores
    scores = np.zeros(tuple(2*x+1 for x in num_steps[::-1]), dtype=np.float32)
    def callback(irm):
        iteration = irm.GetOptimizerIteration()
        indx = np.unravel_index(iteration, scores.shape, order='F')
        scores[indx[0], indx[1], indx[2]] = irm.GetMetricValue()
    irm.AddCommand(sitk.sitkIterationEvent, lambda: callback(irm))

    # get irm
    kwargs['optimizer'] = 'EX'
    kwargs['num_steps'] = num_steps
    kwargs['step_sizes'] = step_sizes
    kwargs['callback'] = callback
    irm = configure_irm(**kwargs)

    # set translation transform
    irm.SetInitialTransform(sitk.TranslationTransform(3), inPlace=True)

    # align
    irm.Execute(
        sitk.Cast(fix_itk, sitk.sitkFloat32),
        sitk.Cast(mov_itk, sitk.sitkFloat32),
    )

    # get best two local minima
    peaks = (minimum_filter(scores, size=3) == scores)
    scores[~peaks] = 0
    min1_indx = np.unravel_index(np.argmin(scores), scores.shape)
    min1 = scores[min1_indx[0], min1_indx[1], min1_indx[2]]
    scores[min1_indx[0], min1_indx[1], min1_indx[2]] = 0
    min2_indx = np.unravel_index(np.argmin(scores), scores.shape)
    min2 = scores[min2_indx[0], min2_indx[1], min2_indx[2]]

    # determine if minimum is good enough
    trans = np.zeros(3)
    if min1 <= min2*peak_ratio:
        trans = (np.array(min1_indx) - num_steps[::-1]) * step_sizes[::-1]

    # return translation in xyz order
    return trans[::-1]


def piecewise_exhaustive_translation(
    fixed, moving, mask,
    fixed_vox, moving_vox,
    query_radius,
    search_radius,
    stride,
    step_sizes,
    smooth_sigma,
    nworkers=100,
    **kwargs,
    ):
    """
    """

    # set up cluster
    with csd.distributedState() as ds:

        # TODO: expose cores/tpw, remove job_extra -P
        ds.initializeLSFCluster(project="ahrens", walltime="4:00")
        ds.initializeClient()
        ds.scaleCluster(njobs=nworkers)

        # get valid sample points as coordinates
        samples = np.zeros_like(fixed)
        samples[search_radius[0]:-search_radius[0]:stride[0],
                search_radius[1]:-search_radius[1]:stride[1],
                search_radius[2]:-search_radius[2]:stride[2]] = 1
        samples = np.nonzero(samples * mask)

        # prepare arrays to hold fixed and moving blocks
        nsamples = len(samples[0])
        fixed_blocks_shape = (nsamples,) + tuple(x*2 for x in search_radius)
        moving_blocks_shape = (nsamples,) + tuple(x*2 for x in query_radius)
        fixed_blocks = np.empty(fixed_blocks_shape, dtype=fixed.dtype)
        moving_blocks = np.empty(moving_blocks_shape, dtype=moving.dtype)

        # get context for all sample points
        for i, (x, y, z) in enumerate(zip(samples[0], samples[1], samples[2])):
            fixed_blocks[i] = fixed[x-search_radius[0]:x+search_radius[0],
                                    y-search_radius[1]:y+search_radius[1],
                                    z-search_radius[2]:z+search_radius[2]]
            moving_blocks[i] = moving[x-query_radius[0]:x+query_radius[0],
                                      y-query_radius[1]:y+query_radius[1],
                                      z-query_radius[2]:z+query_radius[2]]

        # convert to dask arrays
        fixed_blocks_da = da.from_array(
            fixed_blocks, chunks=(1,)+fixed_blocks.shape[1:],
        )
        moving_blocks_da = da.from_array(
            moving_blocks, chunks=(1,)+moving_blocks.shape[1:],
        )

        # compute the query_block origin
        moving_origin = np.array(search_radius) - query_radius
        moving_origin = moving_origin * fixed_vox

        # compute the number of steps needed
        num_steps = np.ceil(moving_origin/step_sizes)
        num_steps = [int(x) for x in num_steps]

        # closure for exhaustive translation alignment
        def my_exhaustive_translation(x, y):
            t = exhaustiveTranslation(
                x, y, fixed_vox, moving_vox,
                num_steps, step_sizes,
                moving_origin=moving_origin,
                **kwargs,
            )
            return np.array(t).reshape((1, 3))

        # distribute
        translations = da.map_blocks(
            my_exhaustive_translation,
            fixed_blocks_da, moving_blocks_da,
            dtype=np.float64, 
            drop_axis=[2,3],
            chunks=[1, 3],
        ).compute()

        # reformat to displacement vector field
        dvf = np.zeros(fixed.shape + (3,), dtype=np.float32)
        weights = np.pad([[[1.]]], [(s, s) for s in stride], mode='linear_ramp')
        for t, x, y, z in zip(translations, samples[0], samples[1], samples[2]):
            s = [slice(max(0, x-stride[0]), x+stride[0]+1),
                 slice(max(0, y-stride[1]), y+stride[1]+1),
                 slice(max(0, z-stride[2]), z+stride[2]+1),]
            dvf[tuple(s)] += t * weights[..., None]

        # smooth
        dvf_s = np.empty_like(dvf)
        for i in range(3):
            dvf_s[..., i] = gaussian_filter(dvf[..., i], smooth_sigma/fixed_vox)

        # return
        return dvf_s


def deformable_align(
    fixed, moving,
    fixed_vox, moving_vox,
    radius=32,
    gradient_smoothing=[3.0, 0.0, 1.0, 2.0],
    field_smoothing=[0.5, 0.0, 1.0, 6.0],
    iterations=[200,100],
    shrink_factors=[2,1],
    smooth_sigmas=[1,0],
    step=5.0,
    ):
    """
    """

    register = grm.greedypy_registration_method(
        fixed, fixed_vox,
        moving, moving_vox,
        iterations,
        shrink_factors,
        smooth_sigmas,
        radius=radius,
        gradient_abcd=gradient_smoothing,
        field_abcd=field_smoothing,
    )

    register.mask_values(0)
    register.optimize()
    return register.get_warp()


def piecewise_deformable_align(
    fixed, moving,
    fixed_vox, moving_vox,
    affine_matrix,
    block_size=[96,96,96],
    overlap=16, 
    distributed_state=None,
    ):
    """
    """

    # reasmple moving image with affine
    moving_res = apply_transform(
        fixed, moving,
        fixed_vox, moving_vox,
        matrix=affine_matrix
    )

    # set up cluster
    # TODO: need way to pass distributed_state as context manager?
    with csd.distributedState() as ds:

        # TODO: expose cores/tpw, remove job_extra -P
        ds.initializeLSFCluster(
            job_extra=["-P scicompsoft"],
            ncpus=1,
            cores=1,
            threads_per_worker=2,
            memory="15GB",
            mem=15000,
        )
        ds.initializeClient()
        nchunks = np.ceil(np.array(fixed.shape)/block_size)
        ds.scaleCluster(njobs=np.prod(nchunks))

    # TODO: refactor into a function, generalize w.r.t. dimension, share on github
    # TODO: pad array so large overlaps will work (chunk can't be smaller than overlap)
    # chunk ndarrays onto workers and stack as single dask array
        bs = block_size  # shorthand
        fixed_blocks = [[
            [da.from_array(fixed[i:i+bs[0], j:j+bs[1], k:k+bs[2]])
            for k in range(0, fixed.shape[2], bs[2])]
            for j in range(0, fixed.shape[1], bs[1])]
            for i in range(0, fixed.shape[0], bs[0])]
        fixed_da = da.block(fixed_blocks)
        moving_blocks = [[
            [da.from_array(moving_res[i:i+bs[0], j:j+bs[1], k:k+bs[2]])
            for k in range(0, moving_res.shape[2], bs[2])]
            for j in range(0, moving_res.shape[1], bs[1])]
            for i in range(0, moving_res.shape[0], bs[0])]
        moving_da = da.block(moving_blocks)
    
        # deform all chunks
        # TODO: need way to get registration parameters as input to this function
        compute_blocks = [x + 2*overlap for x in block_size] + [3,]
        deformation = da.map_overlap(
            lambda w,x,y,z: deformableAlign(w, x, y, z),
            fixed_da, moving_da,
            depth=overlap,
            dtype=np.float32,
            chunks=compute_blocks,
            new_axis=[3,],
            align_arrays=False,
            boundary='reflect',
            y=fixed_vox, z=fixed_vox,
        ).compute()

    # TODO: TEMP
    resampled = apply_transform(
       fixed, moving_res, fixed_vox, fixed_vox, displacement=deformation
    )

    return deformation, resampled


def apply_transform(
    fix, mov,
    fix_spacing, mov_spacing,
    transform_list,
    transform_spacing=None,
    fix_origin=None,
    mov_origin=None,
    ):
    """
    """

    # convert images to sitk objects
    dtype = fix.dtype
    fix = numpy_to_sitk(fix, fix_spacing, fix_origin)
    mov = numpy_to_sitk(mov, mov_spacing, mov_origin)

    # default transform spacing is fixed voxel spacing 
    if transform_spacing is None:
        transform_spacing = fix_spacing

    # construct transform
    transform = sitk.CompositeTransform(3)
    for t in transform_list:
        if len(t.shape) == 2:
            t = matrix_to_affine_transform(t)
        elif len(t.shape) == 4:
            t = field_to_displacement_field_transform(t, transform_spacing)
        transform.AddTransform(t)

    # set up resampler object
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.Cast(fix, sitk.sitkFloat32))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    # execute, return as numpy array
    resampled = resampler.Execute(sitk.Cast(mov, sitk.sitkFloat32))
    return sitk.GetArrayFromImage(resampled).astype(dtype)

