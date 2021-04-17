import os, sys
import numpy as np
import dask.array as da
import SimpleITK as sitk
import ClusterWrap
import CircuitSeeker.utility as ut
import greedypy.greedypy_registration_method as grm
from scipy.ndimage import minimum_filter, gaussian_filter
from dask_stitch.local_affine import local_affines_to_field


def configure_irm(
    metric='MI', bins=128,
    sampling='regular', sampling_percentage=1.0,
    optimizer='GD', iterations=200, learning_rate=1.0,
    min_step=0.1, max_step=1.0,
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
    rigid=False,
    initial_transform=None,
    initialize_with_centering=False,
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
    if initial_transform is not None and np.all(default == np.eye(4)):
        default = initial_transform

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
        fix, fix_spacing_ss = ut.skip_sample(fix, fix_spacing, alignment_spacing)
        mov, mov_spacing_ss = ut.skip_sample(mov, mov_spacing, alignment_spacing)
        if fix_mask is not None:
            fix_mask, _ = ut.skip_sample(fix_mask, fix_vox, alignment_spacing)
        if mov_mask is not None:
            mov_mask, _ = ut.skip_sample(mov_mask, mov_vox, alignment_spacing)
        fix_spacing = fix_spacing_ss
        mov_spacing = mov_spacing_ss

    # convert to sitk images
    fix = ut.numpy_to_sitk(fix, fix_spacing, origin=fix_origin)
    mov = ut.numpy_to_sitk(mov, mov_spacing, origin=mov_origin)

    # set up registration object
    irm = configure_irm(**kwargs)

    # select initial transform type
    if rigid and initial_transform is None:
        transform = sitk.Euler3DTransform()
    elif rigid and initial_transform is not None:
        transform = ut.matrix_to_euler_transform(initial_transform)
    elif not rigid and initial_transform is None:
        transform = sitk.AffineTransform(3)
    elif not rigid and initial_transform is not None:
        transform = ut.matrix_to_affine_transform(initial_transform)

    # consider initializing with centering
    if initial_transform is None and initialize_with_centering:
        transform = sitk.CenteredTransformInitializer(
            fix, mov, transform,
        )

    # set initial transform
    irm.SetInitialTransform(transform, inPlace=True)

    # set masks
    if fix_mask is not None:
        fix_mask = ut.numpy_to_sitk(fix_mask, fix_spacing, origin=fix_origin)
        irm.SetMetricFixedMask(fix_mask)
    if mov_mask is not None:
        mov_mask = ut.numpy_to_sitk(mov_mask, mov_spacing, origin=mov_origin)
        irm.SetMetricMovingMask(mov_mask)

    # execute alignment
    irm.Execute(
        sitk.Cast(fix, sitk.sitkFloat32),
        sitk.Cast(mov, sitk.sitkFloat32),
    )

    # if centered, convert back to Euler3DTransform object
    if not isinstance(rigid, np.ndarray) and initialize_with_centering:
        transform = sitk.Euler3DTransform(transform)

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
        return ut.affine_transform_to_matrix(transform)
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
    if fix_mask is None:
        fix_mask = np.ones(fix.shape, dtype=np.uint8)
    if mov_mask is None:
        mov_mask = np.ones(mov.shape, dtype=np.uint8) 

    # compute block size and overlaps
    blocksize = np.array(fix.shape).astype(np.float32) / nblocks
    blocksize = np.ceil(blocksize).astype(np.int16)
    overlaps = blocksize // 2

    # set up cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # construct dask array versions of objects
        # extra dimensions to match field of affine matrices
        fix_da = da.from_array(fix)
        mov_da = da.from_array(mov)
        fm_da = da.from_array(fix_mask)
        mm_da = da.from_array(mov_mask)

        # pad the ends to fill in the last blocks
        # blocks must all be exact for stitch to work correctly
        pads = [(0, y - x % y) if x % y > 0
            else (0, 0) for x, y in zip(fix.shape, blocksize)]
        fix_da = da.pad(fix_da, pads).rechunk(tuple(blocksize))
        mov_da = da.pad(mov_da, pads).rechunk(tuple(blocksize))
        fm_da = da.pad(fm_da, pads).rechunk(tuple(blocksize))
        mm_da = da.pad(mm_da, pads).rechunk(tuple(blocksize))

        # closure for affine alignment
        def single_affine_align(fix, mov, fm, mm, block_info=None):
            # rigid alignment
            rigid = affine_align(
                fix, mov, fix_spacing, mov_spacing,
                fix_mask=fm, mov_mask=mm,
                rigid=True,
                **kwargs,
            )
            # affine alignment
            affine = affine_align(
                fix, mov, fix_spacing, mov_spacing,
                fix_mask=fm, mov_mask=mm,
                initial_transform=rigid,
                **kwargs,
            )
            # correct for block origin
            idx = block_info[0]['chunk-location']
            origin = np.maximum(0, blocksize * idx - overlaps)
            origin = origin * fix_spacing
            tl, tr = np.eye(4), np.eye(4)
            tl[:3, -1], tr[:3, -1] = origin, -origin
            affine = np.matmul(tl, np.matmul(affine, tr))
            # return result
            return affine.reshape((1,1,1,4,4))

        # affine align all chunks
        affines = da.map_overlap(
            single_affine_align,
            fix_da, mov_da, fm_da, mm_da,
            depth=tuple(overlaps),
            dtype=np.float32,
            boundary='none',
            trim=False,
            align_arrays=False,
            new_axis=[3,4],
            chunks=[1,1,1,4,4],
        ).compute()

        # stitch local affines into displacement field
        field = local_affines_to_field(
            fix.shape, fix_spacing,
            affines, blocksize, overlaps,
        ).compute()

        return field


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
    fix_itk = ut.numpy_to_sitk(fix, fix_spacing, origin=fix_origin)
    mov_itk = ut.numpy_to_sitk(mov, mov_spacing, origin=mov_origin)

    # define callback: keep track of alignment scores
    scores = np.zeros(tuple(2*x+1 for x in num_steps[::-1]), dtype=np.float32)
    def callback(irm):
        iteration = irm.GetOptimizerIteration()
        indx = np.unravel_index(iteration, scores.shape, order='F')
        scores[indx[0], indx[1], indx[2]] = irm.GetMetricValue()

    # get irm
    kwargs['optimizer'] = 'EX'
    kwargs['num_steps'] = num_steps
    kwargs['step_sizes'] = step_sizes * fix_spacing
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
        trans = np.array(min1_indx[::-1]) - num_steps
        trans = trans * step_sizes * fix_spacing

    # return translation in xyz order
    return trans[::-1]


def piecewise_exhaustive_translation(
    fix, mov,
    fix_spacing, mov_spacing,
    stride,
    query_radius,
    num_steps,
    step_sizes,
    smooth_sigma,
    mask=None,
    cluster_kwargs={},
    **kwargs,
    ):
    """
    """

    # compute search radius in voxels
    search_radius = [q+x*y for q, x, y in zip(query_radius, num_steps, step_sizes)]

    # compute edge pad size
    limit = [x if x > y else y for x, y in zip(search_radius, stride)]

    # get valid sample points as coordinates
    samples = np.zeros_like(fix)
    samples[limit[0]:-limit[0]:stride[0],
            limit[1]:-limit[1]:stride[1],
            limit[2]:-limit[2]:stride[2]] = 1
    if mask is not None:
        samples = samples * mask
    samples = np.nonzero(samples)

    # prepare arrays to hold fixed and moving blocks
    nsamples = len(samples[0])
    fix_blocks_shape = (nsamples,) + tuple(x*2 for x in search_radius)
    mov_blocks_shape = (nsamples,) + tuple(x*2 for x in query_radius)
    fix_blocks = np.empty(fix_blocks_shape, dtype=fix.dtype)
    mov_blocks = np.empty(mov_blocks_shape, dtype=mov.dtype)

    # get context for all sample points
    for i, (x, y, z) in enumerate(zip(samples[0], samples[1], samples[2])):
        fix_blocks[i] = fix[x-search_radius[0]:x+search_radius[0],
                            y-search_radius[1]:y+search_radius[1],
                            z-search_radius[2]:z+search_radius[2]]
        mov_blocks[i] = mov[x-query_radius[0]:x+query_radius[0],
                            y-query_radius[1]:y+query_radius[1],
                            z-query_radius[2]:z+query_radius[2]]

    # compute the query_block origin in physical units
    mov_origin = np.array(search_radius) - query_radius
    mov_origin = mov_origin * fix_spacing

    # set up cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # convert to dask arrays
        fix_blocks_da = da.from_array(
            fix_blocks, chunks=(1,)+fix_blocks.shape[1:],
        )
        mov_blocks_da = da.from_array(
            mov_blocks, chunks=(1,)+mov_blocks.shape[1:],
        )

        # closure for exhaustive translation alignment
        def wrapped_exhaustive_translation(x, y):
            t = exhaustive_translation(
                x, y, fix_spacing, mov_spacing,
                num_steps, step_sizes,
                mov_origin=mov_origin,
                **kwargs,
            )
            return np.array(t).reshape((1, 3))

        # distribute
        translations = da.map_blocks(
            wrapped_exhaustive_translation,
            fix_blocks_da, mov_blocks_da,
            dtype=np.float64, 
            drop_axis=[2,3],
            chunks=[1, 3],
        ).compute()

    # reformat to displacement vector field
    dvf = np.zeros(fix.shape + (3,), dtype=np.float32)
    weights = np.pad([[[1.]]], [(s, s) for s in stride], mode='linear_ramp')
    for t, x, y, z in zip(translations, samples[0], samples[1], samples[2]):
        s = [slice(max(0, x-stride[0]), x+stride[0]+1),
             slice(max(0, y-stride[1]), y+stride[1]+1),
             slice(max(0, z-stride[2]), z+stride[2]+1),]
        dvf[tuple(s)] += t * weights[..., None]

    # smooth
    dvf_s = np.empty_like(dvf)
    for i in range(3):
        dvf_s[..., i] = gaussian_filter(dvf[..., i], smooth_sigma/fix_spacing)

    # return
    return dvf_s







# TODO: still need to refactor deformable

def deformable_align_greedypy(
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


def piecewise_deformable_align_greedypy(
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


def deformable_align(
    fix, mov,
    fix_spacing, mov_spacing,
    control_point_spacing,
    control_point_levels,
    initial_transform=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    **kwargs,
):
    """
    """

    # convert to sitk images
    fix = ut.numpy_to_sitk(fix, fix_spacing, origin=fix_origin)
    mov = ut.numpy_to_sitk(mov, mov_spacing, origin=mov_origin)

    # set up registration object
    irm = configure_irm(**kwargs)

    # set initial moving transform
    if initial_transform is not None:
        if len(initial_transform.shape) == 2:
            it = ut.matrix_to_affine_transform(initial_transform)
        irm.SetMovingInitialTransform(it)

    # get control point grid shape
    fix_size_physical = [sz*sp for sz, sp in zip(fix.GetSize(), fix.GetSpacing())]
    x, y = control_point_spacing, control_point_levels[-1]
    control_point_grid = [max(1, int(sz / (x*y))) for sz in fix_size_physical]

    # set initial transform
    transform = sitk.BSplineTransformInitializer(
        image1=fix, transformDomainMeshSize=control_point_grid, order=3,
    )
    irm.SetInitialTransformAsBSpline(
        transform, inPlace=True, scaleFactors=control_point_levels,
    )

    # set masks
    if fix_mask is not None:
        fix_mask = ut.numpy_to_sitk(fix_mask, fix_spacing, origin=fix_origin)
        irm.SetMetricFixedMask(fix_mask)
    if mov_mask is not None:
        mov_mask = ut.numpy_to_sitk(mov_mask, mov_spacing, origin=mov_origin)
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
        return ut.bspline_to_displacement_field(fix, transform)
    else:
        print("Optimization failed to improve metric")
        print("Returning default")
        sys.stdout.flush()
        return None

