import numpy as np
import os
from numpy.random import normal
import CircuitSeeker.fileio as csio
import CircuitSeeker.distributed as csd
import CircuitSeeker.stitch as stitch
import dask.array as da
import greedypy.greedypy_registration_method as grm
import SimpleITK as sitk
from scipy.ndimage import find_objects, zoom, minimum_filter


def skipSample(image, spacing, target_spacing):
    """
    """

    ss = np.maximum(np.round(target_spacing / spacing), 1).astype(np.int)
    image = image[::ss[0], ::ss[1], ::ss[2]]
    spacing = spacing * ss
    return image, spacing


def numpyToSITK(image, spacing, origin=None, isVector=False):
    """
    """

    image = sitk.GetImageFromArray(
        image.copy().astype(np.float32),
        isVector=isVector,
    )
    image.SetSpacing(spacing[::-1])
    if origin is None: origin = np.zeros(len(spacing))
    image.SetOrigin(origin[::-1])
    return image


def affineTransformToMatrix(transform):
    """
    """

    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape((3,3))
    matrix[:3, -1] = np.array(transform.GetTranslation())
    return matrix


def matrixToAffineTransform(matrix):
    """
    """

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix[:3, :3].flatten())
    transform.SetTranslation(matrix[:3, -1].squeeze())
    return transform


def sitkAxesToConventionalAxes(matrix):
    """
    """
    
    corrected = np.eye(4)
    corrected[:3, :3] = matrix[:3, :3][::-1, ::-1]
    corrected[:3, -1] = matrix[:3, -1][::-1]
    return corrected


def fieldToDisplacementFieldTransform(field, spacing):
    """
    """

    field = field.astype(np.float64)
    transform = sitk.GetImageFromArray(field, isVector=True)
    transform.SetSpacing(spacing[::-1])
    return sitk.DisplacementFieldTransform(transform)


def matrixToDisplacementField(reference, matrix, spacing):
    """
    """

    nrows, ncols, nstacks = reference.shape
    grid = np.array(np.mgrid[:nrows, :ncols, :nstacks]).transpose(1,2,3,0)
    grid = grid * spacing
    mm, tt = matrix[:3, :3], matrix[:3, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt - grid


def matchHistograms(fixed, moving, bins=1024):
    """
    """

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(bins)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(moving, fixed)


def speckle(image, scale=0.001):
    """
    """

    mn, mx = np.percentile(image, [1, 99])
    stddev = (mx - mn) * scale
    return image + normal(scale=stddev, size=image.shape)


def getLinearRegistrationModel(
    fixed_vox,
    learning_rate,
    iterations,
    metric_sampling_percentage,
    shrink_factors,
    smooth_sigmas,
    metric='MI',
    number_of_histogram_bins=128,
    lcc_radius=8,
):
    """
    """

    # set up registration object
    ncores = int(os.environ["LSB_DJOB_NUMPROC"])  # TODO: LSF specific!
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2*ncores)
    irm = sitk.ImageRegistrationMethod()
    irm.SetNumberOfThreads(2*ncores)
    irm.SetInterpolator(sitk.sitkLinear)

    # metric
    if metric == 'MI':
        irm.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=number_of_histogram_bins,
        )
    elif metric == 'CC':
        irm.SetMetricAsCorrelation()
    elif metric == 'LCC':
        irm.SetMetricAsANTSNeighborhoodCorrelation(lcc_radius)
    irm.SetMetricSamplingStrategy(irm.RANDOM)
    irm.SetMetricSamplingPercentage(metric_sampling_percentage)

    # optimizer
    irm.SetOptimizerAsGradientDescent(
        numberOfIterations=iterations,
        learningRate=learning_rate,
    )
    irm.SetOptimizerScalesFromPhysicalShift()

    # pyramid
    irm.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    irm.SetSmoothingSigmasPerLevel(smoothingSigmas=smooth_sigmas)
    irm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # callback
    def callback(irm):
        level = irm.GetCurrentLevel()
        iteration = irm.GetOptimizerIteration()
        metric = irm.GetMetricValue()
        print("LEVEL: ", level, " ITERATION: ", iteration, " METRIC: ", metric)
    irm.AddCommand(sitk.sitkIterationEvent, lambda: callback(irm))
    return irm


def rigidAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    metric='MI',
    number_of_histogram_bins=128,
    metric_sampling_percentage=1.0,
    shrink_factors=[2,1],
    smooth_sigmas=[1,0],
    learning_rate=1.0,
    number_of_iterations=250,
    target_spacing=2.0,
    fixed_mask=None,
    moving_mask=None,
    foreground_percentage_threshold=0.2,
):
    """
    """

    # if using mask, ensure there is sufficient foreground
    if fixed_mask is not None:
        if np.sum(fixed_mask)/np.prod(fixed_mask.shape) < foreground_percentage_threshold:
            return np.eye(4)
    if moving_mask is not None:
        if np.sum(moving_mask)/np.prod(moving_mask.shape) < foreground_percentage_threshold:
            return np.eye(4)

    # skip sample
    if target_spacing is not None:
        fixed, fixed_vox_ss = skipSample(fixed, fixed_vox, target_spacing)
        moving, moving_vox_ss = skipSample(moving, moving_vox, target_spacing)

        if fixed_mask is not None:
            fixed_mask, _ = skipSample(fixed_mask, fixed_vox, target_spacing)
        if moving_mask is not None:
            moving_mask, _ = skipSample(moving_mask, moving_vox, target_spacing)

        fixed_vox = fixed_vox_ss
        moving_vox = moving_vox_ss

    # convert to sitk images, set spacing
    fixed = numpyToSITK(fixed, fixed_vox)
    moving = numpyToSITK(moving, moving_vox)

    # set up registration object, initialize
    irm = getLinearRegistrationModel(
        fixed_vox,
        learning_rate,
        number_of_iterations,
        metric_sampling_percentage,
        shrink_factors,
        smooth_sigmas,
        metric=metric,
        number_of_histogram_bins=number_of_histogram_bins,
    )
    irm.SetInitialTransform(sitk.Euler3DTransform())

    # set masks
    if fixed_mask is not None:
        fixed_mask = numpyToSITK(fixed_mask, fixed_vox)
        irm.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        moving_mask = numpyToSITK(moving_mask, moving_vox)
        irm.SetMetricMovingMask(moving_mask)

    # execute, return as ndarray
    transform = irm.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )

    # get initial and final metric values
    initial_metric_value = irm.MetricEvaluate(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )
    final_metric_value = irm.GetMetricValue()

    # if registration improved the metric return result, otherwise return default identity
    if final_metric_value < initial_metric_value:
        etransform = sitk.Euler3DTransform()
        etransform.SetParameters(transform.GetParameters())
        return affineTransformToMatrix(etransform)
    else:
        return np.eye(4)


def affineAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    rigid_matrix=None,
    metric='MI',
    number_of_histogram_bins=128,
    metric_sampling_percentage=1.0,
    shrink_factors=[2,1],
    smooth_sigmas=[1,0],
    learning_rate=1.0,
    number_of_iterations=250,
    target_spacing=2.0,
    fixed_mask=None,
    moving_mask=None,
    foreground_percentage_threshold=0.2,
):
    """
    """

    # if using mask, ensure there is sufficient foreground
    if fixed_mask is not None:
        if np.sum(fixed_mask)/np.prod(fixed_mask.shape) < foreground_percentage_threshold:
            return np.eye(4)
    if moving_mask is not None:
        if np.sum(moving_mask)/np.prod(moving_mask.shape) < foreground_percentage_threshold:
            return np.eye(4)

    # skip sample
    if target_spacing is not None:
        fixed, fixed_vox_ss = skipSample(fixed, fixed_vox, target_spacing)
        moving, moving_vox_ss = skipSample(moving, moving_vox, target_spacing)

        if fixed_mask is not None:
            fixed_mask, _ = skipSample(fixed_mask, fixed_vox, target_spacing)
        if moving_mask is not None:
            moving_mask, _ = skipSample(moving_mask, moving_vox, target_spacing)

        fixed_vox = fixed_vox_ss
        moving_vox = moving_vox_ss

    # convert to sitk images, set spacing
    fixed = numpyToSITK(fixed, fixed_vox)
    moving = numpyToSITK(moving, moving_vox)
    rigid = matrixToAffineTransform(rigid_matrix)

    # set up registration object
    irm = getLinearRegistrationModel(
        fixed_vox,
        learning_rate,
        number_of_iterations,
        metric_sampling_percentage,
        shrink_factors,
        smooth_sigmas,
        metric=metric,
        number_of_histogram_bins=number_of_histogram_bins,
    )

    # initialize
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(rigid.GetMatrix())
    affine.SetTranslation(rigid.GetTranslation())
    affine.SetCenter(rigid.GetCenter())
    irm.SetInitialTransform(affine)

    # set masks
    if fixed_mask is not None:
        fixed_mask = numpyToSITK(fixed_mask, fixed_vox)
        irm.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        moving_mask = numpyToSITK(moving_mask, moving_vox)
        irm.SetMetricMovingMask(moving_mask)

    # execute, return as ndarray
    transform = irm.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                            sitk.Cast(moving, sitk.sitkFloat32),
    )

    # get initial and final metric values
    initial_metric_value = irm.MetricEvaluate(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )
    final_metric_value = irm.GetMetricValue()

    # if registration improved the metric return result, otherwise return default identity
    if final_metric_value < initial_metric_value:
        atransform = sitk.AffineTransform(3)
        atransform.SetParameters(transform.GetParameters())
        return affineTransformToMatrix(atransform)
    else:
        return rigid_matrix


def exhaustiveTranslation(
    fixed, moving,
    fixed_vox, moving_vox,
    num_steps, step_sizes,
    fixed_origin=None,
    moving_origin=None,
    block_info=None,
):
    """
    """

    # squeeze any negligible dimensions
    fixed = fixed.squeeze()
    moving = moving.squeeze()

    # convert to sitk images
    fix_itk = numpyToSITK(fixed, fixed_vox, origin=fixed_origin)
    mov_itk = numpyToSITK(moving, moving_vox, origin=moving_origin)

    # initialize image registration method
    ncores = int(os.environ["LSB_DJOB_NUMPROC"])  # TODO: LSF specific!
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2*ncores)
    irm = sitk.ImageRegistrationMethod()
    irm.SetNumberOfThreads(2*ncores)

    # set interpolation
    irm.SetInterpolator(sitk.sitkLinear)

    # set metric
    irm.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=128,
    )

    # set exhaustive optimizer
    irm.SetOptimizerAsExhaustive(num_steps[::-1])
    irm.SetOptimizerScales(step_sizes[::-1])

    # set translation transform
    tx = sitk.TranslationTransform(3)
    irm.SetInitialTransform(tx)

    # keep track of alignment scores
    scores = np.empty(tuple(2*x+1 for x in num_steps[::-1]), dtype=np.float32)
    def callback(irm):
        iteration = irm.GetOptimizerIteration()
        indx = np.unravel_index(iteration, scores.shape, order='F')
        scores[indx[0], indx[1], indx[2]] = irm.GetMetricValue()
    irm.AddCommand(sitk.sitkIterationEvent, lambda: callback(irm))

    # align
    transform = irm.Execute(
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
    if min1 <= min2*1.2:
        trans = (np.array(min1_indx) - num_steps[::-1]) * step_sizes[::-1]

    # return translation in xyz order
    return trans[::-1]


def piecewiseAffineSingleAxis(
    fixed, moving,
    fixed_vox, moving_vox,
    axis, nblocks, pad=16,
    fixed_mask=None, moving_mask=None,
    **kwargs,
):
    """
    """

    # get mask
    mask = np.ones(fixed.shape, dtype=np.uint8)
    if fixed_mask is not None:
        mask = np.logical_and(mask, fixed_mask)
    if moving_mask is not None:
        mask = np.logical_and(mask, moving_mask)

    # get start, stop, and stride
    bounds = find_objects(mask, max_label=1)
    start = max(bounds[0][axis].start - pad, 0)
    stop = min(bounds[0][axis].stop + pad, fixed.shape[axis])
    stride = np.ceil( (stop - start)/(nblocks + 1) ).astype(np.uint16)

    # create weights array
    # blocks overlap by 50%, block size is 2*stride
    sh = np.ones(4, dtype=int)
    sh[axis] = int(2*stride)
    w = np.linspace(0, 1., stride)
    weights = np.concatenate((w, w[::-1])).reshape(sh)

    # piecewise affine align
    piecewise_affine_deform = np.zeros(fixed.shape + (3,))
    for s in range(start, stop-stride, stride):

        # get chunks
        slc = [slice(None, None),]*3
        slc[axis] = slice(s, s+2*stride)
        f, m = fixed[slc], moving[slc]

        # get mask chunks if applicable
        fm = fixed_mask[slc] if fixed_mask is not None else mask[slc]
        mm = moving_mask[slc] if moving_mask is not None else mask[slc]

        # align
        rigid_sitk = rigidAlign(
            f, m, fixed_vox, moving_vox, fixed_mask=fm, moving_mask=mm, **kwargs,
        )
        affine_sitk = affineAlign(
            f, m, fixed_vox, moving_vox, rigid_matrix=rigid_sitk,
            fixed_mask=fm, moving_mask=mm, **kwargs,
        )

        # correct transform for backward SimpleITK axis conventions
        affine = sitkAxesToConventionalAxes(affine_sitk)

        # convert to vector field
        field = matrixToDisplacementField(f, affine, fixed_vox)

        # multiply by weights
        if field.shape[axis] == weights.shape[axis]:
            field = field * weights
        else:
            w_slc = [slice(None, None),]*4
            w_slc[axis] = slice(None, field.shape[axis])
            field = field * weights[w_slc]

        # assign
        piecewise_affine_deform[slc] += field

    # return result
    return piecewise_affine_deform


def distributedPiecewiseAffine(
    fixed, moving,
    fixed_vox, moving_vox,
    nblocks, pad=16,
    fixed_mask=None, moving_mask=None,
    foreground_percentage_threshold=0.2,
    **kwargs,
    ):
    """
    """

    # set up cluster
    with csd.distributedState() as ds:

        # TODO: expose cores/tpw, remove job_extra -P
        ds.initializeLSFCluster(job_extra=["-P scicompsoft"],
        cores=2, memory="30GB", ncpus=2, mem=30000, threads_per_worker=4,
        walltime="48:00",
        )
        ds.initializeClient()
        ds.scaleCluster(njobs=np.prod(nblocks)+1)

        # get mask
        mask = np.ones(fixed.shape, dtype=np.uint8)
        if fixed_mask is not None:
            mask = np.logical_and(mask, fixed_mask)
        if moving_mask is not None:
            mask = np.logical_and(mask, moving_mask)
    
        # get mask bounds and crop inputs
        bounds = find_objects(mask, max_label=1)
        starts = [max(bounds[0][ax].start - pad, 0) for ax in range(3)]
        stops = [min(bounds[0][ax].stop + pad, fixed.shape[ax]) for ax in range(3)]
        slc = tuple([slice(x, y) for x, y in zip(starts, stops)])
        fixed_c = fixed[slc]
        moving_c = moving[slc]
        fm_c = fixed_mask[slc] if fixed_mask is not None else mask[slc]
        mm_c = moving_mask[slc] if moving_mask is not None else mask[slc]

        # construct dask array versions of objects
        fixed_da = da.from_array(fixed_c)
        moving_da = da.from_array(moving_c)
        fm_da = da.from_array(fm_c)
        mm_da = da.from_array(mm_c)

        # compute block size and overlaps
        blocksize = np.array(fixed_c.shape).astype(np.float32) / nblocks
        blocksize = np.ceil(blocksize).astype(np.int16)
        overlaps = blocksize // 2

        # pad the ends to fill in the last blocks
        orig_sh = fixed_da.shape
        pads = [(0, y - x % y) if x % y > 0 else (0, 0) for x, y in zip(orig_sh, blocksize)]
        fixed_da = da.pad(fixed_da, pads).rechunk(tuple(blocksize))
        moving_da = da.pad(moving_da, pads).rechunk(tuple(blocksize))
        fm_da = da.pad(fm_da, pads).rechunk(tuple(blocksize))
        mm_da = da.pad(mm_da, pads).rechunk(tuple(blocksize))

        # closure for rigid align function
        def my_rigid_align(fix, mov, fm, mm):
            rigid = rigidAlign(
                fix, mov, fixed_vox, moving_vox,
                fixed_mask=fm, moving_mask=mm, **kwargs,
            )
            return rigid.reshape((1,1,1,4,4))

        # rigid align all chunks
        rigids = da.map_overlap(
            my_rigid_align, fixed_da, moving_da, fm_da, mm_da,
            depth=tuple(overlaps),
            dtype=np.float32,
            boundary=0,
            trim=False,
            align_arrays=False,
            new_axis=[3,4],
            chunks=[1,1,1,4,4],
        ).compute()

        # closure for affine align function
        def my_affine_align(fix, mov, fm, mm, block_info=None):
            block_idx = block_info[None]['chunk-location']
            rigid = rigids[block_idx[0], block_idx[1], block_idx[2]]
            affine_sitk = affineAlign(
                fix, mov, fixed_vox, moving_vox,
                fixed_mask=fm, moving_mask=mm, rigid_matrix=rigid,
                **kwargs,
            )
            affine = sitkAxesToConventionalAxes(affine_sitk)
            return affine.reshape((1,1,1,4,4))

        # affine align all chunks
        affines = da.map_overlap(
            my_affine_align, fixed_da, moving_da, fm_da, mm_da,
            depth=tuple(overlaps),
            dtype=np.float32,
            boundary=0,
            trim=False,
            align_arrays=False,
            new_axis=[3,4],
            chunks=[1,1,1,4,4],
        ).compute()

        # stitching is very memory intensive, use smaller field size for stitching
        stitch_sh = np.array(orig_sh).astype(np.int16) // 2
        stitch_blocksize = blocksize // 2
        stitch_vox = fixed_vox * 2.

        # convert local affines to displacement vector field
        stitch_field = stitch.local_affine_to_displacement(
            stitch_sh, stitch_vox, affines, stitch_blocksize,
        ).compute()

        # resample field back to correct size
        field = np.empty(orig_sh + (3,), dtype=np.float32)
        stitch_sh = np.array(stitch_field.shape[:-1])
        for i in range(3):
            field[..., i] = zoom(stitch_field[..., i], orig_sh/stitch_sh, order=1)

        # pad back to original shape
        pads = [(x, z-y) for x, y, z in zip(starts, stops, fixed.shape)]
        pads += [(0, 0),]  # vector dimension

        return np.pad(field, pads)


def distributedNestedExhaustiveRigid(
    fixed, moving, mask,
    fixed_vox, moving_vox,
    query_radius,
    search_radius,
    stride,
    step_sizes,
    smooth_sigma,
    nworkers=100,
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
        def my_exhaustive_translation(x, y, block_info=None):
            t = exhaustiveTranslation(
                x, y, fixed_vox, moving_vox,
                num_steps, step_sizes,
                moving_origin=moving_origin,
                block_info=block_info,
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


def deformableAlign(
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


def distributedDeformableAlign(
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
    moving_res = applyTransformToImage(
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
    resampled = applyTransformToImage(
       fixed, moving_res, fixed_vox, fixed_vox, displacement=deformation
    )

    return deformation, resampled


def applyTransformToImage(
    fixed, moving,
    fixed_vox, moving_vox,
    transform_list,
    transform_spacing=None,
    fixed_orig=None, moving_orig=None,
    ):
    """
    """

    # convert images to sitk objects
    dtype = fixed.dtype
    fixed = numpyToSITK(fixed, fixed_vox, fixed_orig)
    moving = numpyToSITK(moving, moving_vox, moving_orig)

    # default transform spacing is fixed voxel spacing 
    if transform_spacing is None:
        transform_spacing = fixed_vox

    # construct transform
    transform = sitk.CompositeTransform(3)
    for t in transform_list:
        if len(t.shape) == 2:
            t = matrixToAffineTransform(t)
        elif len(t.shape) == 4:
            t = fieldToDisplacementFieldTransform(t, transform_spacing)
        transform.AddTransform(t)

    # set up resampler object
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.Cast(fixed, sitk.sitkFloat32))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    # execute, return as numpy array
    resampled = resampler.Execute(sitk.Cast(moving, sitk.sitkFloat32))
    return sitk.GetArrayFromImage(resampled).astype(dtype)

