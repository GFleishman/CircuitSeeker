import os, sys, psutil
import numpy as np
import dask.array as da
import SimpleITK as sitk
import ClusterWrap
import CircuitSeeker.utility as ut
from CircuitSeeker.transform import apply_transform
import greedypy.greedypy_registration_method as grm
from scipy.ndimage import minimum_filter, gaussian_filter

# TODO: need to refactor stitching
from dask_stitch.local_affine import local_affines_to_field


def configure_irm(
    metric='MI',
    bins=128,
    sampling='regular',
    sampling_percentage=1.0,
    optimizer='GD',
    iterations=200,
    learning_rate=1.0,
    estimate_learning_rate="once",
    min_step=0.1,
    max_step=1.0,
    shrink_factors=[2,1],
    smooth_sigmas=[2.,1.],
    num_steps=[2, 2, 2],
    step_sizes=[1., 1., 1.],
    callback=None,
):
    """
    Wrapper exposing some of the itk::simple::ImageRegistrationMethod API
    Rarely called by the user. Typically used in custom registration functions.

    Parameters
    ----------
    metric : string (default: 'MI')
        The image matching term optimized during alignment
        Options:
            'MI': mutual information
            'CC': correlation coefficient
            'MS': mean squares

    bins : int (default: 128)
        Only used when `metric`='MI'. Number of histogram bins
        for image intensity histograms. Ignored when `metric` is
        'CC' or 'MS'

    sampling : string (default: 'regular')
        How image intensities are sampled during metric calculation
        Options:
            'regular': sample intensities with regular spacing
            'random': sample intensities randomly

    sampling_percentage : float in range [0., 1.] (default: 1.0)
        Percentage of voxels used during metric sampling

    optimizer : string (default 'GD')
        Optimization algorithm used to find a transform
        Options:
            'GD': gradient descent
            'RGD': regular gradient descent
            'EX': exhaustive - regular sampling of transform parameters between
                  given limits

    iterations : int (default: 200)
        Maximum number of iterations at each scale level to run optimization.
        Optimization may still converge early.

    learning_rate : float (default: 1.0)
        Initial gradient descent step size

    estimate_learning_rate : string (default: "once")
        Frequency of estimating the learning rate. Only used if `optimizer`='GD'
        Options:
            'once': only estimate once at the beginning of optimization
            'each_iteration': estimate step size at every iteration
            'never': never estimate step size, `learning_rate` is fixed

    min_step : float (default: 0.1)
        Minimum allowable gradient descent step size. Only used if `optimizer`='RGD'

    max_step : float (default: 1.0)
        Maximum allowable gradient descent step size. Used by both 'GD' and 'RGD'

    shrink_factors : iterable of type int (default: [2, 1])
        Downsampling scale levels at which to optimize

    smooth_sigmas : iterable of type float (default: [2., 1.])
        Sigma of Gaussian used to smooth each scale level image
        Must be same length as `shrink_factors`
        Should be specified in physical units, e.g. mm or um

    num_steps : iterable of type int (default: [2, 2, 2])
        Only used if `optimizer`='EX'
        Number of steps to search in each direction from the initial
        position of the transform parameters

    step_sizes : iterable of type float (default: [1., 1., 1.])
        Only used if `optimizer`='EX'
        Size of step to take during brute force optimization
        Order of parameters and relevant scales should be based on
        the type of transform being optimized

    callable : callable object, e.g. function (default: None)
        A function run at every iteration of optimization
        Should take only the ImageRegistrationMethod object as input: `irm`
        If None then the Level, Iteration, and Metric values are
        printed at each iteration

    Returns
    -------
    irm : itk::simple::ImageRegistrationMethod object
        The configured ImageRegistrationMethod object. Simply needs
        images and a transform type to be ready for optimization.
    """

    # identify number of cores available, assume hyperthreading
    if "LSB_DJOB_NUMPROC" in os.environ:
        ncores = int(os.environ["LSB_DJOB_NUMPROC"])
    else:
        ncores = psutil.cpu_count(logical=False)

    # initialize IRM object, be completely sure nthreads is set
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

    # set metric sampling type and percentage
    if sampling == 'regular':
        irm.SetMetricSamplingStrategy(irm.REGULAR)
    elif sampling == 'random':
        irm.SetMetricSamplingStrategy(irm.RANDOM)
    irm.SetMetricSamplingPercentage(sampling_percentage)

    # set estimate learning rate
    if estimate_learning_rate == "never":
        estimate_learning_rate = irm.Never
    elif estimate_learning_rate == "once":
        estimate_learning_rate = irm.Once
    elif estimate_learning_rate == "each_iteration":
        estimate_learning_rate = irm.EachIteration

    # set optimizer
    if optimizer == 'GD':
        irm.SetOptimizerAsGradientDescent(
            numberOfIterations=iterations,
            learningRate=learning_rate,
            maximumStepSizeInPhysicalUnits=max_step,
            estimateLearningRate=estimate_learning_rate,
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
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    rigid=False,
    initial_transform=None,
    initialize_with_centering=False,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    default=np.eye(4),
    foreground_percentage_threshold=0.2,
    **kwargs,
):
    """
    Affine or rigid alignment of a fixed/moving image pair.
    Lots of flexibility in speed/accuracy trade off.
    Highly configurable and useful in many contexts.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    rigid : bool (default: False)
        Restrict the alignment to rigid motion only

    initial_transform : 4x4 array (default: None)
        An initial rigid or affine matrix from which to initialize
        the optimization

    initialize_with_center : bool (default: False)
        Initialize the optimization center of mass translation
        Cannot be True if `initial_transform` is not None

    alignment_spacing : float (default: None)
        Fixed and moving images are skip sampled to a voxel spacing
        as close as possible to this value. Intended for very fast
        simple alignments (e.g. low amplitude motion correction)

    fix_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the fixed image

    mov_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the moving image

    fix_origin : 1d array (default: None)
        Origin of the fixed image.
        Length must equal `fix.ndim`

    mov_origin : 1d array (default: None)
        Origin of the moving image.
        Length must equal `mov.ndim`

    default : 4x4 array (default: identity matrix)
        If the optimization fails, print error message but return this value

    foreground_percentage_value : float in [0., 1.]
        If masks are given, at least `foreground_percentage_value` percent
        voxels must be in the foreground. Otherwise `default` is returned.
        Useful for distributed workflows where some times are mostly background.

    **kwargs : any additional arguments
        Passed to `configure_irm`
        This is where you would set things like:
        metric, iterations, shrink_factors, and smooth_sigmas

    Returns
    -------
    transform : 4x4 array
        The affine or rigid transform matrix matching moving to fixed
    """

    # update default if an initial transform is provided
    if initial_transform is not None and np.all(default == np.eye(4)):
        default = initial_transform

    # if using masks, ensure there is sufficient foreground
    if fix_mask is not None:
        foreground_percentage = np.sum(fix_mask) / np.prod(fix_mask.shape)
        if foreground_percentage < foreground_percentage_threshold:
            print("Too little foreground data in fixed image")
            print("Returning default")
            sys.stdout.flush()
            return default
    if mov_mask is not None:
        foreground_percentage = np.sum(mov_mask) / np.prod(mov_mask.shape)
        if foreground_percentage < foreground_percentage_threshold:
            print("Too little foreground data in moving image")
            print("Returning default")
            sys.stdout.flush()
            return default

    # skip sample to alignment spacing
    if alignment_spacing is not None:
        fix, fix_spacing_ss = ut.skip_sample(fix, fix_spacing, alignment_spacing)
        mov, mov_spacing_ss = ut.skip_sample(mov, mov_spacing, alignment_spacing)
        if fix_mask is not None:
            fix_mask, _ = ut.skip_sample(fix_mask, fix_spacing, alignment_spacing)
        if mov_mask is not None:
            mov_mask, _ = ut.skip_sample(mov_mask, mov_spacing, alignment_spacing)
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
    if rigid and initialize_with_centering:
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


def distributed_piecewise_affine_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    nblocks,
    overlap=0.5,
    fix_mask=None,
    mov_mask=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    Piecewise rigid + affine alignment of moving to fixed image.
    Overlapping blocks are given to `affine_align` in parallel
    on distributed hardware.

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

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    kwargs : any additional arguments
        Passed to `affine_align` for every block

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

    # set up cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # pad the ends to fill in the last blocks
        # blocks must all be exact for stitch to work correctly
        pads = [(0, y - x % y) if x % y > 0
            else (0, 0) for x, y in zip(fix.shape, blocksize)]
        fix_p = np.pad(fix, pads)
        mov_p = np.pad(mov, pads)

        # pad masks if necessary
        if fix_mask is not None:
            fm_p = np.pad(fix_mask, pads)
        if mov_mask is not None:
            mm_p = np.pad(mov_mask, pads)

        # CONSTRUCT DASK ARRAY VERSION OF OBJECTS
        # fix
        fix_future = cluster.client.scatter(fix_p)
        fix_da = da.from_delayed(
            fix_future, shape=fix_p.shape, dtype=fix_p.dtype
        ).rechunk(tuple(blocksize))

        # mov
        mov_future = cluster.client.scatter(mov_p)
        mov_da = da.from_delayed(
            mov_future, shape=mov_p.shape, dtype=mov_p.dtype
        ).rechunk(tuple(blocksize))

        # fix mask
        if fix_mask is not None:
            fm_future = cluster.client.scatter(fm_p)
            fm_da = da.from_delayed(
                fm_future, shape=fm_p.shape, dtype=fm_p.dtype
            ).rechunk(tuple(blocksize))
        else:
            fm_da = None

        # mov mask
        if mov_mask is not None:
            mm_future = cluster.client.scatter(mm_p)
            mm_da = da.from_delayed(
                mm_future, shape=mm_p.shape, dtype=mm_p.dtype
            ).rechunk(tuple(blocksize))
        else:
            mm_da = None

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

        # TODO: interface may change here
        # stitch local affines into displacement field
        field = local_affines_to_field(
            fix.shape, fix_spacing,
            affines, blocksize, overlaps,
        ).compute()

        # return both formats
        return affines, field


def distributed_twist_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    block_schedule,
    overlap=0.5,
    fix_mask=None,
    mov_mask=None,
    alignment_spacing=None,
    sampling_percentage=1.0,
    iterations=200,
    shrink_factors=[2,1],
    smooth_sigmas=[2.,1.],
    intermediates_path=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    Nested piecewise rigid+affine alignments.
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
        the moving image; `fix.shape` must equal `mov.shape`
        I.e. typically twist alignment is done after
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

    overlap : float or iterable of float in range [0, 1] (default: 0.5)
        The overlap size between blocks. If a single float then the
        overlap percentage is the same for all piecewise rigid+affine
        alignments. If an iterable, the length must equal the total
        number of tuples in `block_schedule`.

    fix_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the fixed image

    mov_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the moving image

    alignment_spacing : float or iterable of float (default: None)
        Fixed and moving images are skip sampled to a voxel spacing
        as close as possible to this value. Good for fast coarse
        alignments. None means use native resolution. If a single
        float then all alignments are done at that spacing. If an
        iterable, the length must equal the total number of tuples
        in `block_schedule`. If an iterable, floats and None can
        both be used.

    sampling_percentage : float or iterable of float (default: 1.0)
        Percentage of voxels used during metric sampling. If a single
        float then all alignments use that percentage. If an iterable
        then length must equal the total number of tuples in
        `block_schedule`.

    iterations : int or iterable of int (default: 200)
        Maximum number of iterations at each scale level to run optimization.
        Optimization may still converge early. If a single int then all
        alignments use that value. If an iterable then length must
        equal the total number of tuples in `block_schedule.

    shrink_factors : iterable of int or iterable of iterables of int (default: [2, 1])
        Downsampling scale levels at which to optimize. If a single iterable
        of type int, then all alignments use those scale levels. If an
        iterable of iterables, then the total number of inner iterables
        must equal the total number of tuples in `block_schedule`.        

    smooth_sigmas : iterable of float or iterable of iterables of float (default: [2., 1.])
        Sigma of Gaussian used to smooth each scale level image. Must be
        same length as `shrink_factors`. Should be specified in physical
        units, e.g. mm or um. If a single iterable of type float, then all
        alignments use those values. If an iterable of iterables, then
        the total number of inner iterables must equal the total number
        of tuples in `block_schedule`.

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
        Passed to `distributed_piecewise_affine_alignment`

    Returns
    -------
    field : ndarray
        Composition of all outer level transforms. A displacement vector
        field of the shape `fix.shape` + (3,) where the last dimension
        is the vector dimension.
    """

    # keep track of each call to distributed_piecewise_affine_align
    counter = 0

    # initialize container and create working copies of image data
    deform = np.zeros(fix.shape + (3,) dtype=np.float32)
    current_moving = np.copy(mov)
    current_moving_mask = None if mov_mask is None else np.copy(mov_mask)

    # Loop over outer levels
    for outer_level, inner_list in enumerate(block_schedule):

        # initialize container for inner level results
        ddd = np.zeros_like(deform)

        # Loop over inner levels
        for inner_level, nblocks in enumerate(inner_list):

            # helper function for determining params
            def set_kwarg(param, param_key, test_type):
                x = param
                if x is not None and type(x) != test_type:
                    x = param[counter]
                kwargs[param_key] = x

            # set kwargs
            set_kwarg(overlap, 'overlap', float)
            set_kwarg(alignment_spacing, 'alignment_spacing', float)
            set_kwarg(sampling_percentage, 'sampling_percentage', float)
            set_kwarg(iterations, 'iterations', int)

            # helper function for determining nested params
            def set_nested_kwarg(param, param_key, test_type):
                x = param
                if x is not None and type(x[0]) != test_type:
                    x = param[counter]
                kwargs[param_key] = x

            # set nested kwargs
            set_nested_kwarg(shrink_factors, 'shrink_factors', int)
            set_nested_kwarg(smooth_sigmas, 'smooth_sigmas', float)

            # align
            ddd += distributed_piecewise_affine_align(
                fix, current_moving,
                fix_spacing, mov_spacing,
                nblocks=nblocks,
                fix_mask=fix_mask,
                mov_mask=mov_mask,
                cluster_kwargs=cluster_kwargs,
                **kwargs,
            )[1]  # only want the field

        # take mean
        ddd = ddd / len(inner_list)

        # compose with existing deform (unless first outer iteration)
        # TODO: add field composition to transform module
        if outer_level > 0:
            for iii in range(3):
                ppp = int(np.ceil(np.max(np.abs(ddd[..., iii])) / fix_spacing[iii]))
                padded = np.pad(deform[..., iii], [(ppp, ppp),]*3, mode='edge')
                deform[..., iii] = apply_transform(
                    deform[..., iii], padded, fix_spacing, fix_spacing,
                    transform_list=[ddd,],
                    mov_origin=fix_spacing * -ppp,
                )
        deform = deform + ddd

        # update working copies
        current_moving = apply_transform(
            fix, mov, fix_spacing, mov_spacing,
            transform_list=[deform,],
        )
        current_mask = apply_transform(
            fix, mov_mask, fix_spacing, mov_spacing,
            transform_list=[deform,],
        )
        current_mask = (current_mask > 0).astype(bool)

        # increment counter
        counter += 1

        # write intermediates
        if intermediates_path is not None:
            ois, iis = str(outer_level), str(inner_level)
            deform_path = (intermediates_path + '/deform_o{}_i{}.npy').format(ois, iis)
            image_path = (intermediates_path + '/twisted_o{}_i{}.npy').format(ois, iis)
            mask_path = (intermediates_path + '/twisted_mask_o{}_i{}.npy').format(ois, iis)
            np.save(deform_path, deform)
            np.save(image_path, current_moving)
            np.save(mask_path, current_mask)

    # return deform
    return deform
    

def exhaustive_translation(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    num_steps,
    step_sizes,
    fix_origin=None,
    mov_origin=None,
    peak_ratio=1.2,
    **kwargs,
):
    """
    Brute force translation alignment; grid search over translations

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    num_steps : iterable of type int
        Number of steps to search in each direction

    step_sizes : iterable of type int
        Size of step to take during brute force optimization
        Specified in voxel units

    fix_origin : 1d array (default: None)
        Origin of the fixed image.
        Length must equal `fix.ndim`

    mov_origin : 1d array (default: None)
        Origin of the moving image.
        Length must equal `mov.ndim`

    peak_ratio : float (default: 1.2)
        Brute force optimization travels through many local minima
        For a result to valid, the ratio of the deepest two minima
        must exceed `peak_ratio`

    kwargs : any additional arguments
        Passed to `configure_irm`

    Returns
    -------
    translation : 1d array
        The translation parameters for each axis
    
    """

    # convert to sitk images
    fix_itk = ut.numpy_to_sitk(fix, fix_spacing, origin=fix_origin)
    mov_itk = ut.numpy_to_sitk(mov, mov_spacing, origin=mov_origin)

    # define callback: keep track of alignment scores
    scores_shape = tuple(2*x+1 for x in num_steps[::-1])
    scores = np.zeros(scores_shape, dtype=np.float32)
    def callback(irm):
        iteration = irm.GetOptimizerIteration()
        indx = np.unravel_index(iteration, scores_shape, order='F')
        scores[indx[0], indx[1], indx[2]] = irm.GetMetricValue()

    # get irm
    kwargs['optimizer'] = 'EX'
    kwargs['num_steps'] = num_steps
    kwargs['step_sizes'] = step_sizes * fix_spacing
    kwargs['callback'] = callback
    kwargs['shrink_factors'] = [1,]
    kwargs['smooth_sigmas'] = [0,]
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
    scores[~peaks] = np.finfo('f').max
    min1_indx = np.unravel_index(np.argmin(scores), scores.shape)
    min1 = scores[min1_indx[0], min1_indx[1], min1_indx[2]]
    scores[min1_indx[0], min1_indx[1], min1_indx[2]] = np.finfo('f').max
    min2_indx = np.unravel_index(np.argmin(scores), scores.shape)
    min2 = scores[min2_indx[0], min2_indx[1], min2_indx[2]]

    # determine if minimum is good enough
    trans = np.zeros(3)
    a, b = sorted([abs(min1), abs(min2)])
    if b / a >= peak_ratio:
        trans = np.array(min1_indx[::-1]) - num_steps
        trans = trans * step_sizes * fix_spacing

    # return translation in xyz order
    return trans


def distributed_piecewise_exhaustive_translation(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
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
    Piecewise brute force/exhaustive translation of moving to fixed image.
    `exhaustive_translation` is run on (possibly overlapping) blocks in
    parallel on distributed hardware.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.shape` must equal `mov.shape`
        I.e. typically piecewise exhaustive translation is done after
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

    stride : iterable of type int
        Per axis spacing between centers of adjacent blocks.
        Length must be equal to `fix.ndims`

    query_radius : iterable of type int
        Per axis radius of moving image block size.
        Length must be equal to `fix.ndims`

    num_steps : iterable of type int
        Number of steps to search in each direction

    step_sizes : iterable of type int
        Size of step to take during brute force optimization
        Specified in voxel units

    smooth_sigma : float
        Size of Gaussian smoothing kernel applied to final displacement
        vector field representation of result. Makes local translations
        more consistent with each other. Specified in physical units.
        Set to 0 for no smoothing.

    mask : ndarray
        Only align blocks whose centers are within this mask.
        `mask.shape` should equal `fix.shape`

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    kwargs : any additional arguments
        Passed to `exhaustive_translation`

    Returns
    -------
    field : nd array
        Local translations stitched together into a displacement field
        Shape is `fix.shape` + (3,) as the last dimension contains
        the displacement vector.
    """

    # compute search radius in voxels
    search_radius = [q+x*y for q, x, y in zip(query_radius, num_steps, step_sizes)]

    # get edge limits
    limit = [x if x > y else y for x, y in zip(search_radius, stride)]

    # get valid sample points as coordinates
    samples = np.zeros(fix.shape, dtype=bool)
    samples[limit[0]:-limit[0]:stride[0],
            limit[1]:-limit[1]:stride[1],
            limit[2]:-limit[2]:stride[2]] = 1
    if mask is not None:
        samples = samples * mask
    samples = np.nonzero(samples)

    # prepare arrays to hold fixed and moving blocks
    nsamples = len(samples[0])
    fix_blocks_shape = (nsamples,) + tuple(x*2+1 for x in search_radius)
    mov_blocks_shape = (nsamples,) + tuple(x*2+1 for x in query_radius)
    fix_blocks = np.empty(fix_blocks_shape, dtype=fix.dtype)
    mov_blocks = np.empty(mov_blocks_shape, dtype=mov.dtype)

    # get context for all sample points
    for i, (x, y, z) in enumerate(zip(samples[0], samples[1], samples[2])):
        fix_blocks[i] = fix[x-search_radius[0]:x+search_radius[0]+1,
                            y-search_radius[1]:y+search_radius[1]+1,
                            z-search_radius[2]:z+search_radius[2]+1]
        mov_blocks[i] = mov[x-query_radius[0]:x+query_radius[0]+1,
                            y-query_radius[1]:y+query_radius[1]+1,
                            z-query_radius[2]:z+query_radius[2]+1]

    # compute the query_block origin in physical units
    mov_origin = np.array(search_radius) - query_radius
    mov_origin = mov_origin * fix_spacing

    # set up cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # fix
        fix_blocks_future = cluster.client.scatter(fix_blocks)
        fix_blocks_da = da.from_delayed(
            fix_blocks_future,
            shape=fix_blocks.shape,
            dtype=fix_blocks.dtype
        ).rechunk((1,)+fix_blocks.shape[1:])

        # mov
        mov_blocks_future = cluster.client.scatter(mov_blocks)
        mov_blocks_da = da.from_delayed(
            mov_blocks_future,
            shape=mov_blocks.shape,
            dtype=mov_blocks.dtype
        ).rechunk((1,)+mov_blocks.shape[1:])

        # closure for exhaustive translation alignment
        def wrapped_exhaustive_translation(x, y):
            t = exhaustive_translation(
                x.squeeze(), y.squeeze(),
                fix_spacing, mov_spacing,
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
        s = [slice(x-stride[0], x+stride[0]+1),
             slice(y-stride[1], y+stride[1]+1),
             slice(z-stride[2], z+stride[2]+1),]
        dvf[tuple(s)] += t * weights[..., None]

    # smooth and return
    if smooth_sigma > 0:
        dvf_s = np.empty_like(dvf)
        for i in range(3):
            dvf_s[..., i] = gaussian_filter(dvf[..., i], smooth_sigma/fix_spacing)
        return dvs_s
    else:
        return dvf


def deformable_align_greedypy(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    radius,
    gradient_smoothing=[3.0, 0.0, 1.0, 2.0],
    field_smoothing=[0.5, 0.0, 1.0, 6.0],
    iterations=[200,100],
    shrink_factors=[2,1],
    smooth_sigmas=[1,0],
    step=5.0,
):
    """
    Deformable alignment of moving to fixed image. Does not use
    itk::simple::ImageRegistrationMethod API, so parameter
    formats are different. See greedypy package for more details.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.shape` must equal `mov.shape`
        I.e. typically deformable alignment is done after
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

    radius : int
        greedypy uses local correlation as an image matching metric.
        This is the radius of neighborhoods used for local correlation.

    gradient_smoothing : list of 4 floats (default: [3., 0., 1., 2.])
        Parameters for smoothing the gradient of the image matching
        metric at each iteration.
        greedypy uses the differential operator format for smoothing.
        These parameters are a, b, c, and d in: (a*lap + b*graddiv + c)^d
        where lap is the Laplacian operator and graddiv is the gradient
        of divergence operator.

    field_smoothing : list of 4 floats (default: [.5, 0., 1., 6.])
        Parameters for smoothing the total field at every iteration.
        See `gradient_smoothing` for more details.

    iterations : iterable of type int (default: [200, 100])
        The maximum number of iterations to run at each scale level.
        Optimization may still converge early.

    shrink_factors : iterable of type int (default: [2, 1])
        Downsampling factors for each scale level.
        `len(shrink_facors)` must equal `len(iterations)`.

    smooth_sigmas : iterable of type float (default: [1., 0.])
        Sigma of Gaussian smoothing kernel applied before downsampling
        images at each scale level. `len(smooth_sigmas)` must equal
        `len(iterations)`

    step : float (default: 5.)
        Gradient descent step size

    Returns
    -------
        field : ndarray
            Displacement vector field matching moving image to fixed
    """

    register = grm.greedypy_registration_method(
        fix,
        fix_spacing,
        mov,
        mov_spacing,
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


def distributed_piecewise_deformable_align_greedypy(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    nblocks,
    radius,
    overlap=0.5, 
    cluster_kwargs={},
    **kwargs,
):
    """
    Deformable alignment of overlapping blocks. Blocks are run
    through `greedypy_deformable_align` in parallel on distributed
    hardware.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.shape` must equal `mov.shape`
        I.e. typically deformable alignment is done after
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

    radius : int
        greedypy uses local correlation as an image matching metric.
        This is the radius of neighborhoods used for local correlation.

    overlap : float in range [0, 1] (default: 0.5)
        Block overlap size as a percentage of block size

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    kwargs : any additional arguments
        Passed to `greedypy_deformable_align` for every block

    Returns
    -------
        field : ndarray
            Displacement vector field stitched from local block alignment
    """

    # compute block size and overlaps
    blocksize = np.array(fix.shape).astype(np.float32) / nblocks
    blocksize = np.ceil(blocksize).astype(np.int16)
    overlaps = np.round(blocksize * overlap).astype(np.int16)

    # set up cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # pad the ends to fill in the last blocks
        # blocks must all be exact for stitch to work correctly
        pads = [(0, y - x % y) if x % y > 0
            else (0, 0) for x, y in zip(fix.shape, blocksize)]
        fix_p = np.pad(fix, pads)
        mov_p = np.pad(mov, pads)

        # scatter fix data to cluster
        fix_future = cluster.client.scatter(fix_p)
        fix_da = da.from_delayed(
            fix_future, shape=fix_p.shape, dtype=fix_p.dtype
        ).rechunk(tuple(blocksize))

        # scatter mov data to cluster
        mov_future = cluster.client.scatter(mov_p)
        mov_da = da.from_delayed(
            mov_future, shape=mov_p.shape, dtype=mov_p.dtype
        ).rechunk(tuple(blocksize))

        # closure for greedypy_deformable_align
        def single_deformable_align(fix, mov):
            return greedypy_deformable_align(
                fix, mov, fix_spacing, mov_spacing,
                radius, **kwargs
            ).reshape((1,)*fix.ndim + fix.shape + (3,))

        # determine output chunk shape
        output_chunks = tuple(x+2*y for x, y in zip(blocksize, overlaps))
        output_chunks = (1,)*fix.ndim + output_chunks + (3,)

        # deform all chunks
        fields = da.map_overlap(
            single_deformable_align,
            fix_da, mov_da,
            depth=tuple(overlaps),
            dtype=np.float32,
            boundary=0,
            trim=False,
            align_arrays=False,
            new_axis=[3,4,5,6],
            chunks=output_chunks,
        ).compute()

        # TODO need a stitching function here
        # stitch local fields
        field = stitch_fields(
            fix.shape, fix_spacing,
            affines, blocksize, overlaps,
        ).compute()

        # return
        return field
            

def bspline_deformable_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    control_point_spacing,
    control_point_levels,
    initial_transform=None,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    default=None,
    **kwargs,
):
    """
    Register moving to fixed image with a bspline parameterized deformation field

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.

    control_point_spacing : float
        The spacing in physical units (e.g. mm or um) between control
        points that parameterize the deformation. Smaller means
        more precise alignment, but also longer compute time. Larger
        means shorter compute time and smoother transform, but less
        precise.

    control_point_levels : list of type int
        The optimization scales for control point spacing. E.g. if
        `control_point_spacing` is 100.0 and `control_point_levels`
        is [1, 2, 4] then method will optimize at 400.0 units control
        points spacing, then optimize again at 200.0 units, then again
        at the requested 100.0 units control point spacing.
    
    initial_transform : 4x4 array (default: None)
        An initial rigid or affine matrix from which to initialize
        the optimization

    alignment_spacing : float (default: None)
        Fixed and moving images are skip sampled to a voxel spacing
        as close as possible to this value. Intended for very fast
        simple alignments (e.g. low amplitude motion correction)

    fix_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the fixed image

    mov_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the moving image

    fix_origin : 1d array (default: None)
        Origin of the fixed image.
        Length must equal `fix.ndim`

    mov_origin : 1d array (default: None)
        Origin of the moving image.
        Length must equal `mov.ndim`

    default : any object (default: None)
        If optimization fails to improve image matching metric,
        print an error but also return this object. If None
        the parameters and displacement field for an identity
        transform are returned.

    **kwargs : any additional arguments
        Passed to `configure_irm`
        This is where you would set things like:
        metric, iterations, shrink_factors, and smooth_sigmas

    Returns
    -------
    params : 1d array
        The complete set of control point parameters concatenated
        as a 1d array.

    field : ndarray
        The displacement field parameterized by the bspline control
        points
    """

    # skip sample to alignment spacing
    if alignment_spacing is not None:
        fix, fix_spacing_ss = ut.skip_sample(fix, fix_spacing, alignment_spacing)
        mov, mov_spacing_ss = ut.skip_sample(mov, mov_spacing, alignment_spacing)
        if fix_mask is not None:
            fix_mask, _ = ut.skip_sample(fix_mask, fix_spacing, alignment_spacing)
        if mov_mask is not None:
            mov_mask, _ = ut.skip_sample(mov_mask, mov_spacing, alignment_spacing)
        fix_spacing = fix_spacing_ss
        mov_spacing = mov_spacing_ss

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

    # store initial transform coordinates as default
    if default is None:
        fp = transform.GetFixedParameters()
        pp = transform.GetParameters()
        default_params = np.array(list(fp) + list(pp))
        default_field = ut.bspline_to_displacement_field(fix, transform)
        default = (default_params, default_field)

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
        fp = transform.GetFixedParameters()
        pp = transform.GetParameters()
        params = np.array(list(fp) + list(pp))
        field = ut.bspline_to_displacement_field(fix, transform)
        return params, field
    else:
        print("Optimization failed to improve metric")
        print("Returning default")
        sys.stdout.flush()
        return default

