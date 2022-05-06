import sys
import numpy as np
import SimpleITK as sitk
import CircuitSeeker.utility as ut
from CircuitSeeker.configure_irm import configure_irm
from CircuitSeeker.transform import apply_transform
from CircuitSeeker.metrics import patch_mutual_information
from scipy.spatial.transform import Rotation


def skip_sample_images(
    fix,
    mov,
    fix_mask,
    mov_mask,
    fix_spacing,
    mov_spacing,
    alignment_spacing,
):
    """
    Convenience function for skip sampling all inputs to alignment_spacing
    """

    fix, fix_spacing_ss = ut.skip_sample(fix, fix_spacing, alignment_spacing)
    mov, mov_spacing_ss = ut.skip_sample(mov, mov_spacing, alignment_spacing)
    if fix_mask: fix_mask, _ = ut.skip_sample(fix_mask, fix_spacing, alignment_spacing)
    if mov_mask: mov_mask, _ = ut.skip_sample(mov_mask, mov_spacing, alignment_spacing)
    return fix, mov, fix_mask, mov_mask, fix_spacing_ss, mov_spacing_ss


def images_to_sitk(
    fix,
    mov,
    fix_mask,
    mov_mask,
    fix_spacing,
    mov_spacing,
    fix_origin,
    mov_origin,
):
    """
    Convenience function for converting all inputs to sitk images
    """

    fix = sitk.Cast(ut.numpy_to_sitk(fix, fix_spacing, origin=fix_origin), sitk.sitkFloat32)
    mov = sitk.Cast(ut.numpy_to_sitk(mov, mov_spacing, origin=mov_origin), sitk.sitkFloat32)
    if fix_mask: fix_mask = ut.numpy_to_sitk(fix_mask, fix_spacing, origin=fix_origin)
    if mov_mask: mov_mask = ut.numpy_to_sitk(mov_mask, mov_spacing, origin=mov_origin)
    return fix, mov, fix_mask, mov_mask


def random_affine_search(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    random_iterations,
    nreturn=1,
    max_translation=None,
    max_rotation=None,
    max_scale=None,
    max_shear=None,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    static_initial_moving_transform_list=[],
    static_initial_moving_transform_spacing=None,
    static_initial_moving_transform_origin=None,
    use_patch_mutual_information=False,
    print_running_improvements=False,
    **kwargs,
):
    """
    Apply random affine matrices within given bounds to moving image. The best
    scoring affines can be further refined with gradient descent based affine
    alignment. The single best result is returned. This function is intended
    to find good initialization for a full affine alignment obtained by calling
    `affine_align`

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

    random_iterations : int
        The number of random affine matrices to sample

    nreturn : int (default: 1)
        The number of affine matrices to return. The best scoring results
        are returned.

    max_translation : float or tuple of float
        The maximum amplitude translation allowed in random sampling.
        Specified in physical units (e.g. um or mm)
        Can be specified per axis.

    max_rotation : float or tuple of float
        The maximum amplitude rotation allowed in random sampling.
        Specified in radians
        Can be specified per axis.

    max_scale : float or tuple of float
        The maximum amplitude scaling allowed in random sampling.
        Can be specified per axis.

    max_shear : float or tuple of float
        The maximum amplitude shearing allowed in random sampling.
        Can be specified per axis.

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

    static_initial_moving_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform

    static_initial_moving_transform_spacing : np.ndarray or tuple of np.ndarray (default: None)
        Spacing of transforms in static_initial_moving_transform_list
        Only necessary for displacement field transforms.

    static_initial_moving_transform_origin : np.ndarray or tuple of np.ndarray (default: None)
        Origin of transforms in static_initial_moving_transform_list
        Only necessary for displacement field transforms.

    use_patch_mutual_information : bool (default: False)
        Uses a custom metric function in CircuitSeeker.metrics

    print_running_improvements : bool (default: False)
        If True, whenever a better transform is found print the
        iteration, score, and parameters

    **kwargs : any additional arguments
        Passed to `configure_irm` This is how you customize the metric.
        If `use_path_mutual_information` is True this is passed to
        the `patch_mutual_information` function instead.

    Returns
    -------
    best transforms : sorted list of 4x4 numpy.ndarrays (affine matrices)
        best nreturn results, first element of list is the best result
    """

    # function to help generalize parameter limits to 3d
    def expand_param_to_3d(param, null_value):
        if isinstance(param, (int, float)):
            param = (param,) * 2
        if isinstance(param, tuple):
            param += (null_value,)
        return param

    # generalize 2d inputs to 3d
    if fix.ndim == 2:
        fix = fix.reshape(fix.shape + (1,))
        mov = mov.reshape(mov.shape + (1,))
        fix_spacing = tuple(fix_spacing) + (1.,)
        mov_spacing = tuple(mov_spacing) + (1.,)
        max_translation = expand_param_to_3d(max_translation, 0)
        max_rotation = expand_param_to_3d(max_rotation, 0)
        max_scale = expand_param_to_3d(max_scale, 1)
        max_shear = expand_param_to_3d(max_shear, 0)
        if fix_mask: fix_mask = fix_mask.reshape(fix_mask.shape + (1,))
        if mov_mask: mov_mask = mov_mask.reshape(mov_mask.shape + (1,))
        if fix_origin: fix_origin = tuple(fix_origin) + (0.,)
        if mov_origin: mov_origin = tuple(mov_origin) + (0.,)

    # generate random parameters, first row is always identity
    params = np.zeros((random_iterations+1, 12))
    params[:, 6:9] = 1  # default for scale params
    F = lambda mx: 2 * (mx * np.random.rand(random_iterations, 3)) - mx
    if max_translation: params[1:, 0:3] = F(max_translation)
    if max_rotation: params[1:, 3:6] = F(max_rotation)
    if max_scale: params[1:, 6:9] = np.e**F(np.log(max_scale))
    if max_shear: params[1:, 9:] = F(max_shear)

    # define conversion from params to affine transform matrix
    def params_to_affine_matrix(params):
        # translation
        translation = np.eye(4)
        translation[:3, -1] = params[:3]
        # rotation
        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_rotvec(params[3:6]).as_matrix()
        center = np.array(fix.shape) / 2 * fix_spacing
        tl, tr = np.eye(4), np.eye(4)
        tl[:3, -1], tr[:3, -1] = center, -center
        rotation = np.matmul(tl, np.matmul(rotation, tr))
        # scale
        scale = np.diag(tuple(params[6:9]) + (1,))
        # shear
        shx, shy, shz = np.eye(4), np.eye(4), np.eye(4)
        shx[1, 0], shx[2, 0] = params[10], params[11]
        shy[0, 1], shy[2, 1] = params[9], params[11]
        shz[0, 2], shz[1, 2] = params[9], params[10]
        shear = np.matmul(shz, np.matmul(shy, shx))
        # compose
        aff = np.matmul(rotation, translation)
        aff = np.matmul(scale, aff)
        aff = np.matmul(shear, aff)
        return aff
        
    # skip sample to alignment spacing
    if alignment_spacing:
        fix, mov, fix_mask, mov_mask, fix_spacing, mov_spacing = skip_sample_image(
            fix, mov, fix_mask, mov_mask, fix_spacing, mov_spacing, alignment_spacing,
        )

    # a useful value later, storing prevents redundant function calls
    WORST_POSSIBLE_SCORE = np.finfo(np.float64).max

    # define metric evaluation
    if use_patch_mutual_information:
        # wrap patch_mi metric
        def score_affine(affine):
            # apply transform
            aligned = apply_transform(
                fix, mov, fix_spacing, mov_spacing,
                transform_list=[affine,],
                fix_origin=fix_origin,
                mov_origin=mov_origin,
            )
            # mov mask
            mov_mask_aligned = None
            if mov_mask:
                mov_mask_aligned = apply_transform(
                    fix, mov_mask, fix_spacing, mov_spacing,
                    transform_list=[affine,],
                    fix_origin=fix_origin,
                    mov_origin=mov_origin,
                    interpolate_with_nn=True,
                )
            return patch_mutual_information(
                fix, aligned, fix_spacing,
                fix_mask=fix_mask,
                mov_mask=mov_mask_aligned,
                return_metric_image=False,
                **kwargs,
            )

    # use an irm metric
    else:
        # construct irm, set images, masks, transforms
        irm = configure_irm(**kwargs)
        fix, mov, fix_mask, mov_mask = images_to_sitk(
            fix, mov, fix_mask, mov_mask,
            fix_spacing, mov_spacing, fix_origin, mov_origin,
        )
        if fix_mask: irm.SetMetricFixedMask(fix_mask)
        if mov_mask: irm.SetMetricMovingMask(mov_mask)
        T = transform_list_to_composite_transform(
            static_initial_moving_transforms_list,
            static_initial_moving_transform_spacing,
            static_initial_moving_transform_origin,
        )
        irm.SetMovingInitialTransform(T)

        # wrap irm metric
        def score_affine(affine):
            irm.SetMovingInitialTransform(ut.matrix_to_affine_transform(affine))
            try:
                return irm.MetricEvaluate(fix, mov)
            except Exception as e:
                return WORST_POSSIBLE_SCORE

    # score all random affines
    current_best_score = WORST_POSSIBLE_SCORE
    scores = np.empty(random_iterations + 1, dtype=np.float64)
    for iii, ppp in enumerate(params):
        scores[iii] = score_affine(params_to_affine_matrix(ppp))
        if print_running_improvements and scores[iii] < current_best_score:
                current_best_score = scores[iii]
                print(iii, ': ', current_best_score, '\n', ppp)
    sys.stdout.flush()

    # return top results
    partition_indx = np.argpartition(scores, nreturn)[:nreturn]
    params, scores = params[partition_indx], scores[partition_indx]
    return [params_to_affine_matrix(p) for p in params[np.argsort(scores)]]


def affine_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    rigid=False,
    initial_condition=None,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    static_initial_moving_transform_list=[],
    static_initial_moving_transform_spacing=None,
    static_initial_moving_transform_origin=None,
    use_patch_mutual_information=False,
    default=np.eye(4),
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

    jaccard_filter_threshold : float in range [0, 1] (default: None)
        If `jaccard_filter_threshold`, `fix_mask`, and `mov_mask` are all
        defined (i.e. not None), then the Jaccard index between the masks
        is computed. If the index is less than this threshold the alignment
        is skipped and the default is returned. Useful for distributed piecewise
        workflows over heterogenous data.

    default : 4x4 array (default: identity matrix)
        If the optimization fails, print error message but return this value

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
    initial_transform_given = isinstance(initial_condition, np.ndarray)
    if initial_transform_given and np.all(default == np.eye(4)):
        default = initial_condition

    # skip sample to alignment spacing
    if alignment_spacing:
        fix, mov, fix_mask, mov_mask, fix_spacing, mov_spacing = skip_sample_image(
            fix, mov, fix_mask, mov_mask, fix_spacing, mov_spacing, alignment_spacing,
        )

    fix, mov, fix_mask, mov_mask = images_to_sitk(
        fix, mov, fix_mask, mov_mask, fix_spacing, mov_spacing, fix_origin, mov_origin,
    )

    # set up registration object
    irm = configure_irm(**kwargs)
    # set initial static transforms
    T = transform_list_to_composite_transform(
        static_initial_moving_transforms_list,
        static_initial_moving_transform_spacing,
        static_initial_moving_transform_origin,
    )
    irm.SetMovingInitialTransform(T)
    # set transform to optimize
    if rigid and not initial_transform_given:
        transform = sitk.Euler3DTransform()
    elif rigid and initial_transform_given:
        transform = ut.matrix_to_euler_transform(initial_condition)
    elif not rigid and not initial_transform_given:
        transform = sitk.AffineTransform(fix.ndim)
    elif not rigid and initial_transform_given:
        transform = ut.matrix_to_affine_transform(initial_condition)
    if isinstance(initial_condition, str) and initial_condition == "CENTER":
        transform = sitk.CenteredTransformInitializer(fix, mov, transform)
    irm.SetInitialTransform(transform, inPlace=True)
    # set masks
    if fix_mask: irm.SetMetricFixedMask(fix_mask)
    if mov_mask: irm.SetMetricMovingMask(mov_mask)

    # execute alignment, for any exceptions return default
    try:
        initial_metric_value = irm.MetricEvaluate(fix, mov)
        irm.Execute(fix, mov)
        final_metric_value = irm.MetricEvaluate(fix, mov)
    except Exception as e:
        print("Registration failed due to ITK exception:\n", e)
        print("Returning default", flush=True)
        return default

    # if centered, convert back to appropriate transform object
    if isinstance(initial_condition, str) and initial_condition == "CENTER":
        if rigid: transform = sitk.Euler3DTransform(transform)
        else: transform = sitk.AffineTransform(transform)

    # if registration improved metric return result
    # otherwise return default
    if final_metric_value < initial_metric_value:
        print("Registration succeeded", flush=True)
        return ut.affine_transform_to_matrix(transform)
    else:
        print("Optimization failed to improve metric")
        print(f"METRIC VALUES initial: {initial_metric_value} final: {final_metric_value}")
        print("Returning default", flush=True)
        return default


def deformable_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    control_point_spacing,
    control_point_levels,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    static_initial_moving_transform_list=[],
    static_initial_moving_transform_spacing=None,
    static_initial_moving_transform_origin=None,
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

    jaccard_filter_threshold : float in range [0, 1] (default: None)
        If `jaccard_filter_threshold`, `fix_mask`, and `mov_mask` are all
        defined (i.e. not None), then the Jaccard index between the masks
        is computed. If the index is less than this threshold the alignment
        is skipped and the default is returned. Useful for distributed piecewise
        workflows over heterogenous data.

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

    # store initial fixed image shape
    initial_fix_shape = fix.shape

    # skip sample to alignment spacing
    if alignment_spacing:
        fix, mov, fix_mask, mov_mask, fix_spacing, mov_spacing = skip_sample_image(
            fix, mov, fix_mask, mov_mask, fix_spacing, mov_spacing, alignment_spacing,
        )

    # convert to sitk images, float32 type
    fix, mov, fix_mask, mov_mask = images_to_sitk(
        fix, mov, fix_mask, mov_mask, fix_spacing, mov_spacing, fix_origin, mov_origin,
    )

    # set up registration object
    irm = configure_irm(**kwargs)
    # set initial static transforms
    T = transform_list_to_composite_transform(
        static_initial_moving_transforms_list,
        static_initial_moving_transform_spacing,
        static_initial_moving_transform_origin,
    )
    irm.SetMovingInitialTransform(T)

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
    if not default:
        a = transform.GetFixedParameters()
        b = transform.GetParameters()
        params = np.concatenate((a, b))
        field = ut.bspline_to_displacement_field(
            fix, transform, shape=initial_fix_shape,
        )
        default = (params, field)

    # set masks
    if fix_mask: irm.SetMetricFixedMask(fix_mask)
    if mov_mask: irm.SetMetricMovingMask(mov_mask)

    # execute alignment, for any exceptions return default
    try:
        initial_metric_value = irm.MetricEvaluate(fix, mov)
        irm.Execute(fix, mov)
        final_metric_value = irm.MetricEvaluate(fix, mov)
    except Exception as e:
        print("Registration failed due to ITK exception:\n", e)
        print("Returning default", flush=True)
        return default

    # if registration improved metric return result
    # otherwise return default
    if final_metric_value < initial_metric_value:
        a = transform.GetFixedParameters()
        b = transform.GetParameters()
        params = np.concatenate((a, b))
        field = ut.bspline_to_displacement_field(
            fix, transform, shape=initial_fix_shape,
        )
        sys.stdout.flush()
        return params, field
    else:
        print("Optimization failed to improve metric")
        print(f"METRIC VALUES initial: {initial_metric_value} final: {final_metric_value}")
        print("Returning default", flush=True)
        return default


def alignment_pipeline(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    steps,
    initial_transform=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    random_kwargs={},
    rigid_kwargs={},
    affine_kwargs={},
    deform_kwargs={},
    **kwargs,
):
    """
    Compose random, rigid, affine, and deformable alignment with one function call

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

    steps : list of strings
        Alignment steps to run. Options include:
        'random' : run `random_affine_search`
        'rigid' : run `affine_align` with `rigid=True`
        'affine' : run `affine_align`
        'deform' : run `deformable_align`

        Currently you cannot run 'random' and 'rigid' in the same pipeline.
        Hoping to enable this in the future.

    initial_transform : ndarray (default: None)
        An initial transform. This should be a 4x4 affine matrix.
        Not compatible with 'random' in `steps`

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

    random_kwargs : dict (default: None)
        Arguments passed to `random_affine_search`

    rigid_kwargs : dict (default: None)
        Arguments passed to `affine_align` when `rigid=True` (rigid alignment)

    affine_kwargs : dict (default: None)
        Arguments passed to `affine_align` when `rigid=False` (affine alignment)

    deform_kwargs : dict (default: None)
        Arguments passed to `deformable_align`

    **kwargs : any additional keyword arguments
        Global arguments that apply to all alignment steps
        These are overwritten by specific arguments passed via
        `random_kwargs`, `rigid_kwargs`, `affine_kwargs`, and
        `deform_kwargs`

    Returns
    -------
    transform : ndarray or tuple of ndarray
        Transform(s) aligning moving to fixed image.

        If 'deform' is not in `steps` then this is a single 4x4 matrix - all
        steps ('random', 'rigid', and/or 'affine') are composed.

        If 'deform' is in `steps` then this is a tuple. The first element
        is the composed 4x4 affine matrix, the second is a displacement
        vector field with shape equal to fix.shape + (3,)
    """

    # TODO: rigid could be done before random, but random would
    #       need to support an initial_transform; possible
    #       with matrix multiplication (composition)
    if 'random' in steps and 'rigid' in steps:
        message = "cannot do rigid alignment after random affine\n"
        message += "remove either 'random' or 'rigid' from steps"
        raise ValueError(message)

    # set default
    affine = initial_transform if initial_transform is not None else np.eye(4)

    # establish all keyword arguments
    random_kwargs = {**kwargs, **random_kwargs}
    rigid_kwargs = {**kwargs, **rigid_kwargs}
    affine_kwargs = {**kwargs, **affine_kwargs}
    deform_kwargs = {**kwargs, **deform_kwargs}

    # random initialization
    if 'random' in steps:
        affine = random_affine_search(
            fix, mov,
            fix_spacing, mov_spacing,
            fix_mask=fix_mask,
            mov_mask=mov_mask,
            fix_origin=fix_origin,
            mov_origin=mov_origin,
            **random_kwargs,
         )
    # rigid alignment
    if 'rigid' in steps:
        affine = affine_align(
            fix, mov,
            fix_spacing, mov_spacing,
            fix_mask=fix_mask,
            mov_mask=mov_mask,
            fix_origin=fix_origin,
            mov_origin=mov_origin,
            initial_transform=affine,
            rigid=True,
            **rigid_kwargs,
        )
    # affine alignment
    if 'affine' in steps:
        affine = affine_align(
            fix, mov,
            fix_spacing, mov_spacing,
            fix_mask=fix_mask,
            mov_mask=mov_mask,
            fix_origin=fix_origin,
            mov_origin=mov_origin,
            initial_transform=affine,
            **affine_kwargs,
        )
    # deformable align
    if 'deform' in steps:
        deform = deformable_align(
            fix, mov,
            fix_spacing, mov_spacing,
            fix_mask=fix_mask,
            mov_mask=mov_mask,
            fix_origin=fix_origin,
            mov_origin=mov_origin,
            initial_transform=affine,
            **deform_kwargs,
        )
        return affine, deform

    # return affine result
    else:
        return affine


