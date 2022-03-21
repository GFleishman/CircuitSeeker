import sys, time
import numpy as np
import dask.array as da
import SimpleITK as sitk
from ClusterWrap.decorator import cluster
import CircuitSeeker.utility as ut
from CircuitSeeker._wrap_irm import configure_irm
from CircuitSeeker.transform import apply_transform
from CircuitSeeker.transform import compose_affine_and_displacement_vector_field
from CircuitSeeker.transform import compose_displacement_vector_fields
from CircuitSeeker.quality import jaccard_filter
from CircuitSeeker.metrics import patch_mutual_information
from scipy.spatial.transform import Rotation


def random_affine_search(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    max_translation,
    max_rotation,
    max_scale,
    max_shear,
    random_iterations,
    affine_align_best=0,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    jaccard_filter_threshold=None,
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

    max_translation : float
        The maximum amplitude translation allowed in random sampling.
        Specified in physical units (e.g. um or mm)

    max_rotation : float
        The maximum amplitude rotation allowed in random sampling.
        Specified in radians

    max_scale : float
        The maximum amplitude scaling allowed in random sampling.

    max_shear : float
        The maximum amplitude shearing allowed in random sampling.

    random_iterations : int
        The number of random affine matrices to sample

    affine_align_best : int (default: 0)
        The best `affine_align_best` random affine matrices are refined
        by calling `affine_align` setting the random affine as the
        `initial_transform`. This is parameterized through **kwargs.

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

    **kwargs : any additional arguments
        Passed to `configure_irm` to score random affines
        Also passed to `affine_align` for gradient descent
        based refinement

    Returns
    -------
    transform : 4x4 array
        The (refined) random affine matrix best initializing a match of
        the moving image to the fixed. Should be further refined by calling
        `affine_align`.
    """

    # check jaccard index
    a = jaccard_filter_threshold is not None
    b = fix_mask is not None
    c = mov_mask is not None
    if a and b and c:
        if not jaccard_filter(fix_mask, mov_mask, jaccard_filter_threshold):
            print("Masks failed jaccard_filter")
            print("Returning default")
            return np.eye(4)

    # define conversion from params to affine transform
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
        scale = np.diag( list(params[6:9]) + [1,])

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
        
    # generate random parameters, first row is always identity
    params = np.zeros((random_iterations+1, 12))
    params[:, 6:9] = 1  # default for scale params
    F = lambda mx: 2 * mx * np.random.rand(random_iterations, 3) - mx
    if max_translation != 0: params[1:, 0:3] = F(max_translation)
    if max_rotation != 0: params[1:, 3:6] = F(max_rotation)
    if max_scale != 1: params[1:, 6:9] = np.e**F(np.log(max_scale))
    if max_shear != 0: params[1:, 9:] = F(max_shear)

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
            if mov_mask is not None:
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

    # otherwise score entire image domain
    else:

        # convert to float32 sitk images
        fix_sitk = ut.numpy_to_sitk(fix, fix_spacing, origin=fix_origin)
        mov_sitk = ut.numpy_to_sitk(mov, mov_spacing, origin=mov_origin)
        fix_sitk = sitk.Cast(fix_sitk, sitk.sitkFloat32)
        mov_sitk = sitk.Cast(mov_sitk, sitk.sitkFloat32)

        # create irm to use metric object
        irm = configure_irm(**kwargs)

        # set masks
        if fix_mask is not None:
            fix_mask_sitk = ut.numpy_to_sitk(fix_mask, fix_spacing, origin=fix_origin)
            irm.SetMetricFixedMask(fix_mask_sitk)
        if mov_mask is not None:
            mov_mask_sitk = ut.numpy_to_sitk(mov_mask, mov_spacing, origin=mov_origin)
            irm.SetMetricMovingMask(mov_mask_sitk)

        # wrap full image mi
        def score_affine(affine):

            # reformat affine as tranfsorm and give to irm
            affine = ut.matrix_to_affine_transform(affine)
            irm.SetMovingInitialTransform(affine)

            # get the metric value
            try:
                return irm.MetricEvaluate(fix_sitk, mov_sitk)
            except Exception as e:
                return np.finfo(scores.dtype).max

    # score all random affines
    fail_count = 0
    current_best_score = 0
    scores = np.empty(random_iterations + 1)
    for iii, ppp in enumerate(params):
        aff = params_to_affine_matrix(ppp)
        scores[iii] = score_affine(aff)
        if scores[iii] == np.finfo(scores.dtype).max:
            fail_count += 1

        # print running improvements
        if print_running_improvements:
            if scores[iii] < current_best_score:
                current_best_score = scores[iii]
                print(iii, ': ', current_best_score, '\n', aff, '\n', flush=True)

        # check for excessive failure
        if fail_count >= 10 or fail_count >= random_iterations + 1:
            print("Random search failed due to ITK exceptions")
            print("Returning default")
            return np.eye(4)

    # sort
    params = params[np.argsort(scores)]

    # gradient descent based refinements
    if affine_align_best == 0:
        return params_to_affine_matrix(params[0])

    else:
        # container to hold the scores
        scores = np.empty(affine_align_best)
        fail_count = 0  # keep track of failures
        for iii in range(affine_align_best):

            # gradient descent affine alignment
            aff = params_to_affine_matrix(params[iii])
            aff = affine_align(
               fix, mov, fix_spacing, mov_spacing,
               initial_transform=aff,
               fix_mask=fix_mask,
               mov_mask=mov_mask,
               fix_origin=fix_origin,
               mov_origin=mov_origin,
               alignment_spacing=None,  # already done in this function
               **kwargs,
            )

            # score the result
            scores[iii] = score_affine(aff)
            if fail_count >= affine_align_best:
                print("Random search failed due to ITK exceptions")
                print("Returning default")
                return np.eye(4)

        # return the best one
        return params_to_affine_matrix(params[np.argmin(scores)])
        

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
    jaccard_filter_threshold=None,
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
    if initial_transform is not None and np.all(default == np.eye(4)):
        default = initial_transform

    # check jaccard index
    a = jaccard_filter_threshold is not None
    b = fix_mask is not None
    c = mov_mask is not None
    if a and b and c:
        if not jaccard_filter(fix_mask, mov_mask, jaccard_filter_threshold):
            print("Masks failed jaccard_filter")
            print("Returning default")
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

    # convert to float32 sitk images
    fix = ut.numpy_to_sitk(fix, fix_spacing, origin=fix_origin)
    mov = ut.numpy_to_sitk(mov, mov_spacing, origin=mov_origin)
    fix = sitk.Cast(fix, sitk.sitkFloat32)
    mov = sitk.Cast(mov, sitk.sitkFloat32)

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

    # execute alignment, for any exceptions return default
    try:
        initial_metric_value = irm.MetricEvaluate(fix, mov)
        irm.Execute(fix, mov)
        final_metric_value = irm.MetricEvaluate(fix, mov)
    except Exception as e:
        print("Registration failed due to ITK exception:\n", e)
        print("\nReturning default")
        sys.stdout.flush()
        return default

    # if centered, convert back to Euler3DTransform object
    if rigid and initialize_with_centering:
        transform = sitk.Euler3DTransform(transform)

    # if registration improved metric return result
    # otherwise return default
    if final_metric_value < initial_metric_value:
        sys.stdout.flush()
        return ut.affine_transform_to_matrix(transform)
    else:
        print("Optimization failed to improve metric")
        print("initial value: {}".format(initial_metric_value))
        print("final value: {}".format(final_metric_value))
        print("Returning default")
        sys.stdout.flush()
        return default


def deformable_align(
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
    jaccard_filter_threshold=None,
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

    # check jaccard index
    a = jaccard_filter_threshold is not None
    b = fix_mask is not None
    c = mov_mask is not None
    failed_jaccard = False
    if a and b and c:
        if not jaccard_filter(fix_mask, mov_mask, jaccard_filter_threshold):
            print("Masks failed jaccard_filter")
            print("Returning default")
            failed_jaccard = True

    # store initial fixed image shape
    initial_fix_shape = fix.shape

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

    # convert to sitk images, float32 type
    fix = ut.numpy_to_sitk(fix, fix_spacing, origin=fix_origin)
    mov = ut.numpy_to_sitk(mov, mov_spacing, origin=mov_origin)
    fix = sitk.Cast(fix, sitk.sitkFloat32)
    mov = sitk.Cast(mov, sitk.sitkFloat32)

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
        default_field = ut.bspline_to_displacement_field(
            fix, transform, shape=initial_fix_shape,
        )
        default = (default_params, default_field)

    # now that default is defined, determine jaccard result
    if failed_jaccard: return default

    # set masks
    if fix_mask is not None:
        fix_mask = ut.numpy_to_sitk(fix_mask, fix_spacing, origin=fix_origin)
        irm.SetMetricFixedMask(fix_mask)
    if mov_mask is not None:
        mov_mask = ut.numpy_to_sitk(mov_mask, mov_spacing, origin=mov_origin)
        irm.SetMetricMovingMask(mov_mask)

    # execute alignment, for any exceptions return default
    try:
        initial_metric_value = irm.MetricEvaluate(fix, mov)
        irm.Execute(fix, mov)
        final_metric_value = irm.MetricEvaluate(fix, mov)
    except Exception as e:
        print("Registration failed due to ITK exception:\n", e)
        print("\nReturning default")
        sys.stdout.flush()
        return default

    # if registration improved metric return result
    # otherwise return default
    if final_metric_value < initial_metric_value:
        sys.stdout.flush()
        fp = transform.GetFixedParameters()
        pp = transform.GetParameters()
        params = np.array(list(fp) + list(pp))
        field = ut.bspline_to_displacement_field(
            fix, transform, shape=initial_fix_shape,
        )
        return params, field
    else:
        print("Optimization failed to improve metric")
        print("initial value: {}".format(initial_metric_value))
        print("final value: {}".format(final_metric_value))
        print("Returning default")
        sys.stdout.flush()
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

    # wait for at least one worker to be fully  instantiated
    while ((cluster.client.status == "running") and
           (len(cluster.client.scheduler_info()["workers"]) < 1)):
        time.sleep(1.0)

    # compute block size and overlaps
    blocksize = np.array(fix.shape).astype(np.float32) / nblocks
    blocksize = np.ceil(blocksize).astype(np.int16)
    overlaps = tuple(np.round(blocksize * overlap).astype(np.int16))

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
        fm_da = da.from_array([None,], chunks=(1,))

    # mov mask
    if mov_mask is not None:
        mm_future = cluster.client.scatter(mm_p)
        mm_da = da.from_delayed(
            mm_future, shape=mm_p.shape, dtype=mm_p.dtype
        ).rechunk(tuple(blocksize))
    else:
        mm_da = da.from_array([None,], chunks=(1,))

    # establish all keyword arguments
    random_kwargs = {**kwargs, **random_kwargs}
    rigid_kwargs = {**kwargs, **rigid_kwargs}
    affine_kwargs = {**kwargs, **affine_kwargs}
    deform_kwargs = {**kwargs, **deform_kwargs}

    # closure for alignment pipeline
    def align_single_block(fix, mov, fm, mm, block_info=None):
        
        # check for masks
        if fm.shape != fix.shape: fm = None
        if mm.shape != mov.shape: mm = None

        # run alignment pipeline
        transform = alignment_pipeline(
            fix, mov, fix_spacing, mov_spacing, steps,
            fix_mask=fm, mov_mask=mm,
            random_kwargs=random_kwargs,
            rigid_kwargs=rigid_kwargs,
            affine_kwargs=affine_kwargs,
            deform_kwargs=deform_kwargs,
        )

        # convert to single vector field
        if isinstance(transform, tuple):
            affine, deform = transform[0], transform[1][1]
            transform = compose_affine_and_displacement_vector_field(
                affine, deform, fix_spacing,
            )
        else:
            transform = ut.matrix_to_displacement_field(
                fix, transform, fix_spacing,
            )

        # get block index and block grid size
        block_grid = block_info[0]['num-chunks']
        block_index = block_info[0]['chunk-location']

        # create weights array
        core, pad_ones, pad_linear = [], [], []
        for i in range(3):

            # get core shape and pad sizes
            o = max(0, 2*overlaps[i]-1)
            c = blocksize[i] - o + 1
            p_ones, p_linear = [0, 0], [o, o]
            if block_index[i] == 0:
                p_ones[0], p_linear[0] = o//2, 0
            if block_index[i] == block_grid[i] - 1:
                p_ones[1], p_linear[1] = o//2, 0
            core.append(c)
            pad_ones.append(tuple(p_ones))
            pad_linear.append(tuple(p_linear))

        # create weights core
        weights = np.ones(core, dtype=np.float32)

        # extend weights
        weights = np.pad(
            weights, pad_ones, mode='constant', constant_values=1,
        )
        weights = np.pad(
            weights, pad_linear, mode='linear_ramp', end_values=0,
        )

        # apply weights
        transform = transform * weights[..., None]

        # package and return result
        result = np.empty((1,1,1), dtype=object)
        result[0, 0, 0] = transform
        return result
    # END CLOSURE

    # determine overlaps list
    overlaps_list = [overlaps, overlaps]
    if fix_mask is not None:
        overlaps_list.append(overlaps)
    else:
        overlaps_list.append(0)
    if mov_mask is not None:
        overlaps_list.append(overlaps)
    else:
        overlaps_list.append(0)

    # align all chunks
    fields = da.map_overlap(
        align_single_block,
        fix_da, mov_da, fm_da, mm_da,
        depth=overlaps_list,
        dtype=object,
        boundary='none',
        trim=False,
        align_arrays=False,
        chunks=[1,1,1],
    ).compute()

    # reconstruct single transform
    transform = np.zeros(fix_p.shape + (3,), dtype=np.float32)

    # adds fields to transform
    block_grid = fields.shape[:3]
    for ijk in range(np.prod(block_grid)):
        i, j, k = np.unravel_index(ijk, block_grid)
        x, y, z = (max(0, x*y-z) for x, y, z in zip((i, j, k), blocksize, overlaps))
        xr, yr, zr = fields[i, j, k].shape[:3]
        transform[x:x+xr, y:y+yr, z:z+zr] += fields[i, j, k][...]

    # crop back to original shape and return
    x, y, z = fix.shape
    return transform[:x, :y, :z]


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
            deform = compose_displacement_vector_fields(
                deform, ddd, fix_spacing,
            )
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
    

