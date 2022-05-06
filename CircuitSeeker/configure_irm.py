import os, psutil
import SimpleITK as sitk


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

