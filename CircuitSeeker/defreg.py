import numpy as np
import os

import CircuitSeeker.fileio as csio
import CircuitSeeker.distributed as csd
import dask.array as da

import SimpleITK as sitk

# TODO: TEMP
import sys


def skipSample(
    fixed, moving,
    fixed_vox, moving_vox,
    target_spacing
    ):
    """
    """

    # determine skip sample factors
    fss = np.maximum(np.round(target_spacing / fixed_vox), 1).astype(np.int)
    mss = np.maximum(np.round(target_spacing / moving_vox), 1).astype(np.int)

    # skip sample the images
    fixed = fixed[::fss[0], ::fss[1], ::fss[2]]
    moving = moving[::mss[0], ::mss[1], ::mss[2]]
    fixed_vox = fixed_vox * fss
    moving_vox = moving_vox * mss
    return fixed, moving, fixed_vox, moving_vox


def numpyToSITK(
    fixed, moving,
    fixed_vox, moving_vox,
    early_return=False,
    ):
    """
    """

    fixed = sitk.GetImageFromArray(fixed.copy().astype(np.float32))
    moving = sitk.GetImageFromArray(moving.copy().astype(np.float32))
    fixed.SetSpacing(fixed_vox[::-1])
    moving.SetSpacing(moving_vox[::-1])
    return fixed, moving


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


def getLinearRegistrationModel(
    fixed_vox,
    learning_rate,
    iterations,
    number_of_histogram_bins,
    metric_sampling_percentage,
    shrink_factors,
    smooth_sigmas,
    ):
    """
    """

    # set up registration object
    irm = sitk.ImageRegistrationMethod()
    ncores = int(os.environ["LSB_DJOB_NUMPROC"])  # LSF specific!
    irm.SetNumberOfThreads(2*ncores)
    irm.SetInterpolator(sitk.sitkLinear)

    # metric
    irm.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=number_of_histogram_bins,
    )
    irm.SetMetricSamplingStrategy(irm.RANDOM)
    irm.SetMetricSamplingPercentage(metric_sampling_percentage)

    # optimizer
    max_step = np.min(fixed_vox)
    irm.SetOptimizerAsGradientDescent(
        numberOfIterations=iterations,
        learningRate=learning_rate,
        maximumStepSizeInPhysicalUnits=max_step,
    )
    irm.SetOptimizerScalesFromPhysicalShift()

    # pyramid
    irm.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    irm.SetSmoothingSigmasPerLevel(smoothingSigmas=smooth_sigmas)
    irm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # callback
    irm.AddCommand(
        sitk.sitkIterationEvent, lambda: print("METRIC: ", irm.GetMetricValue())
    )
    return irm


def getDeformableRegistrationModel(
    fixed_vox,
    learning_rate,
    iterations,
    shrink_factors,
    smooth_sigmas,
    ncc_radius,
    ):
    """
    """

    # set up registration object
    irm = sitk.ImageRegistrationMethod()
    ncores = int(os.environ["LSB_DJOB_NUMPROC"])  # LSF specific!
    irm.SetNumberOfThreads(2*ncores)
    irm.SetInterpolator(sitk.sitkLinear)

    # metric
    irm.SetMetricAsANTSNeighborhoodCorrelation(ncc_radius)
    irm.MetricUseFixedImageGradientFilterOff()

    # optimizer
    max_step = np.min(fixed_vox)
    irm.SetOptimizerAsGradientDescent(
        numberOfIterations=iterations,
        learningRate=learning_rate,
        maximumStepSizeInPhysicalUnits=max_step,
    )
    irm.SetOptimizerScalesFromPhysicalShift()
    
    # pyramid
    irm.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    irm.SetSmoothingSigmasPerLevel(smoothingSigmas=smooth_sigmas)
    irm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # callback
    irm.AddCommand(
        sitk.sitkIterationEvent, lambda: print("METRIC: ", irm.GetMetricValue())
    )
    return irm


def rigidAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    number_of_histogram_bins=128,
    metric_sampling_percentage=0.25,
    shrink_factors=[2,1],
    smooth_sigmas=[1,0],
    learning_rate=1.0,
    number_of_iterations=250,
    target_spacing=2.0):
    """
    """

    # skip sample
    if target_spacing is not None:
        fixed, moving, fixed_vox, moving_vox = skipSample(
            fixed, moving, fixed_vox, moving_vox, target_spacing
        )

    # convert to sitk images, set spacing
    fixed, moving = numpyToSITK(fixed, moving, fixed_vox, moving_vox)

    # set up registration object, initialize
    irm = getLinearRegistrationModel(
        fixed_vox,
        learning_rate,
        number_of_iterations,
        number_of_histogram_bins,
        metric_sampling_percentage,
        shrink_factors,
        smooth_sigmas,
    )
    irm.SetInitialTransform(sitk.Euler3DTransform())

    # execute, return as ndarray
    transform = irm.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                            sitk.Cast(moving, sitk.sitkFloat32),
    )
    etransform = sitk.Euler3DTransform()
    etransform.SetParameters(transform.GetParameters())
    return affineTransformToMatrix(etransform)


def affineAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    rigid_matrix=None,
    number_of_histogram_bins=128,
    metric_sampling_percentage=0.25,
    shrink_factors=[2,1],
    smooth_sigmas=[1,0],
    learning_rate=1.0,
    number_of_iterations=250,
    target_spacing=2.0):
    """
    """

    # skip sample
    if target_spacing is not None:
        fixed, moving, fixed_vox, moving_vox = skipSample(
            fixed, moving, fixed_vox, moving_vox, target_spacing
        )

    # convert to sitk images, set spacing
    fixed, moving = numpyToSITK(fixed, moving, fixed_vox, moving_vox)
    rigid = matrixToAffineTransform(rigid_matrix)

    # set up registration object
    irm = getLinearRegistrationModel(
        fixed_vox,
        learning_rate,
        number_of_iterations,
        number_of_histogram_bins,
        metric_sampling_percentage,
        shrink_factors,
        smooth_sigmas,
    )

    # initialize
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(rigid.GetMatrix())
    affine.SetTranslation(rigid.GetTranslation())
    affine.SetCenter(rigid.GetCenter())
    irm.SetInitialTransform(affine)

    # execute, return as ndarray
    transform = irm.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                            sitk.Cast(moving, sitk.sitkFloat32),
    )
    atransform = sitk.AffineTransform(3)
    atransform.SetParameters(transform.GetParameters())
    return affineTransformToMatrix(atransform)


def deformableAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    affine_matrix,
    ncc_radius=8,
    gradient_smoothing=0.5,
    field_smoothing=1.5,
    shrink_factors=[2,1],
    smooth_sigmas=[1,0],
    learning_rate=1.0,
    number_of_iterations=5,
    ):
    """
    """

    fixed_copy = np.copy(fixed)
    moving_copy = np.copy(moving)

    # convert to sitk images, set spacing
    fixed, moving = numpyToSITK(fixed, moving, fixed_vox, moving_vox, early_return=True)
    affine = matrixToAffineTransform(affine_matrix)

    # set up registration object
    irm = getDeformableRegistrationModel(
        fixed_vox,
        learning_rate,
        number_of_iterations,
        shrink_factors,
        smooth_sigmas,
        ncc_radius,
    )

    # initialize
    tdff = sitk.TransformToDisplacementFieldFilter()
    tdff.SetReferenceImage(fixed)
    df = tdff.Execute(affine)
    dft = sitk.DisplacementFieldTransform(df)
    dft.SetSmoothingGaussianOnUpdate(
        varianceForUpdateField=gradient_smoothing,
        varianceForTotalField=field_smoothing,
    )
    irm.SetInitialTransform(dft, inPlace=True)

    # execute
    deformation = irm.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                              sitk.Cast(moving, sitk.sitkFloat32),
   )

    # convert to displacement vector field and return as ndarray
    tdff.SetOutputPixelType(sitk.sitkVectorFloat32)
    return sitk.GetArrayFromImage(tdff.Execute(deformation))


def distributedDeformableAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    affine_matrix,
    block_size=[256,256,256],
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

    # set up the distributed environment
    ds = distributed_state
    if distributed_state is None:
        ds = csd.distributedState()
        # TODO: expose cores/tpw, remove job_extra -P
        ds.initializeLSFCluster(
            job_extra=["-P scicompsoft"],
            ncpus=4,
            cores=4,
            threads_per_worker=8,
            memory="60GB",
            mem=60000,
        )
        ds.initializeClient()
        nchunks = np.ceil(np.array(fixed.shape)/block_size)
        ds.scaleCluster(njobs=np.prod(nchunks))

    # TODO: refactor into a function, generalize w.r.t. dimension, share on github
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
    # TODO: need to tell map_overlap about new return dimension
    deformation = da.map_overlap(
        lambda v,w,x,y,z: deformableAlign(v, w, x, y, z),
        fixed_da, moving_da, depth=overlap,
        dtype=fixed.dtype, chunks=block_size+[3,], new_axis=[3,],
        x=fixed_vox, y=fixed_vox, z=affine_matrix,
    ).compute()

    # TODO: consider smoothing seams (smooth whole image, replace seams)

    # release resources
    if distributed_state is None:
        ds.closeClient()

    # TODO: TEMP
#    resampled = applyTransformToImage(
#        fixed, moving_res, fixed_vox, fixed_vox, displacement=deformation
#    )

    resampled = 1
    return deformation, resampled


def applyTransformToImage(
    fixed, moving,
    fixed_vox, moving_vox,
    matrix=None, displacement=None,
    ):
    """
    """

    # need matrix or displacement
    error = "affine matrix or diplacement field required, but not both"
    assert( (matrix is not None) != (displacement is not None) ), error

    # convert to sitk objects
    dtype = fixed.dtype
    fixed, moving = numpyToSITK(fixed, moving, fixed_vox, moving_vox)
    if matrix is not None:
        transform = matrixToAffineTransform(matrix)
    elif displacement is not None:
        displacement = displacement.astype(np.float64)
        transform = sitk.GetImageFromArray(displacement, isVector=True)
        transform.SetSpacing(fixed_vox)
        transform = sitk.DisplacementFieldTransform(transform)

    # set up resampler object
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.Cast(fixed, sitk.sitkFloat32))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    # execute, return as numpy array
    resampled = resampler.Execute(sitk.Cast(moving, sitk.sitkFloat32))
    return sitk.GetArrayFromImage(resampled).astype(dtype)

