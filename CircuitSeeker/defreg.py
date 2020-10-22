import numpy as np
import os

import CircuitSeeker.fileio as csio
import CircuitSeeker.distributed as csd
import dask.array as da

import greedypy.greedypy_registration_method as grm

import SimpleITK as sitk

# TODO: TEMP
import sys
from numpy.random import normal


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
    fixed_orig=None, moving_orig=None,
    ):
    """
    """

    fixed = sitk.GetImageFromArray(fixed.copy().astype(np.float32))
    moving = sitk.GetImageFromArray(moving.copy().astype(np.float32))
    fixed.SetSpacing(fixed_vox[::-1])
    moving.SetSpacing(moving_vox[::-1])

    if fixed_orig is None: fixed_orig = np.zeros(len(fixed_vox))
    if moving_orig is None: moving_orig = np.zeros(len(moving_vox))
    fixed.SetOrigin(fixed_orig[::-1])
    moving.SetOrigin(moving_orig[::-1])

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


def fieldToDisplacementFieldTransform(field, spacing):
    """
    """

    field = field.astype(np.float64)
    transform = sitk.GetImageFromArray(field, isVector=True)
    transform.SetSpacing(spacing[::-1])
    return sitk.DisplacementFieldTransform(transform)


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
    number_of_histogram_bins,
    metric_sampling_percentage,
    shrink_factors,
    smooth_sigmas,
    ):
    """
    """

    # set up registration object
    ncores = int(os.environ["LSB_DJOB_NUMPROC"])  # LSF specific!
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2*ncores)
    irm = sitk.ImageRegistrationMethod()
    irm.SetNumberOfThreads(2*ncores)
    irm.SetInterpolator(sitk.sitkLinear)

    # metric
    irm.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=number_of_histogram_bins,
    )
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
    transform = irm.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )
    etransform = sitk.Euler3DTransform()
    etransform.SetParameters(transform.GetParameters())
    return affineTransformToMatrix(etransform)


# TODO: default sampling percentage should be 0.25 - changed for piecewise testing
def affineAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    rigid_matrix=None,
    number_of_histogram_bins=128,
    metric_sampling_percentage=1.0,
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


def distributedPiecewiseAffineAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    block_size=[128,128,128],
    overlap=64,
    ):
    """
    """

    # set up cluster
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
            [da.from_array(moving[i:i+bs[0], j:j+bs[1], k:k+bs[2]])
            for k in range(0, moving.shape[2], bs[2])]
            for j in range(0, moving.shape[1], bs[1])]
            for i in range(0, moving.shape[0], bs[0])]
        moving_da = da.block(moving_blocks)

        # affine align all chunks
        # TODO: need way to get registration parameters as input to this function
        piecewise_affine_aligned = da.map_overlap(
            lambda w,x,y,z: affineAlign(w, x, y, z, target_spacing=None, rigid_matrix=np.eye(4)),
            fixed_da, moving_da,
            depth=overlap,
            dtype=np.float32,
            boundary='reflect',
            y=fixed_vox, z=fixed_vox,
        ).compute()

        return piecewise_affine_aligned

    

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
    fixed, moving = numpyToSITK(
        fixed, moving,
        fixed_vox, moving_vox,
        fixed_orig, moving_orig,
    )

    # default transform spacing is fixed voxel spacing 
    if transform_spacing is None:
        transform_spacing = fixed_vox

    # construct transform
    transform = sitk.Transform()
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

