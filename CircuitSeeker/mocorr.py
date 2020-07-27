from glob import glob
import numpy as np
import os

import CircuitSeeker.fileio as csio
import CircuitSeeker.distributed as csd
import dask.array as da
import dask.bag as db
import dask.delayed as delayed

import SimpleITK as sitk
from scipy.ndimage import percentile_filter, gaussian_filter1d
import zarr
from numcodecs import Blosc


def ensureArray(reference, dataset_path):
    """
    """

    if not isinstance(reference, np.ndarray):
        if not isinstance(reference, str):
            raise ValueError("image references must be ndarrays or filepaths")
        reference = csio.readImage(reference, dataset_path)[...]  # hdf5 arrays are lazy
    return reference


def rigidAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    dataset_path=None,
    metric_sample_percentage=0.1,
    shrink_factors=[2,1],
    smooth_sigmas=[1,0],
    minStep=0.1,
    learningRate=1.0,
    numberOfIterations=50,
    target_spacing=2.0,
):
    """
    Returns rigid transform parameters aligning `fixed` coords to `moving` coords
    `fixed` and `moving` must be numpy arrays or file paths
    `fixed_vox` and `moving_vox` must be fixed and moving image voxel spacings as numpy arrays
    if `fixed` and/or `moving` are hdf5 filepaths, you must specify `dataset_path`
    remaining arguments adjust the rigid registration algorithm

    Images are skip sub-sampled before registration. The skip stride is determined by
    `target_spacing` which is the target voxel spacing after skip sub-sampling.
    Images are never up-sampled so axes with spacing greater than `target_spacing` are
    not skip sub-sampled.
    """

    # get moving/fixed images as ndarrays
    fixed = ensureArray(fixed, dataset_path)
    moving = ensureArray(moving, dataset_path)

    # determine skip sample factors
    fss = np.maximum(np.round(target_spacing / fixed_vox), 1).astype(np.int)
    mss = np.maximum(np.round(target_spacing / moving_vox), 1).astype(np.int)

    # skip sample the images
    fixed = fixed[::fss[0], ::fss[1], ::fss[2]]
    moving = moving[::mss[0], ::mss[1], ::mss[2]]
    fixed_vox = fixed_vox * fss
    moving_vox = moving_vox * mss

    # convert to sitk images, set spacing
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)
    fixed.SetSpacing(fixed_vox[::-1])  # numpy z,y,x --> itk x,y,z
    moving.SetSpacing(moving_vox[::-1])

    # set up registration object
    irm = sitk.ImageRegistrationMethod()
    ncores = int(os.environ["LSB_DJOB_NUMPROC"])  # LSF specific!
    irm.SetNumberOfThreads(2*ncores)
    irm.SetInterpolator(sitk.sitkLinear)

    # metric, built for speed
    irm.SetMetricAsMeanSquares()
    irm.SetMetricSamplingStrategy(irm.RANDOM)
    irm.SetMetricSamplingPercentage(metric_sample_percentage)

    # optimizer, built for simplicity
    max_step = np.min(fixed_vox)
    irm.SetOptimizerAsRegularStepGradientDescent(
        minStep=minStep, learningRate=learningRate,
        numberOfIterations=numberOfIterations,
        maximumStepSizeInPhysicalUnits=max_step
    )
    irm.SetOptimizerScalesFromPhysicalShift()

    # pyramid
    irm.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    irm.SetSmoothingSigmasPerLevel(smoothingSigmas=smooth_sigmas)
    irm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # initialize
    irm.SetInitialTransform(sitk.Euler3DTransform())

    # execute, convert to numpy and return
    transform = irm.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                            sitk.Cast(moving, sitk.sitkFloat32),
    )
    return transform.GetParameters()


def applyTransform(
    moving,
    moving_vox,
    params,
    dataset_path=None):
    """
    """

    # get the moving image as a numpy array
    moving = ensureArray(moving, dataset_path)

    # use sitk transform and interpolation to apply transform
    moving = sitk.GetImageFromArray(moving)
    moving.SetSpacing(moving_vox[::-1])  # numpy z,y,x --> itk x,y,z
    transform = _parametersToEuler3DTransform(params)
    transformed = sitk.Resample(moving, moving, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    # return as numpy array
    return sitk.GetArrayFromImage(transformed)


# useful format conversions for rigid transforms
def _euler3DTransformToParameters(euler):
    """
    """
    return np.array(( euler.GetAngleX(),
                      euler.GetAngleY(),
                      euler.GetAngleZ() ) +
                      euler.GetTranslation()
                   )
def _parametersToEuler3DTransform(params):
    """
    """
    transform = sitk.Euler3DTransform()
    transform.SetRotation(*params[:3])
    transform.SetTranslation(params[3:])
    return transform
def _parametersToRigidMatrix(params):
    """
    """
    transform = _parametersToEuler3DTransform(params)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape((3,3))
    matrix[:3, -1] = np.array(transform.GetTranslation())
    return matrix


# TODO: refactor motionCorrect
def motionCorrect(
    folder, prefix, suffix,
    fixed, fixed_vox, moving_vox,
    write_path, dataset_path=None,
    distributed_state=None, sigma=7,
    transforms_dir=None
):
    """
    """

    # set up the distributed environment
    ds = distributed_state
    if distributed_state is None:
        ds = csd.distributedState()
        ds.initializeLSFCluster(job_extra=["-P scicompsoft"])
        ds.initializeClient()

    # create (lazy) dask bag from all frames
    frames = csio.daskBagOfFilePaths(folder, prefix, suffix)
    nframes = frames.npartitions
    ds.scaleCluster(njobs=nframes)

    # align all
    dfixed = delayed(fixed)
    dfixed_vox = delayed(fixed_vox)
    dmoving_vox = delayed(moving_vox)
    ddataset_path = delayed(dataset_path)
    params = frames.map(lambda b,w,x,y,z: rigidAlign(w,b,x,y, dataset_path=z),
        w=dfixed, x=dfixed_vox, y=dmoving_vox, z=ddataset_path,
    ).compute()
    params = np.array(list(params))

    # (weak) outlier removal and smoothing
    params = percentile_filter(params, 50, footprint=np.ones((3,1)))
    params = gaussian_filter1d(params, sigma, axis=0)

    # write transforms as matrices
    if transforms_dir is not None:
        paths = list(frames)
        for ind, p in enumerate(params):
            transform = _parametersToRigidMatrix(p)
            basename = os.path.splitext(os.path.basename(paths[ind]))[0]
            path = os.path.join(transforms_dir, basename) + '_rigid.mat'
            np.savetxt(path, transform)

    # apply transforms to all images
    params = db.from_sequence(params, npartitions=nframes)
    transformed = frames.map(lambda b,x,y,z: applyTransform(b,x,y, dataset_path=z),
        x=dmoving_vox, y=params, z=ddataset_path,
    ).to_delayed()

    # convert to a (lazy) 4D dask array
    sh = transformed[0][0].shape.compute()
    dd = transformed[0][0].dtype.compute()
    arrays = [da.from_delayed(t[0], sh, dtype=dd) for t in transformed]
    transformed = da.stack(arrays, axis=0)

    # write in parallel as 4D array to zarr file
    compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
    transformed_disk = zarr.open(write_path, 'w',
        shape=transformed.shape, chunks=(256, 10, 256, 256),
        dtype=transformed.dtype, compressor=compressor
    )
    da.to_zarr(transformed, transformed_disk)

    # release resources
    if distributed_state is None:
        ds.closeClient()

    # return reference to data on disk
    return transformed_disk


def distributedImageMean(
    folder, prefix, suffix, dataset_path=None,
    distributed_state=None, write_path=None,
):
    """
    Returns mean over images matching `folder/prefix*suffix`
    If images are hdf5 you must specify `dataset_path`
    To additionally write the mean image to disk, specify `write_path`
    Computations are distributed, to supply your own dask scheduler and cluster set
        `distributed_state` to an existing `CircuitSeeker.distribued.distributedState` object
        otherwise a new cluster will be created
    """

    # set up the distributed environment
    ds = distributed_state
    if distributed_state is None:
        ds = csd.distributedState()
        ds.initializeLSFCluster(job_extra=["-P scicompsoft"])
        ds.initializeClient()

    # hdf5 files use dask.array
    if csio.testPathExtensionForHDF5(suffix):
        frames = csio.daskArrayBackedByHDF5(folder, prefix, suffix, dataset_path)
        nframes = frames.shape[0]
        ds.scaleCluster(njobs=nframes)
        frames_mean = frames.mean(axis=0).compute()
        frames_mean = np.round(frames_mean).astype(frames[0].dtype)
    # other types use dask.bag
    else:
        frames = csio.daskBagOfFilePaths(folder, prefix, suffix)
        nframes = frames.npartitions
        ds.scaleCluster(njobs=nframes)
        frames_mean = frames.map(csio.readImage).reduction(np.sum, np.sum).compute()
        frames_mean = np.array(frames_mean.compute())
        dtype = frames_mean.dtype
        frames_mean = np.round(frames_mean/np.float(nframes)).astype(dtype)

    # release resources
    if distributed_state is None:
        ds.closeClient()

    # write result
    if write_path is not None:
        if csio.testPathExtensionForHDF5(write_path):
            csio.writeHDF5(write_path, dataset_path, frames_mean)
        else:
            csio.writeImage(write_path, frames_mean)

    # return reference to mean image
    return frames_mean

