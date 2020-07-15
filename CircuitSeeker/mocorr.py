from glob import glob
import numpy as np
import h5py
import os

import CircuitSeeker.distributed as csd
import dask.array as da
import dask.bag as db
import dask.delayed as delayed

import SimpleITK as sitk
from scipy.ndimage import percentile_filter, gaussian_filter1d
import zarr
from numcodecs import Blosc

# TODO: consider usability redesign: should probably pass arrays to rigidAlign for example
# TODO: consider moving away from bags or file paths to a full 4D array using Davis suggestions


def distributedImageMean(
    frame_dir, frame_prefix, dset_name='default',
    distributed_state=None, write_path=None,
):
    """
    Average all images in `frame_dir` beginning with `frame_prefix`
    """

    # get paths to all images
    frame_dir = os.path.abspath(frame_dir)
    frames = sorted(glob(os.path.join(frame_dir, frame_prefix) + '*.h5'))
    nframes = len(frames)

    # set up the distributed environment
    if distributed_state is None:
        ds = csd.distributedState()
        ds.initializeLSFCluster()
        ds.initializeClient()
    else:
        ds = distributed_state
    ds.scaleCluster(nframes)

    # create (lazy) dask array from all images
    dsets = [h5py.File(frame, mode='r')[dset_name] for frame in frames]
    arrays = [da.from_array(dset, chunks=(256,)*3) for dset in dsets]
    array = da.stack(arrays, axis=0)

    # take mean
    frames_mean = array.mean(axis=0).compute()
    frames_mean = np.round(frames_mean).astype(dsets[0].dtype)

    # release resources
    if distributed_state is None:
        ds.closeClient()

    # write result
    if write_path is not None:
        with h5py.File(write_path, mode='w') as f:
            dset = f.create_dataset('default', frames_mean.shape, frames_mean.dtype)
            dset[...] = frames_mean

    return frames_mean


def rigidAlign(fixed, moving, fixed_vox, moving_vox, dset_name='default',
    metric_sample_percentage=0.1, shrink_factors=[2,1], smooth_sigmas=[1,0],
    opt_minStep=0.1, opt_learningRate=1.0, opt_numberOfIterations=50,
    fixed_ss=[2,5,5], moving_ss=[1,5,5]
):
    """
    rigid align `(ndarray) moving` to `(ndarray) fixed`; must provide
    `(ndarray) fixed_vox` and `(ndarray) moving_vox` voxel spacings
    """

    # get the fixed image as a numpy array
    if not isinstance(fixed, np.ndarray):
        if not isinstance(fixed, str):
            raise ValueError("fixed must be an ndarray or a filepath")
    fixed = h5py.File(fixed, mode='r')[dset_name]

    # get the moving image as a numpy array
    moving = h5py.File(moving, mode='r')[dset_name]

    # skip sample
    fixed = fixed[::fixed_ss[0], ::fixed_ss[1], ::fixed_ss[2]]
    moving = moving[::moving_ss[0], ::moving_ss[1], ::moving_ss[2]]
    fixed_vox = fixed_vox * np.array(fixed_ss)
    moving_vox = moving_vox * np.array(moving_ss)

    # convert to sitk images, set spacing
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)
    fixed.SetSpacing(fixed_vox[::-1])  # numpy z,y,x --> itk x,y,z
    moving.SetSpacing(moving_vox[::-1])

    # set up registration object
    irm = sitk.ImageRegistrationMethod()
    irm.SetNumberOfThreads(2)  # TEMP: should be 2*ncores, ncores determined automatically
    irm.SetInterpolator(sitk.sitkLinear)

    # metric, built for speed
    irm.SetMetricAsMeanSquares()
    irm.SetMetricSamplingStrategy(irm.RANDOM)
    irm.SetMetricSamplingPercentage(metric_sample_percentage)

    # optimizer, built for simplicity
    max_step = np.min(fixed_vox)
    irm.SetOptimizerAsRegularStepGradientDescent(
        minStep=opt_minStep, learningRate=opt_learningRate,
        numberOfIterations=opt_numberOfIterations,
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


def applyTransform(moving, moving_vox,
    params, dset_name='default'):
    """
    """

    # get the moving image as a numpy array
    moving = h5py.File(moving, mode='r')[dset_name][...]

    # use sitk transform and interpolation to apply transform
    moving = sitk.GetImageFromArray(moving)
    moving.SetSpacing(moving_vox[::-1])  # numpy z,y,x --> itk x,y,z
    transform = _parametersToEuler3DTransform(params)
    transformed = sitk.Resample(moving, moving, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    # return as numpy array
    return sitk.GetArrayFromImage(transformed)
    

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


def motionCorrect(
    frame_dir, frame_prefix,
    moving_vox, fixed, fixed_vox,
    write_path,
    distributed_state=None, sigma=7,
    transforms_dir=None
):
    """
    """

    # get paths to all images
    frame_dir = os.path.abspath(frame_dir)
    frames = sorted(glob(os.path.join(frame_dir, frame_prefix) + '*.h5'))
    nframes = len(frames)

    # TODO: frames can be corrupted/h5py can't read them
    #       even a single corrupted frame can cause the whole system (all cores) to crash
    #       consider putting in a check here for readability, throw exception if problems

    # set up the distributed environment
    if distributed_state is None:
        ds = csd.distributedState()
        ds.initializeLSFCluster()
        ds.initializeClient()
    else:
        ds = distributed_state
    ds.scaleCluster(nframes)

    # create (lazy) dask bag from all frames
    bag = db.from_sequence(frames, npartitions=nframes)

    # align all
    dfixed = delayed(fixed)
    dfixed_vox = delayed(fixed_vox)
    dmoving_vox = delayed(moving_vox)
    params = bag.map(lambda b,x,y,z: rigidAlign(x,b,y,z),
        x=dfixed, y=dfixed_vox, z=dmoving_vox,
    ).compute()
    params = np.array(list(params))

    # (weak) outlier removal and smoothing
    params = percentile_filter(params, 50, footprint=np.ones((3,1)))
    params = gaussian_filter1d(params, sigma, axis=0)

    # write transforms as matrices
    if transforms_dir is not None:
        for ind, p in enumerate(params):
            transform = _parametersToRigidMatrix(p)
            basename = os.path.splitext(os.path.basename(frames[ind]))[0]
            path = os.path.join(transforms_dir, basename) + '_rigid.mat'
            np.savetxt(path, transform)

    # apply transforms to all images
    params = db.from_sequence(params, npartitions=nframes)
    transformed = bag.map(lambda b,x,y: applyTransform(b,x,y),
        x=dmoving_vox, y=params,
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

