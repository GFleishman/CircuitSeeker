import numpy as np
import os
import ClusterWrap
import CircuitSeeker.fileio as csio
import CircuitSeeker.utility as ut
from CircuitSeeker.align import affine_align
from CircuitSeeker.transform import apply_transform
import dask.array as da
import dask.delayed as delayed
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom
import zarr
from numcodecs import Blosc
from pathlib import Path
from glob import glob


def distributed_image_mean(
    frames,
    cluster_kwargs={},
):
    """
    """

    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # hdf5 files use dask.array
        if csio.testPathExtensionForHDF5(frames['suffix']):
            frames = csio.daskArrayBackedByHDF5(
                frames['folder'], frames['prefix'],
                frames['suffix'], frames['dataset_path'],
            )
            frames_mean = frames.mean(axis=0, dtype=np.float32).compute()
            frames_mean = np.round(frames_mean).astype(frames[0].dtype)
        # other types use dask.bag
        else:
            frames = csio.daskBagOfFilePaths(
                frames['folder'], frames['prefix'], frames['suffix'],
            )
            nframes = frames.npartitions
            frames_mean = frames.map(csio.readImage).reduction(sum, sum).compute()
            dtype = frames_mean.dtype
            frames_mean = np.round(frames_mean/np.float(nframes)).astype(dtype)

    # return reference to mean image
    return frames_mean


def motion_correct(
    fix, frames,
    fix_spacing, frames_spacing,
    time_stride=1,
    sigma=7,
    cluster_kwargs={},
    **kwargs,
):
    """
    """

    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # wrap fixed data as delayed object
        fix_d = delayed(fix)

        # get total number of frames
        total_frames = len(csio.globPaths(
            frames['folder'], frames['prefix'], frames['suffix'],
        ))

        # create dask array of all frames
        frames_data = csio.daskArrayBackedByHDF5(
            frames['folder'], frames['prefix'],
            frames['suffix'], frames['dataset_path'],
            stride=time_stride,
        )
        compute_frames = frames_data.shape[0]

        # set alignment defaults
        alignment_defaults = {
            'rigid':True,
            'alignment_spacing':2.0,
            'metric':'MS',
            'sampling':'random',
            'sampling_percentage':0.1,
            'optimizer':'RGD',
            'iterations':50,
            'max_step':2.0,
        }
        for k, v in alignment_defaults.items():
            if k not in kwargs:
                kwargs[k] = v
 
        # wrap align function
        def wrapped_affine_align(mov, fix_d):
            mov = mov.squeeze()
            t = affine_align(
                fix_d, mov, fix_spacing, frames_spacing,
                **kwargs,
            )
            e = ut.matrix_to_euler_transform(t)
            p = ut.euler_transform_to_parameters(e)
            return p[None, :]

        params = da.map_blocks(
            wrapped_affine_align, frames_data,
            fix_d=fix_d,
            dtype=np.float64,
            drop_axis=[2, 3,],
            chunks=[1, 6],
        ).compute()

    # (weak) outlier removal and smoothing
    params = median_filter(params, footprint=np.ones((3,1)))
    params = gaussian_filter1d(params, sigma, axis=0)

    # interpolate
    if time_stride > 1:
        x = np.linspace(0, compute_frames-1, total_frames)
        coords = np.meshgrid(x, np.mgrid[:6], indexing='ij')
        params = map_coordinates(params, coords)

    # convert to matrices
    transforms = np.empty((total_frames, 4, 4))
    for i in range(params.shape[0]):
        e = ut.parameters_to_euler_transform(params[i])
        t = ut.affine_transform_to_matrix(e)
        transforms[i] = t

    # return all transforms
    return transforms


def save_transforms(transforms, folder, prefix='rigid_transform'):
    """
    """

    Path(folder).mkdir(parents=True, exist_ok=True)
    for i in range(transforms.shape[0]):
        basename = prefix + f'_frame_{i:08d}.mat'
        path = os.path.join(folder, basename)
        np.savetxt(path, transforms[i])


def read_transforms(folder, prefix='rigid_transform'):
    """
    """

    tp = sorted(glob(folder + '/' + prefix + "*"))
    transforms = np.empty( (len(tp),) + (4,4) )
    for i in range(transforms.shape[0]):
        transforms[i] = np.loadtxt(tp[i])
    return transforms


def resample_frames(
    frames,
    frames_spacing,
    transforms,
    write_path,
    mask=None,
    time_stride=1,
    compression_level=4,
    cluster_kwargs={},
):
    """
    """

    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # create dask array of all frames
        frames_data = csio.daskArrayBackedByHDF5(
            frames['folder'], frames['prefix'],
            frames['suffix'], frames['dataset_path'],
            stride=time_stride,
        )
        total_frames = frames_data.shape[0]

        # wrap transforms as dask array
        # extra dimension to match frames_data ndims
        transforms = transforms[::time_stride, None, :, :]
        transforms_d = da.from_array(transforms, chunks=(1, 1, 4, 4))

        # wrap mask
        mask_d = None
        if mask is not None:
            mask_sh, frame_sh = mask.shape, frames_data.shape[1:]
            if mask_sh != frame_sh:
                mask = zoom(mask, np.array(frame_sh) / mask_sh, order=0)
            mask_d = delayed(mask)

        # wrap transform function
        def wrapped_apply_transform(mov, t, mask_d=None):
            mov = mov.squeeze()
            t = t.squeeze()
            aligned = apply_transform(
                mov, mov, frames_spacing, frames_spacing,
                transform_list=[t,],
            )
            if mask_d is not None:
                aligned = aligned * mask_d
            return aligned[None, ...]

        # apply transform to all frames
        frames_aligned = da.map_blocks(
            wrapped_apply_transform, frames_data, transforms_d,
            mask_d=mask_d,
            dtype=np.uint16,
            chunks=[1,] + list(frames_data.shape[1:]),
        )

        # write in parallel as 4D array to zarr file
        compressor = Blosc(
            cname='zstd',
            clevel=compression_level,
            shuffle=Blosc.BITSHUFFLE,
        )
        aligned_disk = zarr.open(
            write_path, 'w',
            shape=frames_aligned.shape,
            chunks=[1,] + list(frames_data.shape[1:]),
            dtype=frames_aligned.dtype,
            compressor=compressor
        )
        da.to_zarr(frames_aligned, aligned_disk)

        # return reference to zarr store
        return aligned_disk

