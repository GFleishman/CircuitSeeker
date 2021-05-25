import numpy as np
import os
import ClusterWrap
import CircuitSeeker.fileio as csio
import CircuitSeeker.utility as ut
from CircuitSeeker.align import affine_align
from CircuitSeeker.align import bspline_deformable_align
from CircuitSeeker.transform import apply_transform
import dask.array as da
import dask.delayed as delayed
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom
import zarr
from numcodecs import Blosc
from glob import glob
import json


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


def deformable_motion_correct(
    fix, frames,
    fix_spacing, frames_spacing,
    time_stride=1,
    sigma=7,
    fix_mask=None,
    affine_kwargs={},
    bspline_kwargs={},
    cluster_kwargs={},
):
    """
    """

    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # wrap fixed data as delayed object
        fix_d = delayed(fix)

        # wrap fixx mask if given
        fix_mask_d = delayed(np.ones(fix.shape, dtype=np.uint8))
        if fix_mask is not None:
            fix_mask_d = delayed(fix_mask)

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

        # affine defaults
        affine_defaults = {
            'alignment_spacing':2.0,
            'metric':'MS',
            'sampling':'random',
            'sampling_percentage':0.1,
            'optimizer':'RGD',
            'iterations':50,
            'max_step':2.0,
        }
        for k, v in affine_defaults.items():
            if k not in affine_kwargs: affine_kwargs[k] = v

        # bspline defaults
        bspline_defaults = {
            'alignment_spacing':1.0,
            'metric':'MI',
            'sampling':'random',
            'sampling_percentage':0.01,
            'iterations':250,
            'shrink_factors':[2,],
            'smooth_sigmas':[2,],
            'max_step':1.0,
            'control_point_spacing':100.,
            'control_point_levels':[1,],
        }
        for k, v in bspline_defaults.items():
            if k not in bspline_kwargs: bspline_kwargs[k] = v
 
        # wrap align function
        def wrapped_bspline_align(mov, fix_d, fix_mask_d):
            mov = mov.squeeze()
            a = affine_align(
                fix_d, mov, fix_spacing, frames_spacing,
                **affine_kwargs,
            )
            b = bspline_deformable_align(
                fix_d, mov, fix_spacing, frames_spacing,
                fix_mask=fix_mask_d,
                initial_transform=a,
                return_parameters=True,
                **bspline_kwargs,
            )
            return np.hstack((a.flatten(), b))[None, :]

        # total number of params
        # 16 for affine, 18 for bspline fixed params
        # need to compute number of bspline control point params
        xxx = fix.shape * fix_spacing
        y = bspline_kwargs['control_point_spacing']
        cp_grid = [max(1, int(x/y)) + 3 for x in xxx]
        n_params = 16 + 18 + np.prod(cp_grid)*3

        # execute
        params = da.map_blocks(
            wrapped_bspline_align, frames_data,
            fix_d=fix_d,
            fix_mask_d=fix_mask_d,
            dtype=np.float64,
            drop_axis=[2, 3,],
            chunks=[1, n_params],
        ).compute()

    # (weak) outlier removal and smoothing
    params = median_filter(params, footprint=np.ones((3,1)))
    params = gaussian_filter1d(params, sigma, axis=0)

    # interpolate
    if time_stride > 1:
        x = np.linspace(0, compute_frames-1, total_frames)
        coords = np.meshgrid(x, np.mgrid[:n_params], indexing='ij')
        params = map_coordinates(params, coords, order=1)

    # return all parameters
    return params


def save_transforms(path, transforms):
    """
    """

    n = transforms.shape[0]
    d = {i:transforms[i].tolist() for i in range(n)}
    with open(path, 'w') as f:
        json.dump(d, f, indent=4)


def read_transforms(path):
    """
    """

    with open(path, 'r') as f:
        d = json.load(f)
    return np.array([d[str(i)] for i in range(len(d))])


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
        if len(transforms.shape) == 3:
            transforms = transforms[::time_stride, None, :, :]
        elif len(transforms.shape) == 2:
            transforms = transforms[::time_stride, None, None, :]
        transforms_d = da.from_array(transforms, chunks=(1,)+transforms[0].shape)

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

            # just an affine matrix
            transform_list = [t,]

            # affine plus bspline
            if len(t.shape) == 1:
                transform_list = [t[:16].reshape((4,4)), t[16:]]

            # apply transform(s)
            aligned = apply_transform(
                mov, mov, frames_spacing, frames_spacing,
                transform_list=transform_list,
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


