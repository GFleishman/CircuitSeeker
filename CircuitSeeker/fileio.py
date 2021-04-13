import SimpleITK as sitk
import h5py
from glob import glob
from os import path
import numpy as np
import ClusterWrap

import dask.bag as db
import dask.array as da


def testPathExtensionForHDF5(image_path):
    """
    Returns true if `image_path` has an hdf5 like extension
    Currently: `['.h5', '.hdf5']
    """

    hdf5_extensions = ['.h5', '.hdf5']
    if image_path in hdf5_extensions:
        return True
    else:
        ext = path.splitext(image_path)[-1]
        return ext in hdf5_extensions


def globPaths(folder, prefix, suffix):
    """
    Returns sorted list of all absolue paths matching `folder/prefix*suffix`
    If no such paths are found, throws AsserionError
    """

    folder = path.abspath(folder)
    images = sorted(glob(path.join(folder, prefix) + '*' + suffix))
    assert (len(images) > 0), f"No images in {folder} matching {prefix}*{suffix}"
    return images


def daskBagOfFilePaths(folder, prefix, suffix, npartitions=None):
    """
    Returns dask.bag of absolute paths matching `folder/prefix*suffix`
    Specify bag partitions with `npartitions`; default None
    if `npartitions == None` then each filepath is its own partition
    """

    images = globPaths(folder, prefix, suffix)
    if npartitions is None: npartitions = len(images)
    return db.from_sequence(images, npartitions=npartitions)


def daskArrayBackedByHDF5(folder, prefix, suffix, dataset_path, stride=None):
    """
    Returns dask.array backed by HDF5 files matching absolute path `folder/prefix*suffix`
    You must specify hdf5 dataset path with `dataset_path`
    """

    error_message = "daskArrayBackedByHDF5 requires hdf5 files with .h5 or .hdf5 extension"
    assert (testPathExtensionForHDF5(suffix)), error_message
    images = globPaths(folder, prefix, suffix)
    if stride is not None:
        images = images[::stride]
    dsets = [readHDF5(image, dataset_path) for image in images]
    arrays = [da.from_array(dset, chunks=dset.shape) for dset in dsets]
    return da.stack(arrays, axis=0)


def readHDF5(image_path, dataset_path):
    """
    Returns (lazy) h5py array to dataset at `image_path[dataset_path]`
    """

    error_message = "readHDF5 requires hdf5 files with .h5 or .hdf5 extension"
    assert (testPathExtensionForHDF5(image_path)), error_message
    return h5py.File(image_path, 'r')[dataset_path]


def readImage(image_path, dataset_path=None):
    """
    Returns array like object for dataset at `image_path`
    If `image_path` is an hdf5 file, you must specify `dataset_path`
    If `image_path` is an hdf5 file, the return array is a lazy h5py File object
    Else, `image_path` is read by SimpleITK and an in memory numpy array is returned
    File format must be either hdf5 or supported by SimpleITK file readers
    """

    if testPathExtensionForHDF5(image_path):
        assert(dataset_path is not None), "Must provide dataset_path for .h5/.hdf5 files"
        return readHDF5(image_path, dataset_path)
    else:
        return sitk.GetArrayFromImage(sitk.ReadImage(image_path))


def writeHDF5(image_path, dataset_path, array):
    """
    Writes `array` to `image_path[dataset_path]` as an hdf5 file
    """

    with h5py.File(image_path, 'w') as f:
        dset = f.create_dataset(dataset_path, array.shape, array.dtype)
        dset[...] = array


def writeImage(image_path, array, spacing=None, axis_order='zyx'):
    """
    Writes `array` to `image_path`
    `image_path` extension determines format - must be supported by SimpleITK image writers
    Many formats support voxel spacing in the metadata, to specify set `spacing`
    If format supports voxel spacing in meta data, you can set with `spacing`
    Use `axis_order` to specify current axis ordering; default: 'zyx'
    """

    # argsort axis_order
    transpose_order = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(axis_order))]
    array = array.transpose(transpose_order[::-1])  # sitk inverts axis order
    img = sitk.GetImageFromArray(array)
    if spacing is not None:
        spacing = spacing[transpose_order]
        img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, image_path)


def ensureArray(reference, dataset_path):
    """
    """

    if not isinstance(reference, np.ndarray):
        if not isinstance(reference, str):
            raise ValueError("image references must be ndarrays or filepaths")
        reference = readImage(reference, dataset_path)[...]  # hdf5 arrays are lazy
    return reference


def stack_to_hdf5(stack_path, write_path, dims, dtype):
    """
    """

    stack = np.fromfile(stack_path, dtype=dtype).reshape(dims)
    writeHDF5(write_path, '/default', stack)


def distributed_stack_to_hdf5(
    folder, prefix, suffix,
    dims, dtype,
    cluster_kwargs={}
):
    """
    """

    stack_paths = globPaths(folder, prefix, suffix)
    write_paths = [path.splitext(s)[0] + '.h5' for s in stack_paths]
    nimages = len(stack_paths)
    stack_paths_b = db.from_sequence(stack_paths, npartitions=nimages)
    write_paths_b = db.from_sequence(write_paths, npartitions=nimages)
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:
        stack_paths_b.map(stack_to_hdf5, write_paths_b, dims, dtype).compute()

