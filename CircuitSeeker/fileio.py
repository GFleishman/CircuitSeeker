import SimpleITK as sitk
import h5py
from glob import glob
from os import path

import dask.bag as db
import dask.array as da


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
    npartitions = len(images) if npartitions is None
    return db.from_sequence(images, npartitions=npartitions)


def daskArrayBackedByHDF5(folder, prefix, suffix, dataset_path):
    """
    Returns dask.array backed by HDF5 files matching absolute path `folder/prefix*suffix`
    You must specify hdf5 dataset path with `dataset_path`
    """

    error_message = "daskArrayBackedByHDF5 requires hdf5 files with .h5 or .hdf5 extension"
    assert (path.splitext(suffix)[-1] in ['.h5', '.hdf5']), error_message
    images = globPaths(folder, prefix, suffix)
    dsets = [readHDF5(image, dataset_path) for image in images]
    arrays = [da.from_array(dset, chunks=(256,)*dset.ndim) for dset in dsets]
    return da.stack(arrays, axis=0)


def readHDF5(image_path, dataset_path):
    """
    Returns (lazy) h5py array to dataset at `image_path[dataset_path]`
    """

    error_message = "readHDF5 requires hdf5 files with .h5 or .hdf5 extension"
    assert (path.splitext(image_path)[-1] in ['.h5', '.hdf5'], error_message
    return h5py.File(image_path, 'r')[dataset_path]


def readImage(image_path, dataset_path=None)
    """
    Returns array like object for dataset at `image_path`
    If `image_path` is an hdf5 file, you must specify `dataset_path`
    If `image_path` is an hdf5 file, the return array is a lazy h5py File object
    Else, `image_path` is read by SimpleITK and an in memory numpy array is returned
    File format must be either hdf5 or supported by SimpleITK file readers
    """

    if path.splitext(image_path)[-1] in ['.h5', '.hdf5']:
        assert(dataset_path is not None), "Must provide dataset_path for .h5/.hdf5 files"
        return readHDF5(image_path, dataset_path)
    else:
        return sitk.GetArrayFromImage(sitk.ReadImage(image_path))

