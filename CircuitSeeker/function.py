import numpy as np
import zarr
import ClusterWrap
import dask.array as da
from numcodecs import Blosc


def deltafoverf(
    array,
    window_size,
):
    """
    """

    # create container to store result
    new_shape = (array.shape[0] - window_size,) + array.shape[1:]
    dff = np.empty(new_shape, dtype=np.float16)  # TODO: user specifies precision

    # get window start and stop indices and sum image
    w_start, w_stop = 0, window_size
    sum_image = np.sum(array[w_start:w_stop], axis=0)

    # loop over frames
    for iii in range(window_size, array.shape[0]):
        baseline = sum_image / window_size
        dff[iii - window_size] = (array[iii] - baseline) / (baseline + 1)
        sum_image = sum_image - array[w_start] + array[iii]
        w_start += 1

    # return result
    return dff


def distributed_deltafoverf(
    zarr_array,
    window_size,
    batch_size,
    write_path,
    compression_level=4,
    cluster_kwargs={},
):
    """
    """

    # launch cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # wrap data as dask array
        chunks = (batch_size,) + zarr_array.chunks[1:]
        array_da = da.from_array(zarr_array, chunks=chunks)

        # define asymmetric depth for time axis only
        depth = {**{0:(window_size, 0)}, **{i: 0 for i in range(1, zarr_array.ndim)}}

        # define closure for deltafoverf
        def wrapped_deltafoverf(array, block_info=None):
            dff = deltafoverf(array, window_size)
            if block_info[0]['chunk-location'][0] == 0:
                pad = [(window_size, 0),] + [(0, 0),]*(dff.ndim - 1)
                dff = np.pad(dff, pad, mode='empty')
            return dff

        # map deltafoverf function over blocks
        dff = da.map_overlap(
            wrapped_deltafoverf, array_da,
            depth=depth,
            dtype=np.float16,
            boundary='none',
            trim=False,
        )

        # trim beginning and rechunk to single frames
        dff = dff[window_size:].rechunk(zarr_array.chunks)

        # write to output zarr
        compressor = Blosc(
            cname='zstd',
            clevel=compression_level,
            shuffle=Blosc.BITSHUFFLE,
        )
        dff_disk = zarr.open(
            write_path, 'w',
            shape=dff.shape,
            chunks=zarr_array.chunks,
            dtype=dff.dtype,
            compressor=compressor,
        )
        da.to_zarr(dff, dff_disk)

        # return reference to zarr store
        return dff_disk

