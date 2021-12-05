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
    zarr_path,
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

        # lazy load zarr to get metadata
        metadata = zarr.open(zarr_path, 'r')

        # get block start indices
        start_indices, start_index = [], 0
        while start_index + window_size < metadata.shape[0]:
            start_indices.append(start_index)
            start_index = start_index + batch_size - window_size

        # convert to dask array
        start_indices_da = da.from_array(start_indices, chunks=(1,))

        # wrap deltafoverf function
        def wrapped_deltafoverf(index):
            zarr_file = zarr.open(zarr_path, 'r')
            data = zarr_file[index[0]:index[0]+batch_size]
            return deltafoverf(data, window_size)
 
        # map function to each block
        dff = da.map_blocks(
            wrapped_deltafoverf, start_indices_da,
            dtype=np.float16,
            new_axis=list(range(1, metadata.ndim)),
            chunks=(batch_size-window_size,) + metadata.chunks[1:],
        )

        # ensure the correct shape and rechunk for faster writing
        dff = dff[:metadata.shape[0] - window_size]
        dff = dff.rechunk((1,) + metadata.chunks[1:])

        # persist dff before writing to zarr, prevents RAM conflicts
        dff = dff.persist()

        # write to output zarr
        compressor = Blosc(
            cname='zstd',
            clevel=compression_level,
            shuffle=Blosc.BITSHUFFLE,
        )
        dff_disk = zarr.open(
            write_path, 'w',
            shape=dff.shape,
            chunks=metadata.chunks,
            dtype=dff.dtype,
            compressor=compressor,
        )
        da.to_zarr(dff, dff_disk)

        # return reference to zarr store
        return dff_disk

