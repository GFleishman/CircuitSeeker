import numpy as np
import zarr
from ClusterWrap.decorator import cluster
import dask.array as da


def deltafoverf(
    array,
    window_size,
):
    """
    """

    # create container to store result
    new_shape = (array.shape[0] - window_size,) + array.shape[1:]
    dff = np.empty(new_shape, dtype=np.float32)  # TODO: user specifies precision

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


@cluster
def distributed_deltafoverf(
    zarr_array,
    window_size,
    batch_size,
    write_path,
    compression_level=4,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # create dask array of start indices
    start_indices, start_index = [], 0
    while start_index + window_size < zarr_array.shape[0]:
        start_indices.append(start_index)
        start_index = start_index + batch_size - window_size
    start_indices_da = da.from_array(start_indices, chunks=(1,))

    # wrap deltafoverf function
    def wrapped_deltafoverf(index):
        data = zarr_array[index[0]:index[0]+batch_size]
        return deltafoverf(data, window_size)

    # map function to each block
    dff = da.map_blocks(
        wrapped_deltafoverf, start_indices_da,
        dtype=np.float32,
        new_axis=list(range(1, zarr_array.ndim)),
        chunks=(batch_size-window_size,) + zarr_array.chunks[1:],
    )

    # ensure the correct shape and rechunk for faster writing
    dff = dff[:zarr_array.shape[0] - window_size]
    dff = dff.rechunk((1,) + zarr_array.chunks[1:])

    # persist on cluster, write to zarr, return reference
    dff = dff.persist()
    da.to_zarr(dff, write_path)
    return zarr.open(write_path, mode='r+')

