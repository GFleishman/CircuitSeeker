import numpy as np
import zarr
from numcodecs import Blosc
from ClusterWrap.decorator import cluster
import dask.array as da
from scipy.ndimage import find_objects
from dask.distributed import as_completed


def deltafoverf(
    array,
    window_size,
):
    """
    Computes change in activity level as a percentage of baseline for time series
    data. E.g. the Delta F over F for calcium imaging data.

    Parameters
    ----------
    array : ndarray
        The time series data. Time should be the first axis.

    window_size : int
        The number of frames used to compute the rolling average baseline

    Returns
    -------
    dff : ndarray
        Only frames with a complete baseline are returned, so the shape is
        window_size fewer frames than the input.
    """

    # create container to store result
    new_shape = (array.shape[0] - window_size,) + array.shape[1:]
    dff = np.empty(new_shape, dtype=np.float32)

    # initialize window start index and the sum image
    w_start = 0
    sum_image = np.sum(array[w_start:window_size], axis=0, dtype=np.float32)

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
    Calls deltafoverf on chunks of a larger-than-memory time series in parallel
    on distributed hardware.

    Parameters
    ----------
    zarr_array : zarr.Array
        The time series data. Time should be the first axis.

    window_size : int
        The number of frames used to compute the rolling average baseline

    batch_size : int
        The number of frames that make up each chunk of the time series data
        to work on in parallel.

    write_path : string
        The path where the computed delta f over f zarr file should be written

    compression_level : int in range [0, 9] (default: 4)
        The amount the final output is compressed. Lower numbers will write faster
        but take up more space. Be warned, large compression_level can result
        in very long compression/write times.

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, the close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be ClusterWrap.janelia_lsf_cluster.
        If on a workstation this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    Returns
    -------
    """

    # create dask array of start indices
    start_indices, start_index = [], 0
    while start_index + window_size < zarr_array.shape[0]:
        start_indices.append(start_index)
        start_index = start_index + batch_size - window_size

    # create output zarr
    compressor = Blosc(cname='zstd', clevel=4, shuffle=Blosc.BITSHUFFLE)
    output_zarr = zarr.open(
        write_path, 'w',
        shape=(zarr_array.shape[0] - window_size,) + zarr_array.shape[1:],
        chunks=(1,) + zarr_array.shape[1:],
        dtype=np.float32,
        compressor=compressor,
    )

    # wrap deltafoverf function
    def wrapped_deltafoverf(index):
        read_slice = slice(index, index+batch_size)
        write_slice = slice(index, index+batch_size-window_size)
        output_zarr[write_slice] = deltafoverf(zarr_array[read_slice], window_size)
        return True

    # submit all
    futures = cluster.client.map(wrapped_deltafoverf, start_indices)
    all_written = np.all(cluster.client.gather(futures))
    if not all_written: print('SOMETHING FAILED, CHECK LOGS')
    return output_zarr


def apply_cell_mask(
    masks,
    data,
    max_label=0,
):
    """
    """

    boxes = find_objects(masks, max_label=max_label)
    result = np.empty((len(boxes), + data.shape[0]), dtype=data.dtype)
    result.fill(np.nan)
    for iii, box in enumerate(boxes):
        if box:
            mask = masks[box] == iii + 1
            masked_crop = data[ (slice(None),) + box ]
            result[iii] = np.mean(masked_crop, axis=(1, 2, 3), where=mask)
    return result


@cluster
def distributed_apply_cell_mask(
    masks,
    zarr_array,
    batch_size,
    max_label=0,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # construct dask versions of inputs
    masks_d = cluster.client.scatter(masks, broadcast=True)
    start_indices = list(range(0, zarr_array.shape[0], batch_size))

    # wrap function
    def wrapped_apply_cell_mask(index, masks_d):
        data = zarr_array[index:index+batch_size]
        return apply_cell_mask(masks_d, data, max_label=max_label)

    # submit all blocks
    futures = cluster.client.map(
        wrapped_apply_cell_mask, start_indices,
        masks_d=masks_d
    )
    future_keys = [f.key for f in futures]

    # collect results and store in array
    nrows = max_label if max_label else masks.max()
    results = np.empty((nrows, zarr_array.shape[0]), dtype=zarr_array.dtype)
    for batch in as_completed(futures, with_results=True).batches():
        for future, result in batch:
            iii = future_keys.index(future.key)
            results[:, iii*batch_size:(iii+1)*batch_size] = result

    return results

