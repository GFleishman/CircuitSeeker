import numpy as np
import dask.array as da
import copy
from itertools import product

def weight_block(block, blocksize, block_info=None):
    """
    """

    # compute essential parameters
    overlaps = (np.array(blocksize) // 2).astype(int)

    # determine which faces need linear weighting
    core_shape, pads_ones, pads_linear = [], [], []
    block_index = block_info[0]['chunk-location']
    block_grid = block_info[0]['num-chunks']
    for i in range(3):

        # get core shape and pad sizes
        p = overlaps[i]
        core = 3 if blocksize[i] % 2 else 2
        pad_ones, pad_linear = [0, 0], [2*p-1, 2*p-1]
        if block_index[i] == 0:
            pad_ones[0], pad_linear[0] = 2*p-1, 0
        if block_index[i] == block_grid[i] - 1:
            pad_ones[1], pad_linear[1] = 2*p-1, 0
        core_shape.append(core)
        pads_ones.append(tuple(pad_ones))
        pads_linear.append(tuple(pad_linear))

    # create weights core
    weights = np.ones(core_shape, dtype=np.float32)

    # extend weights
    weights = da.pad(
        weights, pads_ones, mode='constant', constant_values=1,
    )
    weights = da.pad(
        weights, pads_linear, mode='linear_ramp', end_values=0,
    )
    weights = weights[..., None]

    # multiply data by weights and return
    return da.multiply(block, weights)


def merge_overlaps(block, blocksize):
    """
    """

    p = np.array(blocksize) // 2
    core = [slice(2*x, -2*x) for x in p]
    result = np.copy(block[tuple(core)])

    # faces
    for ax in range(3):
        # the left side
        slc1 = [slice(None, None)]*3
        slc1[ax] = slice(0, p[ax])
        slc2 = copy.deepcopy(core)
        slc2[ax] = slice(0, p[ax])
        result[tuple(slc1)] += block[tuple(slc2)]
        # the right side
        slc1 = [slice(None, None)]*3
        slc1[ax] = slice(-1*p[ax], None)
        slc2 = copy.deepcopy(core)
        slc2[ax] = slice(-1*p[ax], None)
        result[tuple(slc1)] += block[tuple(slc2)]

    # edges
    for edge in product([0, 1], repeat=2):
        for ax in range(3):
            pp = np.delete(p, ax)
            left = [slice(None, pe) for pe in pp]
            right = [slice(-1*pe, None) for pe in pp]
            slc1 = [left[i] if e == 0 else right[i] for i, e in enumerate(edge)]
            slc2 = copy.deepcopy(slc1)
            slc1.insert(ax, slice(None, None))
            slc2.insert(ax, core[ax])
            result[tuple(slc1)] += block[tuple(slc2)]

    # corners
    for corner in product([0, 1], repeat=3):
        left = [slice(None, pe) for pe in p]
        right = [slice(-1*pe, None) for pe in p]
        slc = [left[i] if c == 0 else right[i] for i, c in enumerate(corner)]
        result[tuple(slc)] += block[tuple(slc)]

    return result


def stitch_fields(fields, blocksize):
    """
    """

    # weight block edges
    weighted_fields = da.map_blocks(
        weight_block, fields, blocksize=blocksize, dtype=np.float32,
    )

    # remove block index dimensions
    sh = fields.shape[:3]
    list_of_blocks = [[[[weighted_fields[i,j,k]] for k in range(sh[2])]
                                                 for j in range(sh[1])]
                                                 for i in range(sh[0])]
    aug_fields = da.block(list_of_blocks)

    # merge overlap regions
    overlaps = tuple(blocksize.astype(np.int16) // 2) + (0,)

    return da.map_overlap(
        merge_overlaps, aug_fields, blocksize=blocksize,
        depth=overlaps, boundary=0., trim=False,
        dtype=np.float32, chunks=list(blocksize)+[3,],
    )


def position_grid(shape, blocksize):
    """
    """

    coords = da.meshgrid(*[range(x) for x in shape], indexing='ij')
    coords = da.stack(coords, axis=-1).astype(np.int16)
    return da.rechunk(coords, chunks=tuple(blocksize) + (3,))


def affine_to_grid(matrix, grid):
    """
    """

    # reformat matrix, keep track of trimmed dimensions
    ndims = len(matrix.shape)
    matrix = matrix.astype(np.float32).squeeze()
    lost_dims = ndims - len(matrix.shape)

    # apply affine to coordinates
    mm = matrix[:3, :3]
    tt = matrix[:3, -1]
    result = da.einsum('...ij,...j->...i', mm, grid) + tt

    # convert positions to displacements
    result = result - grid

    # restore trimmed dimensions
    if lost_dims > 0:
        result = result.reshape((1,)*lost_dims + result.shape)
    return result


def local_affine_to_displacement(shape, spacing, affines, blocksize):
    """
    """

    # define some helpful variables
    overlaps = list(blocksize // 2)
    nblocks = affines.shape[:3]

    # adjust affines for block origins
    for i in range(np.prod(nblocks)):
        x, y, z = np.unravel_index(i, nblocks)
        origin = np.maximum(
            np.array(blocksize) * [x, y, z] - overlaps, 0,
        )
        origin = origin * spacing
        tl, tr = np.eye(4), np.eye(4)
        a, tl[:3, -1], tr[:3, -1] = affines[x, y, z], origin, -origin
        affines[x, y, z] = np.matmul(tl, np.matmul(a, tr))

    # get a coordinate grid
    grid = position_grid(
        np.array(blocksize) * nblocks, blocksize,
    )
    grid = grid * spacing.astype(np.float32)
    grid = grid[..., None]  # needed for map_overlap

    # wrap local_affines as dask array
    affines_da = da.from_array(
        affines, chunks=(1, 1, 1, 4, 4),
    )

    # strip dummy axis off grid
    def wrapped_affine_to_grid(x, y):
        y = y.squeeze()
        return affine_to_grid(x, y)

    # compute affine transforms as displacement fields, lazy dask arrays
    blocksize_with_overlaps = tuple(x+2*y for x, y in zip(blocksize, overlaps))
    coords = da.map_overlap(
        wrapped_affine_to_grid, affines_da, grid,
        depth=[0, tuple(overlaps)+(0, 0)],
        boundary=0,
        trim=False,
        align_arrays=False,
        dtype=np.float32,
        new_axis=[5,6],
        chunks=(1,1,1,) + blocksize_with_overlaps + (3,),
    )

    # stitch affine position fields
    coords = stitch_fields(coords, blocksize)

    # crop to original shape
    coords = coords[:shape[0], :shape[1], :shape[2]]

    # return result
    return coords


