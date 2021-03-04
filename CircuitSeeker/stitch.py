import numpy as np
import dask.array as da
import copy
from itertools import product


def weight_block(block, blocksize):
    """
    """

    overlaps = blocksize // 2
    weights = da.ones(blocksize % 2 + 2, dtype=np.float32)
    pads = [(2*p-1, 2*p-1) for p in overlaps]
    weights = da.pad(weights, pads, mode='linear_ramp', end_values=0)
    weights = weights.reshape(weights.shape + (1,))
    result = da.multiply(block, weights)
    return da.multiply(block, weights)


def merge_overlaps(block, blocksize):
    """
    """

    p = blocksize // 2
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
    coords = da.stack(coords, axis=-1).astype(np.uint16)
    return da.rechunk(coords, chunks=tuple(blocksize + [3,]))


def affine_to_grid(matrix, grid):
    """
    """

    ndims = len(matrix.shape)
    matrix = matrix.astype(np.float32).squeeze()
    lost_dims = ndims - len(matrix.shape)

    mm = matrix[:3, :-1]
    tt = matrix[:3, -1]
    result = da.einsum('...ij,...j->...i', mm, grid) + tt

    result = result - grid

    if lost_dims > 0:
        result = result.reshape((1,)*lost_dims + result.shape)
    return result


def local_affine_to_displacement(shape, spacing, affines, blocksize):
    """
    """

    # augment the blocksize by the fixed overlap size
    pads = [2*(x//2) for x in blocksize]
    blocksize_with_overlap = list(np.array(blocksize) + pads)

    # get a grid used for each affine
    grid = position_grid(blocksize_with_overlap, blocksize_with_overlap)
    grid = grid * spacing.astype(np.float32)

    # wrap local_affines as dask array
    affines_da = da.from_array(affines, chunks=(1, 1, 1, 4, 4))

    # compute affine transforms as displacement fields, lazy dask arrays
    coords = da.map_blocks(
        affine_to_grid, affines_da, grid=grid,
        new_axis=[5,6], chunks=(1,1,1,)+tuple(grid.shape), dtype=np.float32,
    )

    # stitch affine position fields
    coords = stitch_fields(coords, blocksize)

    # crop to original shape
    coords = coords[:shape[0], :shape[1], :shape[2]]

    # return result
    return coords


