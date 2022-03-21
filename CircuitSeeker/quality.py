import os, psutil
import numpy as np
import pyfftw
from itertools import product
from CircuitSeeker.transform import apply_transform


def bounded_fourier_shell_correlation(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    bounds,
):
    """
    """

    # require same data type
    if fix.dtype != mov.dtype:
        error = "Image must have the same data type"
        raise TypeError(error)

    # ensure datasets are on the same grid and spacing
    if fix.shape != mov.shape or not np.all(fix_spacing == mov_spacing):
        mov = apply_transform(
            fix, mov, fix_spacing, mov_spacing, transform_list=[np.eye(4),],
        )

    # establish data types
    if fix.dtype == np.uint16:
        fix = fix.astype(np.float32)
    if fix.dtype == np.float32:
        cdtype = np.complex64
    elif fix.dtype == np.float64:
        cdtype = np.complex128
    else:
        error = "Images must be uint16, float32, or float64 \n"
        error += "fix dtype is: " + str(fix.dtype) + " mov dtype is: " + str(mov.dtype)
        raise TypeError(error)

    # ensure datatypes stay the same
    if fix.dtype != mov.dtype:
        mov = mov.astype(fix.dtype)

    # determine cores available to set fft threads
    if "LSB_DJOB_NUMPROC" in os.environ:
        ncores = int(os.environ["LSB_DJOB_NUMPROC"])
    else:
        ncores = psutil.cpu_count(logical=False)

    # construct fft object
    out_sh = fix.shape[:-1] + (fix.shape[-1]//2 + 1,)  # assuming real valued input
    inp = pyfftw.empty_aligned(fix.shape, dtype=fix.dtype)
    outp = pyfftw.empty_aligned(out_sh, dtype=cdtype)
    fft = pyfftw.FFTW(inp, outp, axes=list(range(len(fix.shape))), threads=2*ncores)
    
    # we'll need to ensure memory layout is compatible with fft object
    flags_to_set = [f for f in ['C', 'F', 'O', 'W', 'A'] if inp.flags[f]]

    # get image frequencies
    inp[:] = np.require(fix, requirements=flags_to_set)
    fix_fourier = np.copy(fft())
    inp[:] = np.require(mov, requirements=flags_to_set)
    mov_fourier = fft()  # no need to copy the last one

    # get frequency position field amplitudes
    a, b = fix.shape, fix_spacing
    freqs = np.meshgrid(*[np.fft.fftfreq(x, y) for x, y in zip(a, b)], indexing='ij')
    freqs = np.array([f[..., :outp.shape[-1]] for f in freqs])
    freqs = np.linalg.norm(freqs, axis=0)**2

    # return frequency shell correlation
    shell = (freqs >= 1./bounds[1]) * (freqs <= 1./bounds[0])
    a = fix_fourier[shell].flatten()
    b = mov_fourier[shell].flatten()
    return np.real(np.corrcoef(a, b)[0, 1])


def cell_quality_score(
    image,
    spacing,
    bounds,
    shift_radius=1,
):
    """
    """

    # closure for correlation function
    CF = lambda a, b: bounded_fourier_shell_correlation(a, b, spacing, spacing, bounds)

    # define all shifts
    # TODO: does not include secondary diagonals (e.g. in 2d: i+1, j-1)
    shifts = product(range(0, shift_radius+1), repeat=3)

    # get correlation for all shifts
    correlations = []
    for iii, shift in enumerate(shifts):
        if shift == (0, 0, 0): continue
        slice_a = tuple(slice(0, -s) if s != 0 else slice(0, None) for s in shift)
        slice_b = tuple(slice(s, None) for s in shift)
        correlations.append( CF(image[slice_a], image[slice_b]) )

    # return median result
    return np.median(correlations)


def blockwise_cell_quality_score(
    image,
    spacing,
    bounds,
    radius,
    mask=None,
    **kwargs,
):
    """
    """

    # determine stride in voxels
    stride = np.round(radius/spacing).astype(int)

    # define weights for linear blend averaging
    ndim = len(image.shape)
    core = np.array([1.]).reshape((1,)*ndim)
    pad = tuple([s, s] for s in stride)
    weights = np.pad(core, pad, mode='linear_ramp')

    # pad the array to prevent edge effects in averaging
    image = np.pad(image, pad, mode='constant')
    mask = np.pad(mask, pad, mode='constant')

    # get valid sample point coordinates
    samples = np.zeros(image.shape, dtype=bool)
    sample_grid = tuple(slice(s, -s, s) for s in stride)
    samples[sample_grid] = 1
    if mask is not None: samples = samples * mask
    samples = np.nonzero(samples)

    # get container to hold scores
    scores = np.zeros(image.shape, dtype=np.float32)

    # score all blocks
    for iii, coordinate in enumerate(zip(samples[0], samples[1], samples[2])):
        if iii % 100 == 0: print("{} percent complete".format(iii/len(samples[0])))
        context = tuple(slice(x-r, x+r+1) for x, r in zip(coordinate, stride))
        score = cell_quality_score(image[context], spacing, bounds, **kwargs)
        scores[context] += score * weights

    # return result
    crop = tuple(slice(s, -s) for s in stride)
    return scores[crop]


def jaccard_filter(
    fix_mask,
    mov_mask,
    threshold,
):
    """
    """

    fix_mask = fix_mask > 0
    mov_mask = mov_mask > 0
    score = np.sum(fix_mask * mov_mask) / np.sum(fix_mask + mov_mask)
    return True if score >= threshold else False

