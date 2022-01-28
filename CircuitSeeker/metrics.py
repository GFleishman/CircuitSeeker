import numpy as np
import SimpleITK as sitk
from CircuitSeeker._wrap_irm import configure_irm
import CircuitSeeker.utility as ut


def patch_mutual_information(
    fix,
    mov,
    spacing,
    radius,
    stride,
    percentile_cutoff=0,
    fix_mask=None,
    mov_mask=None,
    return_metric_image=False,
    **kwargs,
):
    """
    """

    # create sitk versions of data
    fix_sitk = ut.numpy_to_sitk(fix.transpose(2, 1, 0), spacing[::-1])
    fix_sitk = sitk.Cast(fix_sitk, sitk.sitkFloat32)
    mov_sitk = ut.numpy_to_sitk(mov.transpose(2, 1, 0), spacing[::-1])
    mov_sitk = sitk.Cast(mov_sitk, sitk.sitkFloat32)

    # convert to voxel units
    radius = np.round(radius / spacing).astype(np.uint16)
    stride = np.round(stride / spacing).astype(np.uint16)

    # determine patch sample centers
    samples = np.zeros_like(fix)
    sample_points = tuple(slice(r, -r, s) for r, s in zip(radius, stride))
    samples[sample_points] = 1

    # mask sample points
    if fix_mask is not None: samples = samples * fix_mask
    if mov_mask is not None: samples = samples * mov_mask

    # convert to list of coordinates
    samples = np.column_stack(np.nonzero(samples))

    # create irm for evaluating metric
    irm = configure_irm(**kwargs)

    # create container for metric image
    if return_metric_image:
        metric_image = np.zeros(fix.shape, dtype=np.float32)

    # loop over patches and evaluate
    scores = []
    for sample in samples:

        # get patches
        patch = tuple(slice(s-r, s+r+1) for s, r in zip(sample, radius))
        f = fix_sitk[patch]
        m = mov_sitk[patch]

        # evaluate metric
        try:
            scores.append( irm.MetricEvaluate(f, m) )
        except Exception as e:
            scores.append( 0 )

        # update metric image
        if return_metric_image:
            metric_image[patch] = scores[-1]

    # threshold scores
    scores = np.array(scores)
    if percentile_cutoff > 0:
        cutoff = np.percentile(-scores, percentile_cutoff)
        scores = scores[-scores > cutoff]

    # return results
    if return_metric_image:
        return np.mean(scores), metric_image
    else:
        return np.mean(scores)

