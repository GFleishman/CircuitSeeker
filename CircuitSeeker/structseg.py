import CircuitSeeker.fileio as csio
import CircuitSeeker.networks as csn

from stardist.models import StarDist3D
from csbdeep.utils import normalize

import numpy as np
from scipy.ndimage import zoom

def toXYZ(order):
    """
    """

    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(order))]


def transpose(image, spacing, orientation, target_orientation):
    """
    """

    image_to_xyz = toXYZ(orientation)
    xyz_to_target = np.argsort(toXYZ(target_orientation))
    image_trans = image.transpose(image_to_xyz).transpose(xyz_to_target)
    spacing_trans = spacing[image_to_xyz][xyz_to_target]
    return image_trans.copy(), spacing_trans  # copy ensures c order after transpose


def resample(image, spacing, target_spacing, order=1):
    """
    """

    # zoom image to match target voxel spacing
    return zoom(image, spacing/target_spacing, order=order)


def rescaleIntensity(image, target_histogram,
    winsor_min=0.001, winsor_max=0.999):
    """
    """

    # linearly rescale intensities using Winsorized min/max
    cdf = np.cumsum(target_histogram[0])
    cdf = cdf / cdf[-1]
    new_mn = target_histogram[1, np.argmax(cdf >= winsor_min)]
    new_mx = target_histogram[1, np.argmax(cdf >= winsor_max)-1]
    old_mn, old_mx = np.quantile(image, [winsor_min, winsor_max])
    return (image - old_mn) * (new_mx - new_mn) / (old_mx - old_mn) + new_mn


def segmentPretrainedStardist(image, spacing, orientation,
    model='confocal_nucleargcamp', write_path=None,
    dataset_path=None, n_tiles=(16, 32, 2),
    prob_thresh=0.5, nms_thresh=0.3):
    """
    """

    # get network details
    ns = csn.getNetworkSpecs(model)
    target_orientation = ns['training_image_orientation']
    target_spacing = ns['training_image_voxel_spacing']
    target_hist = np.load(ns['training_image_histogram_path'])
    target_norm = ns['norm_range']
    network = ns['network_path']

    # load image and normalize
    image = csio.ensureArray(image, dataset_path)
    image_res, spacing_res = transpose(
        image, spacing, orientation, target_orientation,
    )
    image_res = resample(image_res, spacing_res, target_spacing)
    image_res = rescaleIntensity(image_res, target_hist)
    image_norm = normalize(image_res, target_norm[0], target_norm[1])

    # segment
    model = StarDist3D(None, name='stardist', basedir=network)
    pred, det = model.predict_instances(image_norm, n_tiles=n_tiles,
        prob_thresh=prob_thresh, nms_thresh=nms_thresh, verbose=True,
        sparse=True, nms_kwargs={'use_kdtree':True},
    )

    # return to native spacing and orientation
    pred_res = resample(pred, target_spacing, spacing_res, order=0)
    pred_res, _ = transpose(
        pred_res, spacing_res, target_orientation, orientation
    )

    # save results
    if write_path is not None:
        csio.writeImage(write_path, pred_res, axis_order=orientation)
    return pred

