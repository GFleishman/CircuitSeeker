import CircuitSeeker.fileio as csio
import CircuitSeeker.networks as csn

from stardist.models import StarDist3D
from csbdeep.utils import normalize

import numpy as np
from scipy.ndimage import zoom


def resample(image, image_spacing, target_spacing, order=1):
    """
    """

    # zoom image to match target voxel spacing
    return zoom(image, image_spacing/target_spacing, order=order)


def rescaleIntensity(image, target_histogram
    winsor_min=0.01, winsor_max=0.99):
    """
    """

    # linearly rescale intensities using Winsorized min/max
    cdf = np.cumsum(target_histogram[0])
    cdf = cdf / cdf[-1]
    new_mn = target_histogram[1, np.argmax(cdf >= winsor_min)]
    new_mx = target_histogram[1, np.argmax(cdf >= winsor_max)-1]
    old_mn, old_mx = np.quantile(image, [winsor_min, winsor_max])
    return (image - old_mn) * (new_mx - new_mn) / (old_mx - old_mn) + new_mn


def segmentPretrainedStardist(image, voxel_spacing,
    model='confocal_nucleargcamp', write_path=None,
    dataset_path=None):
    """
    """

    image = csio.ensureArray(image, dataset_path)
    ns = csn.getNetworkSpecs(model)
    image_res = resample(image, voxel_spacing, ns['training_image_voxel_spacing'])
    image_res = rescaleIntensity(image_res, np.load(ns['training_image_histogram_path']))
    image_norm = normalize(image_res, ns['norm_range'][0], ns['norm_range'][1])
    model = StarDist3D(None, name='stardist', basedir=ns['model_path'])
