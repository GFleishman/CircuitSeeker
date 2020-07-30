import numpy as np
from os.path import abspath, dirname, join

package_dir = dirname(abspath(__file__))
networks_dir = join(package_dir, 'networks')
data_dir = join(package_dir, 'data')


# include info useful for matching pretrained network to new data
STARDIST_CONFOCAL_NUCLEARGCAMP = {
    'network_path':
        join(networks_dir, 'stardist_confocal_nucleargcamp'),
    'training_image_path':
        [join(data_dir, 'confocal', 'training_forebrain_crop.nrrd'),
         join(data_dir, 'confocal', 'training_hindbrain_crop.nrrd'),
        ],
    'training_labels_path':
        [join(data_dir, 'confocal', 'training_forebrain_crop_labels.nrrd'),
         join(data_dir, 'confocal', 'training_hindbrain_crop_labels.nrrd'),
        ],
    'training_image_orientation':
        'xyz',
    'training_image_voxel_spacing':
        np.array([0.5189, 0.5189, 1.0]),
    'training_image_histogram_path':
        join(data_dir, 'confocal', 'training_image_histogram.npy'),
    'norm_range':
        np.array([1, 99.8]),
}


def getNetworkSpecs(network):
    """
    """
    if network == 'confocal_nucleargcamp':
        return STARDIST_CONFOCAL_NUCLEARGCAMP

