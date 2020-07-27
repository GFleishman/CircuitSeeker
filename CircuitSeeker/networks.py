import numpy as np


# include info useful for matching pretrained network to new data
STARDIST_CONFOCAL_NUCLEARGCAMP = {
    network_path:'../networks/stardist_confocal_nucleargcamp',
    training_image_path:'../data/confocal/',
    training_labels_path:'',
    training_image_orientation:'xyz',
    training_image_voxel_spacing:np.array([0.5189, 0.5189, 1.0]),
    training_image_histogram_path:'',
}


def getNetworkSpecs(network):
    """
    """
    if network == 'confocal_nucleargcamp':
        return STARDIST_CONFOCAL_NUCLEARGCAMP

