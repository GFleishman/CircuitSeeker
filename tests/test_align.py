import numpy as np
from pytest import fixture
from CircuitSeeker.transform import apply_transform
import CircuitSeeker.utility as ut
import CircuitSeeker.align as align


@fixture
def dummy_fix_image():
    """
    """

    x = np.zeros((128,)*3, dtype=np.uint16)
    x[48:81, 48:81, 48:81] = 1
    return x


@fixture
def dummy_affine_matrix():
    """
    """

    a = (4, 6, 8)
    b = (0, 0, np.pi/4)
    c = (1.1, 1.2, 0.8)
    d = (0.1, 0.1, 0.2)
    params = np.concatenate((a, b, c, d))
    center = (64,)*3
    return ut.physical_parameters_to_affine_matrix(params, center)


@fixture
def dummy_mov_image(
    dummy_fix_image,
    dummy_affine_matrix,
):
    """
    """

    return apply_transform(
        dummy_fix_image, dummy_fix_image, (1,)*3, (1,)*3,
        transform_list=[dummy_affine_matrix],
    )


def test_skip_sample_images(
    dummy_fix_image, dummy_mov_image,
):
    """
    """

    import nrrd
    nrrd.write('./a.nrrd', dummy_fix_image)
    nrrd.write('./b.nrrd', dummy_mov_image)
    assert True


