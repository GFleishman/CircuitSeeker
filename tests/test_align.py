import numpy as np
from pytest import fixture
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from CircuitSeeker.transform import apply_transform
import CircuitSeeker.utility as ut
import CircuitSeeker.align as align


@fixture
def dummy_fix_image():
    """
    """

    x = np.zeros((40,)*3, dtype=np.float32)
    x[14:26, 14:26, 14:26] = 1
    return gaussian_filter(x, 2)


@fixture
def dummy_affine_matrix():
    """
    """

    a = (4, 6, 3)
    b = (0, 0, 10 * np.pi/180)
    c = (1.1, 1, 1)
    d = (0, 0, 0.05)
    params = np.concatenate((a, b, c, d))
    return ut.physical_parameters_to_affine_matrix(params, (0,)*3)


@fixture
def dummy_mov_image(
    dummy_fix_image,
    dummy_affine_matrix,
):
    """
    """

    return apply_transform(
        dummy_fix_image, dummy_fix_image, (1.,)*3, (1.,)*3,
        transform_list=[dummy_affine_matrix],
    )


def test_resolve_sampling(
    dummy_fix_image, dummy_mov_image,
):
    """
    """

    X = align.resolve_sampling(
        dummy_fix_image, dummy_mov_image,
        dummy_fix_image[::2, ::2, ::2],
        dummy_mov_image[::2, ::2, ::2],
        (1.,)*3, (1.,)*3, (2, 3, 4),
    )
    assert X[0].shape == (20, 14, 10)
    assert X[1].shape == (20, 14, 10)
    assert X[2].shape == (20, 10, 10)
    assert X[3].shape == (20, 10, 10)
    assert np.all(X[4] == np.array([2, 3, 4]))
    assert np.all(X[5] == np.array([2, 3, 4]))
    assert np.all(X[6] == np.array([2, 4, 4]))
    assert np.all(X[7] == np.array([2, 4, 4]))


def test_images_to_sitk(
    dummy_fix_image, dummy_mov_image,
):
    """
    """

    a, b, c, d = align.images_to_sitk(
        dummy_fix_image, dummy_mov_image,
        dummy_fix_image, dummy_mov_image,
        (1,)*3, (2,)*3, (1,)*3, (2,)*3,
        (1, 2, 3), (4, 5, 6),
    )
    assert isinstance(a, sitk.Image)
    assert isinstance(b, sitk.Image)
    assert isinstance(c, sitk.Image)
    assert isinstance(d, sitk.Image)
    assert a.GetSpacing() == (1.,)*3
    assert b.GetSpacing() == (2,)*3
    assert c.GetSpacing() == (1.,)*3
    assert d.GetSpacing() == (2,)*3
    assert a.GetOrigin()[::-1] == (1, 2, 3)
    assert b.GetOrigin()[::-1] == (4, 5, 6)
    assert c.GetOrigin()[::-1] == (1, 2, 3)
    assert d.GetOrigin()[::-1] == (4, 5, 6)


def test_random_affine_search(
    dummy_fix_image, dummy_mov_image,
):
    """
    """

    affine = align.random_affine_search(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3, 1000,
        max_translation=7,
        max_rotation=15 * np.pi/180,
        max_scale=1.15,
        max_shear=0.07,
        metric='MS',
    )[0]

    aligned = apply_transform(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3,
        transform_list=[affine,],
    )

    before = np.sum((dummy_fix_image - dummy_mov_image)**2)
    after = np.sum((dummy_fix_image - aligned)**2)

    assert isinstance(affine, np.ndarray)
    assert affine.shape == (4, 4)
    assert after <= before


def test_rigid_alignment(
    dummy_fix_image, dummy_mov_image,
):
    """
    """

    affine = align.affine_align(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3, rigid=True,
        metric='MS',
        optimizer_args={
            'learningRate':.1,
            'minStep':.1,
            'numberOfIterations':100,
        }
    )

    aligned = apply_transform(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3,
        transform_list=[affine,],
    )

    before = np.sum((dummy_fix_image - dummy_mov_image)**2)
    after = np.sum((dummy_fix_image - aligned)**2)

    assert isinstance(affine, np.ndarray)
    assert affine.shape == (4, 4)
    assert after <= before


def test_affine_alignment(
    dummy_fix_image, dummy_mov_image,
):
    """
    """

    affine = align.affine_align(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3,
        metric='MS',
        optimizer_args={
            'learningRate':.1,
            'minStep':.1,
            'numberOfIterations':100,
        }
    )

    aligned = apply_transform(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3,
        transform_list=[affine,],
    )

    before = np.sum((dummy_fix_image - dummy_mov_image)**2)
    after = np.sum((dummy_fix_image - aligned)**2)

    assert isinstance(affine, np.ndarray)
    assert affine.shape == (4, 4)
    assert after <= before


def test_deformable_align(
    dummy_fix_image, dummy_mov_image,
):
    """
    """

    rigid = align.affine_align(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3, rigid=True,
        metric='MS',
        optimizer_args={
            'learningRate':.1,
            'minStep':.1,
            'numberOfIterations':100,
        }
    )

    deform = align.deformable_align(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3, 10.0, (1,),
        static_transform_list=[rigid,],
        metric='MS',
        optimizer_args={
            'learningRate':.1,
            'minStep':.1,
            'numberOfIterations':100,
        }
    )[1]

    rigid_aligned = apply_transform(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3,
        transform_list=[rigid,],
    )

    deform_aligned = apply_transform(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3,
        transform_list=[rigid, deform,],
    )

    before = np.sum((dummy_fix_image - dummy_mov_image)**2)
    middle = np.sum((dummy_fix_image - rigid_aligned)**2)
    after = np.sum((dummy_fix_image - deform_aligned)**2)

    assert isinstance(deform, np.ndarray)
    assert deform.shape == (40,)*3 + (3,)
    assert middle <= before
    assert after <= middle


def test_alignment_pipeline(
    dummy_fix_image, dummy_mov_image,
):
    """
    """

    affine, deform = align.alignment_pipeline(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3, ['random', 'rigid', 'affine', 'deform'],
        metric='MS',
        optimizer_args={
            'learningRate':.1,
            'minStep':.1,
            'numberOfIterations':100,
        },
        random_kwargs={
            'optimizer_args':{},
            'random_iterations':1000,
            'max_translation':7,
            'max_rotation':15 * np.pi/180,
            'max_scale':1.15,
            'max_shear':0.07,
            'print_running_improvements':True,
        },
        deform_kwargs={
            'control_point_spacing':10.0,
            'control_point_levels':(1,),
        },
    )

    aligned = apply_transform(
        dummy_fix_image, dummy_mov_image,
        (1.,)*3, (1.,)*3,
        transform_list=[affine, deform,],
    )

    before = np.sum((dummy_fix_image - dummy_mov_image)**2)
    after = np.sum((dummy_fix_image - aligned)**2)

    assert isinstance(affine, np.ndarray)
    assert isinstance(deform, np.ndarray)
    assert affine.shape == (4, 4)
    assert deform.shape == (40,)*3 + (3,)
    assert after <= before

