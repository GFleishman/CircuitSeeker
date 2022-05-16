import numpy as np
import SimpleITK as sitk
import zarr, os, tempfile
from pytest import fixture
import CircuitSeeker.utility as ut


@fixture
def dummy_affine_matrix():
    """
    """

    matrix = np.arange(16).reshape((4,4))
    matrix[3] = [0, 0, 0, 1]
    return matrix


@fixture
def dummy_scale_translate_matrix():
    """
    """

    scale, trans = (2, 3, 4), (2, 4, 6)
    matrix = np.diag(scale + (1,))
    matrix[:3, -1] = trans
    return matrix


@fixture
def dummy_bspline_params():
    """
    """

    a = (4,)*3    # mesh_size
    b = (-1,)*3   # mesh_origin
    c = (1,)*3    # mesh_spacing
    d = np.eye(3).ravel()    # mesh_directions
    e = np.ones(np.prod(a) * 3) * 6    #coefficients
    return np.concatenate((a, b, c, d, e))


@fixture
def dummy_vector_field():
    """
    """

    return np.ones((10,)*3 + (3,)) * 12


def test_skip_sample():
    """
    """

    image = np.arange(20**3).reshape((20,)*3)
    result, spacing = ut.skip_sample(image, (1, 1, 1), (2, 2.6, 0.5))
    assert np.all(result == image[::2, ::3, ::1])
    assert np.all(spacing == (2, 3, 1))
    

def test_numpy_to_sitk(dummy_vector_field):
    """
    """

    sitk_image = ut.numpy_to_sitk(dummy_vector_field, (1,)*3, (0,)*3, vector=True)
    result = sitk.GetArrayFromImage(sitk_image)
    assert isinstance(sitk_image, sitk.Image)
    assert np.all(dummy_vector_field == result)


def test_invert_matrix_axes(dummy_affine_matrix):
    """
    """

    correct = np.array( [[10, 9, 8, 11],
                         [6, 5, 4, 7],
                         [2, 1, 0, 3],
                         [0, 0, 0, 1]] )
    result = ut.invert_matrix_axes(dummy_affine_matrix)
    assert np.all(result == correct)


def test_change_affine_matrix_origin(dummy_affine_matrix):
    """
    """

    correct = np.array( [[0, 1, 2, 4],
                         [4, 5, 6, -7],
                         [8, 9, 10, -23],
                         [0, 0, 0, 1]] )
    result = ut.change_affine_matrix_origin(dummy_affine_matrix, (2, 3, -1))
    assert np.all(result == correct)


def test_integration_matrix_to_affine_and_inverse(dummy_affine_matrix):
    """
    """

    transform = ut.matrix_to_affine_transform(dummy_affine_matrix)
    affine = ut.affine_transform_to_matrix(transform)
    assert isinstance(transform, sitk.AffineTransform)
    assert np.all(dummy_affine_matrix == affine)


def test_integration_matrix_to_euler_and_inverse():
    """
    """

    params = np.array([np.pi, np.pi/2, np.pi/4, 10, 20, 30])
    transform = ut.parameters_to_euler_transform(params)
    params2 = ut.euler_transform_to_parameters(transform)
    matrix = ut.affine_transform_to_matrix(transform)
    transform2 = ut.matrix_to_euler_transform(matrix)
    matrix2 = ut.affine_transform_to_matrix(transform2)
    assert isinstance(transform, sitk.Euler3DTransform)
    assert isinstance(transform2, sitk.Euler3DTransform)
    assert np.all(params == params2)
    assert np.allclose(matrix, matrix2)


def test_physical_parameters_to_affine_matrix():
    """
    """

    params = np.array( [ 10, 20, 30,
                         np.pi/6, np.pi/4, np.pi/2,
                         1.05, 0.95, 0.8,
                         0.2, 0.3, 0.4, ] )
    array = ut.physical_parameters_to_affine_matrix(params, (0, 0, 0))
    correct = np.array([[ 0.0535884 , -0.71271186,  1.03889313, 17.44844071],
                        [ 0.94484514, -0.17477094,  0.70403709, 27.07414529],
                        [ 0.20343283,  0.21233866,  0.99801973, 36.22169328],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    assert np.allclose(array, correct)


def test_matrix_to_displacement_field(dummy_scale_translate_matrix):
    """
    """

    scale = np.diag(dummy_scale_translate_matrix)[:-1]
    trans = dummy_scale_translate_matrix[:3, -1]
    shape, spacing = (10, 10, 10,), (0.5, 0.25, 2.0)
    correct = np.array(np.mgrid[:10, :10, :10]).transpose(1,2,3,0) * spacing
    correct = correct * scale + trans - correct
    field = ut.matrix_to_displacement_field(dummy_scale_translate_matrix, shape, spacing)
    assert np.all(field == correct)


def test_field_to_displacement_field_transform(dummy_scale_translate_matrix):
    """
    """

    scale = np.diag(dummy_scale_translate_matrix)[:-1]
    trans = dummy_scale_translate_matrix[:3, -1]
    shape, spacing = (10, 10, 10,), (0.5, 0.25, 2.0)
    correct = np.array(np.mgrid[:10, :10, :10]).transpose(1,2,3,0) * spacing
    correct = correct * scale + trans - correct
    field = ut.matrix_to_displacement_field(dummy_scale_translate_matrix, shape, spacing)
    transform = ut.field_to_displacement_field_transform(field, spacing)
    field2 = sitk.GetArrayFromImage(transform.GetDisplacementField())[..., ::-1]
    assert np.all(field2 == correct)


def test_integration_bspline_parameters_to_transform_to_field(
    dummy_bspline_params,
):
    """
    """

    transform = ut.bspline_parameters_to_transform(dummy_bspline_params)
    field = ut.bspline_to_displacement_field(transform, (10,)*3, spacing=(0.1,)*3)
    correct = np.ones((10,)*3 + (3,)) * 6
    assert isinstance(transform, sitk.BSplineTransform)
    assert np.all(field == correct)


def test_relative_spacing():
    """
    """

    a = np.empty((4, 10, 1))
    b = np.empty((10, 5, 3))
    c = np.array((1, 1, 1))
    d = ut.relative_spacing(a, b, c)
    assert np.all(d == (2.5, 0.5, 3))


def test_transform_list_to_composite_transform(
    dummy_affine_matrix,
    dummy_scale_translate_matrix,
    dummy_bspline_params,
    dummy_vector_field,
):
    """
    """

    aff1 = dummy_affine_matrix
    aff2 = dummy_scale_translate_matrix
    bsp = dummy_bspline_params
    dvf = dummy_vector_field
    transform_list = [aff1, aff2, bsp, dvf]
    spacing = (None, None, None, (0.1, 0.1, 0.1))
    composite = ut.transform_list_to_composite_transform(
        transform_list, spacing,
    )

    aff1_transform = sitk.AffineTransform(composite.GetNthTransform(0))
    aff2_transform = sitk.AffineTransform(composite.GetNthTransform(1))
    aff1_params = aff1_transform.GetParameters()
    aff2_params = aff2_transform.GetParameters()
    bsp_params = composite.GetNthTransform(2).GetParameters()
    dvf_params = composite.GetNthTransform(3).GetParameters()

    assert isinstance(composite, sitk.CompositeTransform)
    assert composite.GetNumberOfTransforms() == 4
    assert len(aff1_params) == 12
    assert len(aff2_params) == 12
    assert len(bsp_params) == 192
    assert len(dvf_params) == 3000

    aff1_ = ut.affine_transform_to_matrix(aff1_transform)
    aff2_ = ut.affine_transform_to_matrix(aff2_transform)
    assert np.all(aff1 == aff1_)
    assert np.all(aff2 == aff2_)
    assert np.all(bsp_params == np.array((6,)))
    assert np.all(dvf_params == np.array((12,)))


def test_create_zarr():
    """
    """

    temp_directory = os.getcwd() + '/.circuitseeker_tests'
    with tempfile.TemporaryDirectory() as temp_directory:
        path = temp_directory + '/test_create_zarr.zarr'
        zarr_array = ut.create_zarr(path, (128,)*3, (32,)*3, np.float32)
        assert isinstance(zarr_array, zarr.Array)
        assert zarr_array.shape == (128,)*3
        assert zarr_array.chunks == (32,)*3
        assert zarr_array.dtype == np.float32


def test_numpy_to_zarr():
    """
    """

    array = np.ones((128,)*3)
    chunks = (32,)*3
    temp_directory = os.getcwd() + '/.circuitseeker_tests'
    with tempfile.TemporaryDirectory() as temp_directory:
        path = temp_directory + '/test_numpy_to_zarr.zarr'
        zarr_array = ut.numpy_to_zarr(array, chunks, path)
        assert isinstance(zarr_array, zarr.Array)
        assert zarr_array.shape == (128,)*3
        assert zarr_array.chunks == (32,)*3
        assert zarr_array.dtype == array.dtype

        
