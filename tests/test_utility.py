import numpy as np
import SimpleITK as sitk
import CircuitSeeker.utility as ut


def test_skip_sample():
    """
    """

    image = np.arange(20**3).reshape((20,)*3)
    spacing = np.ones(3)
    result = ut.skip_sample(image, spacing, (2, 2.6, 0.5))
    assert np.all(result == image[::2, ::3, ::1])


def test_numpy_to_sitk():
    """
    """

    image = np.arange(20**3 * 3).reshape((20,)*3 + (3,))
    sitk_image = ut.numpy_to_sitk(image, (1,)*3, (0,)*3, vector=True)
    result = sitk_image.GetArrayFromImage()
    assert isinstance(result sitk.Image)
    assert np.all(image == result)


def test_invert_matrix_axes():
    """
    """

    matrix = np.arange(16).reshape((4,4))
    matrix[3] = [0, 0, 0, 1]
    correct = np.array( [[10, 9, 8, 11],
                         [6, 5, 4, 7],
                         [2, 1, 0, 3],
                         [0, 0, 0, 1]] )
    result = ut.invert_matrix_axes(matrix)
    assert np.all(result == correct)


def test_change_affine_matrix_origin():
    """
    """

    matrix = np.arange(16).reshape((4,4))
    matrix[3] = [0, 0, 0, 1]
    correct = np.array( [[0, 1, 2, 4],
                         [4, 5, 6, -7],
                         [8, 9, 10, -23],
                         [0, 0, 0, 1]] )
    result = ut.change_affine_matrix_origin(matrix, (2, 3, -1))
    assert np.all(result == correct)


def test_integration_matrix_to_affine_and_inverse():
    """
    """

    matrix = np.arange(16).reshape((4,4))
    matrix[3] = [0, 0, 0, 1]
    transform = ut.matrix_to_affine_transform(matrix)
    affine = ut.affine_transform_to_matrix(transform)
    assert isinstance(transform, sitk.AffineTransform)
    assert np.all(matrix == affine)


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
    assert np.all(matrix == matrix2)


def test_matrix_to_displacement_field():
    """
    """

    scale, trans = (2, 3, 4), (2, 4, 6)
    shape, spacing = (10, 10, 10,), (0.5, 0.25, 2.0)
    matrix = np.diag(scale + (1,))
    matrix[:3, -1] = trans
    correct = np.array(np.mgrid[:10, :10, :10]).transpose(1,2,3,0)
    correct = correct * spacing * scale + trans
    field = ut.matrix_to_displacement_field(matrix, shape, spacing)
    assert np.all(field == correct)


def test_field_to_displacement_field_transform():
    """
    """

    scale, trans = (2, 3, 4), (2, 4, 6)
    shape, spacing = (10, 10, 10,), (0.5, 0.25, 2.0)
    matrix = np.diag(scale + (1,))
    matrix[:3, -1] = trans
    correct = np.array(np.mgrid[:10, :10, :10]).transpose(1,2,3,0)
    correct = correct * spacing * scale + trans
    field = ut.matrix_to_displacement_field(matrix, shape, spacing)
    transform = ut.field_to_displacement_field_transform(field, spacing)
    field2 = sitk.GetArrayFromImage(transform.GetDisplacementField())[..., ::-1]
    assert np.all(field2 == correct)


def test_bspline_parameters_to_transform():
    """
    """

    assert True

def test_bspline_to_displacement_field():
    """
    """

    assert True

def test_transform_list_to_composite_transform():
    """
    """

    assert True

def test_create_zarr():
    """
    """

    assert True

def test_numpy_to_zarr():
    """
    """

    assert True
    
