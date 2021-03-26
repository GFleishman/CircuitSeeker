import numpy as np
import SimpleITK as sitk


def skip_sample(image, spacing, ss_spacing):
    """
    """

    ss = np.maximum(np.round(ss_spacing / spacing), 1).astype(np.int)
    image = image[::ss[0], ::ss[1], ::ss[2]]
    spacing = spacing * ss
    return image, spacing


def numpy_to_sitk(image, spacing, origin=None, vector=False):
    """
    """

    image = sitk.GetImageFromArray(image.copy(), isVector=vector)
    image.SetSpacing(spacing[::-1])
    if origin is None:
        origin = np.zeros(len(spacing))
    image.SetOrigin(origin[::-1])
    return image


def invert_matrix_axes(matrix):
    """
    """

    corrected = np.eye(4)
    corrected[:3, :3] = matrix[:3, :3][::-1, ::-1]
    corrected[:3, -1] = matrix[:3, -1][::-1]
    return corrected


def affine_transform_to_matrix(transform):
    """
    """

    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape((3,3))
    matrix[:3, -1] = np.array(transform.GetTranslation())
    return invert_matrix_axes(matrix)


def matrix_to_affine_transform(matrix):
    """
    """

    matrix_sitk = invert_matrix_axes(matrix)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix_sitk[:3, :3].flatten())
    transform.SetTranslation(matrix_sitk[:3, -1].squeeze())
    return transform


def matrix_to_displacement_field(reference, matrix, spacing):
    """
    """

    nrows, ncols, nstacks = reference.shape
    grid = np.array(np.mgrid[:nrows, :ncols, :nstacks]).transpose(1,2,3,0)
    grid = grid * spacing
    mm, tt = matrix[:3, :3], matrix[:3, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt - grid


def field_to_displacement_field_transform(field, spacing):
    """
    """

    field = field.astype(np.float64)
    transform = numpy_to_sitk(field, spacing, vector=True)
    return sitk.DisplacementFieldTransform(transform)


