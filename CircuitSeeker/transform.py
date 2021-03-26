import SimpleITK as sitk
import CircuitSeeker.utility as ut


def apply_transform(
    fix, mov,
    fix_spacing, mov_spacing,
    transform_list,
    transform_spacing=None,
    fix_origin=None,
    mov_origin=None,
    ):
    """
    """

    # convert images to sitk objects
    dtype = fix.dtype
    fix = ut.numpy_to_sitk(fix, fix_spacing, fix_origin)
    mov = ut.numpy_to_sitk(mov, mov_spacing, mov_origin)

    # default transform spacing is fixed voxel spacing
    if transform_spacing is None:
        transform_spacing = fix_spacing

    # construct transform
    transform = sitk.CompositeTransform(3)
    for t in transform_list:
        if len(t.shape) == 2:
            t = ut.matrix_to_affine_transform(t)
        elif len(t.shape) == 4:
            t = ut.field_to_displacement_field_transform(t, transform_spacing)
        transform.AddTransform(t)

    # set up resampler object
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.Cast(fix, sitk.sitkFloat32))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    # execute, return as numpy array
    resampled = resampler.Execute(sitk.Cast(mov, sitk.sitkFloat32))
    return sitk.GetArrayFromImage(resampled).astype(dtype)


