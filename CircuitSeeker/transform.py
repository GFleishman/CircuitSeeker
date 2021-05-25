import SimpleITK as sitk
import CircuitSeeker.utility as ut
import os, psutil


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

    # set global number of threads
    if "LSB_DJOB_NUMPROC" in os.environ:
        ncores = int(os.environ["LSB_DJOB_NUMPROC"])
    else:
        ncores = psutil.cpu_count(logical=False)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2*ncores)

    # set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetNumberOfThreads(2*ncores)

    # convert images to sitk objects
    dtype = fix.dtype
    fix = ut.numpy_to_sitk(fix, fix_spacing, fix_origin)
    mov = ut.numpy_to_sitk(mov, mov_spacing, mov_origin)

    # construct transform
    transform = sitk.CompositeTransform(3)
    for i, t in enumerate(transform_list):

        # affine transforms
        if len(t.shape) == 2:
            t = ut.matrix_to_affine_transform(t)

        # bspline parameters
        elif len(t.shape) == 1:
            t = ut.bspline_parameters_to_transform(t)

        # fields
        elif len(t.shape) == 4:
            # set transform_spacing
            if transform_spacing is None:
                sp = fix_spacing
            elif isinstance(transform_spacing[i], tuple):
                sp = transform_spacing[i]
            else:
                sp = transform_spacing
            # create field
            t = ut.field_to_displacement_field_transform(t, sp)

        # add to composite transform
        transform.AddTransform(t)

    # set up resampler object
    resampler.SetReferenceImage(sitk.Cast(fix, sitk.sitkFloat32))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    # execute, return as numpy array
    resampled = resampler.Execute(sitk.Cast(mov, sitk.sitkFloat32))
    return sitk.GetArrayFromImage(resampled).astype(dtype)


