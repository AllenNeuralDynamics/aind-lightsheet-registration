"""Map every voxel of a brain to its mirror across the midline, by registering the brain's L→R flip to itself."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def mirror_indices(image: sitk.Image, mask: sitk.Image) -> np.ndarray:
    """Continuous index of each voxel's mirror, from an affine registration of the flipped brain at half resolution.

    The brain's array x axis must run across the midline.

    Parameters
    ----------
    image : sitk.Image
        Brain, non-negative.
    mask : sitk.Image
        Tissue mask on ``image``'s grid.

    Returns
    -------
    numpy.ndarray
        Float32 (3, z, y, x): the (z, y, x) continuous index of every voxel's mirror.
    """
    logi = sitk.Log(sitk.Clamp(sitk.Cast(image, sitk.sitkFloat32), lowerBound=1.0)) * sitk.Cast(mask, sitk.sitkFloat32)
    small = sitk.BinShrink(sitk.SmoothingRecursiveGaussian(logi, image.GetSpacing()[0]), [2, 2, 2])
    flipped = sitk.GetImageFromArray(sitk.GetArrayFromImage(small)[:, :, ::-1].copy())
    flipped.CopyInformation(small)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    init = sitk.CenteredTransformInitializer(
        small, flipped, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    reg.SetInitialTransform(init, inPlace=False)
    tx = reg.Execute(small, flipped)
    affine = sitk.AffineTransform(tx.GetNthTransform(0) if tx.GetName() == "CompositeTransform" else tx)

    # p maps to T(p) in the flipped image, whose index i shows the original at the x-flipped index; in index terms
    # mirror = (D^-1 A D) i + D^-1 (A (o - c) + c + t - o), with D = direction * spacing
    nx, ny, nz = image.GetSize()
    origin = np.array(image.GetOrigin())
    d = np.array(image.GetDirection()).reshape(3, 3) * np.array(image.GetSpacing())
    a = np.array(affine.GetMatrix()).reshape(3, 3)
    center, translation = np.array(affine.GetCenter()), np.array(affine.GetTranslation())
    linear = np.linalg.solve(d, a @ d)
    constant = np.linalg.solve(d, a @ (origin - center) + center + translation - origin)
    i, j, k = (np.arange(n, dtype=np.float32) for n in (nx, ny, nz))
    out = np.empty((3, nz, ny, nx), np.float32)
    for axis, row in zip((2, 1, 0), range(3), strict=True):  # ITK index axis row -> array axis
        out[axis] = (
            linear[row, 0] * i[None, None, :] + linear[row, 1] * j[None, :, None] + linear[row, 2] * k[:, None, None]
        ) + constant[row]
    # the flip reflects about the shrunk image's center, which is not the full grid's
    flip_center = small.TransformContinuousIndexToPhysicalPoint([(small.GetSize()[0] - 1) / 2, 0.0, 0.0])
    out[2] = 2 * image.TransformPhysicalPointToContinuousIndex(flip_center)[0] - out[2]
    return out
