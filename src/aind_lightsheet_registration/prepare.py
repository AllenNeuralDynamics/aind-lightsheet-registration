"""Mask, bias-correct and normalize a brain image for registration.

Every function takes and returns SimpleITK images in mm and keeps their geometry, so the result stays in the
acquisition frame.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import SimpleITK as sitk

ThresholdMethod = Literal["otsu", "li", "triangle", "huang"]

_THRESHOLDS: dict[ThresholdMethod, Callable[..., sitk.Image]] = {
    "otsu": sitk.OtsuThreshold,
    "li": sitk.LiThreshold,
    "triangle": sitk.TriangleThreshold,
    "huang": sitk.HuangThreshold,
}


def _radius_voxels(image: sitk.Image, radius_mm: float) -> list[int]:
    return [max(0, round(radius_mm / s)) for s in image.GetSpacing()]


def brain_mask(
    image: sitk.Image,
    *,
    method: ThresholdMethod = "otsu",
    log_intensity: bool = True,
    closing_radius_mm: float = 0.2,
) -> sitk.Image:
    """Tissue mask: an automatic threshold, the largest component, a closing, then holes filled.

    Parameters
    ----------
    image : sitk.Image
        Brain image with non-negative intensities.
    method : {"otsu", "li", "triangle", "huang"}
        Histogram threshold; clearing protocols differ in which separates tissue from background.
    log_intensity : bool
        Threshold ``log(1 + intensity)``. Light-sheet tissue brightness spans orders of magnitude, and on a linear
        histogram the threshold splits bright structures from dim tissue rather than tissue from background.
    closing_radius_mm : float
        Radius of the closing that bridges gaps in the thresholded tissue.

    Returns
    -------
    sitk.Image
        UInt8 mask, 1 inside the brain.
    """
    thresholded = sitk.Cast(image, sitk.sitkFloat32)
    if log_intensity:
        thresholded = sitk.Log(sitk.Clamp(thresholded, lowerBound=0.0) + 1.0)
    # Histogram thresholds give insideValue to pixels at or below the threshold.
    tissue = _THRESHOLDS[method](thresholded, 0, 1)
    largest = sitk.RelabelComponent(sitk.ConnectedComponent(tissue), sortByObjectSize=True) == 1
    radius = _radius_voxels(image, closing_radius_mm)
    if any(radius):
        largest = sitk.BinaryMorphologicalClosing(largest, radius)
    mask: sitk.Image = sitk.Cast(sitk.BinaryFillhole(largest), sitk.sitkUInt8)
    return mask


def dilate_mask(mask: sitk.Image, radius_mm: float) -> sitk.Image:
    """Grow a mask by ``radius_mm``.

    ANTs metric masks reject samples rather than extend the domain, so a SyN mask has to reach past the tissue
    boundary for the boundary to be registered.

    Parameters
    ----------
    mask : sitk.Image
        UInt8 mask.
    radius_mm : float
        Dilation radius.

    Returns
    -------
    sitk.Image
        UInt8 mask.
    """
    grown: sitk.Image = sitk.Cast(sitk.BinaryDilate(mask, _radius_voxels(mask, radius_mm)), sitk.sitkUInt8)
    return grown


def n4_correct(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    shrink_factor: int = 2,
    spline_distance_mm: float = 1.5,
    iterations: Sequence[int] = (50, 50, 50, 50),
    convergence_threshold: float = 1e-7,
) -> sitk.Image:
    """Remove a smooth multiplicative bias field estimated inside ``mask``.

    The field is fit on a shrunken copy and evaluated at full size, which costs little accuracy because the
    field is smooth on the scale of ``spline_distance_mm``.

    Parameters
    ----------
    image : sitk.Image
        Brain image with positive tissue intensities; N4 ignores pixels at or below zero.
    mask : sitk.Image
        UInt8 mask, 1 where the field is fit.
    shrink_factor : int
        Downsampling of the fit.
    spline_distance_mm : float
        B-spline control point spacing, which bounds how fast the field can vary.
    iterations : sequence of int
        Iterations per fitting level; the number of entries is the number of levels.
    convergence_threshold : float
        Stop a level when the field changes less than this.

    Returns
    -------
    sitk.Image
        Float32 corrected image.
    """
    full = sitk.Cast(image, sitk.sitkFloat32)
    shrink = [shrink_factor] * full.GetDimension()
    small = sitk.Shrink(full, shrink)
    small_mask = sitk.Shrink(sitk.Cast(mask, sitk.sitkUInt8), shrink)
    extent_mm = [n * s for n, s in zip(full.GetSize(), full.GetSpacing(), strict=True)]
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(i) for i in iterations])
    corrector.SetConvergenceThreshold(convergence_threshold)
    # A cubic B-spline needs mesh size + order control points per axis, and ITK requires at least 4.
    corrector.SetNumberOfControlPoints([max(4, math.ceil(e / spline_distance_mm) + 3) for e in extent_mm])
    corrector.Execute(small, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(full)
    corrected: sitk.Image = sitk.Cast(full / sitk.Exp(log_bias), sitk.sitkFloat32)
    return corrected


def normalize_percentiles(
    image: sitk.Image, mask: sitk.Image, *, low_percentile: float = 0.5, high_percentile: float = 99.5
) -> sitk.Image:
    """Map the in-mask percentile range to [0, 1], clipping both tails.

    The background stays in the image, clipped to 0, rather than being zeroed by the mask.

    Parameters
    ----------
    image : sitk.Image
        Brain image.
    mask : sitk.Image
        UInt8 mask whose voxels set the percentiles.
    low_percentile, high_percentile : float
        Percentiles mapped to 0 and 1.

    Returns
    -------
    sitk.Image
        Float32 image with the input's geometry.
    """
    values = sitk.GetArrayViewFromImage(image)
    inside = values[sitk.GetArrayViewFromImage(mask) > 0]
    low, high = np.percentile(inside, [low_percentile, high_percentile])
    if high <= low:
        raise ValueError(f"In-mask percentiles {low_percentile} and {high_percentile} are both {low}")
    scaled = np.clip((values.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    out = sitk.GetImageFromArray(scaled)
    out.CopyInformation(image)
    return out


def center_of_mass_transform(fixed_mask: sitk.Image, moving_mask: sitk.Image) -> sitk.AffineTransform:
    """Return the translation that maps the fixed mask's centroid onto the moving mask's.

    Like every ANTs and ITK registration transform it maps fixed points to moving points, so it can be given to
    ``antsRegistration`` as the initial moving transform.

    Parameters
    ----------
    fixed_mask, moving_mask : sitk.Image
        UInt8 masks, 1 inside the brain.

    Returns
    -------
    sitk.AffineTransform
        Identity matrix with the centroid offset as translation.
    """
    fixed_center = _centroid(fixed_mask)
    moving_center = _centroid(moving_mask)
    transform = sitk.AffineTransform(3)
    transform.SetTranslation([m - f for m, f in zip(moving_center, fixed_center, strict=True)])
    return transform


def write_transform(transform: sitk.Transform, path: Path) -> Path:
    """Write ``transform`` as an ITK ``.mat`` file that ANTs reads.

    Parameters
    ----------
    transform : sitk.Transform
        Linear transform.
    path : Path
        Output path ending in ``.mat``.

    Returns
    -------
    Path
        ``path``.
    """
    if path.suffix != ".mat":
        raise ValueError(f"{path}: ANTs needs a .mat transform file")
    sitk.WriteTransform(transform, str(path))
    return path


def _centroid(mask: sitk.Image) -> tuple[float, ...]:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask, sitk.sitkUInt8))
    if not stats.HasLabel(1):
        raise ValueError("Mask is empty")
    return tuple(stats.GetCentroid(1))
