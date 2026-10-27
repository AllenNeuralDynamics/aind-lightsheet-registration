"""Mask, bias-correct and normalize a brain image for registration.

Every function takes and returns SimpleITK images in mm and keeps their geometry, so the result stays in the
acquisition frame.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import SimpleITK as sitk

ThresholdMethod = Literal["otsu", "li", "triangle", "huang"]
Normalization = Literal["iqr", "percentile"]

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


def signed_distance_map(mask: sitk.Image, scale_mm: float = 0.3) -> sitk.Image:
    """``tanh(d / scale_mm)`` of the signed distance ``d`` to a mask's boundary, negative inside.

    Matching two of these by mean squares pulls a structure toward its counterpart from up to a few ``scale_mm``
    away; a binary mask gives no gradient off its boundary, and an unsaturated distance lets far voxels dominate.

    Parameters
    ----------
    mask : sitk.Image
        UInt8 mask.
    scale_mm : float
        Distance over which the map saturates.

    Returns
    -------
    sitk.Image
        Float32 map in (-1, 1) on the mask's grid.
    """
    distance = sitk.SignedMaurerDistanceMap(
        sitk.Cast(mask > 0, sitk.sitkUInt8), insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )
    mapped = sitk.GetImageFromArray(np.tanh(sitk.GetArrayViewFromImage(distance) / scale_mm).astype(np.float32))
    mapped.CopyInformation(mask)
    return mapped


def block_average(image: sitk.Image, factor: int) -> sitk.Image:
    """Average ``factor``-cubed blocks, keeping the physical extent; voxels past the last whole block are dropped.

    Parameters
    ----------
    image : sitk.Image
        Image to downsample.
    factor : int
        Block edge in voxels; 1 returns a float32 copy.

    Returns
    -------
    sitk.Image
        Float32 image whose voxel centers are the block centers.
    """
    floated: sitk.Image = sitk.Cast(image, sitk.sitkFloat32)
    if factor == 1:
        return floated
    shrunk: sitk.Image = sitk.BinShrink(floated, [factor] * image.GetDimension())
    return shrunk


def block_average_mask(mask: sitk.Image, factor: int) -> sitk.Image:
    """Downsample a mask by :func:`block_average`, keeping blocks that are mostly inside.

    Parameters
    ----------
    mask : sitk.Image
        UInt8 mask.
    factor : int
        Block edge in voxels.

    Returns
    -------
    sitk.Image
        UInt8 mask.
    """
    kept: sitk.Image = sitk.Cast(block_average(mask, factor) > 0.5, sitk.sitkUInt8)
    return kept


def set_orientation(image: sitk.Image, orientation: str) -> sitk.Image:
    """Replace the axis directions with an orientation code, keeping the volume's physical center.

    For brains whose acquisition metadata names the wrong axis directions. The code is SimpleITK's: each letter is
    where its index axis points, so the identity direction is "LPS".

    Parameters
    ----------
    image : sitk.Image
        Image whose direction is wrong.
    orientation : str
        Three-letter code, e.g. "LPS" or "RAI".

    Returns
    -------
    sitk.Image
        The same voxels with the new direction and an origin that leaves the center in place.
    """
    direction = sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(orientation.upper())
    center_index = [(n - 1) / 2 for n in image.GetSize()]
    center = np.array(image.TransformContinuousIndexToPhysicalPoint(center_index))
    out = sitk.Image(image)
    out.SetDirection(direction)
    offset = np.array(out.TransformContinuousIndexToPhysicalPoint(center_index)) - np.array(out.GetOrigin())
    out.SetOrigin(tuple(center - offset))
    return out


def n4_correct(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    shrink_factor: int = 2,
    final_spline_spacing_mm: float = 1.6,
    iterations: Sequence[int] = (50, 50, 50, 50),
    convergence_threshold: float = 1e-7,
) -> sitk.Image:
    """Remove a smooth multiplicative bias field estimated inside ``mask``.

    The field is fit on a shrunken copy and evaluated at full size, which costs little accuracy because the
    field is smooth on the scale of ``final_spline_spacing_mm``. Each fitting level doubles the B-spline mesh, so the
    first level's mesh is the final spacing times ``2 ** (levels - 1)``, at least one span per axis. The defaults
    give one starting span on a mouse brain and about 1.6 mm at the end, as ``aind-ccf-registration`` does; a finer
    field follows anatomy.

    Parameters
    ----------
    image : sitk.Image
        Brain image with positive tissue intensities; N4 ignores pixels at or below zero.
    mask : sitk.Image
        UInt8 mask, 1 where the field is fit.
    shrink_factor : int
        Downsampling of the fit.
    final_spline_spacing_mm : float
        B-spline span after the last level, which bounds how fast the field can vary. Achieved only approximately,
        since spans per axis are whole and grow in powers of 2.
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
    first_spacing_mm = final_spline_spacing_mm * 2 ** (len(iterations) - 1)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(i) for i in iterations])
    corrector.SetConvergenceThreshold(convergence_threshold)
    # A cubic B-spline has spans + 3 control points per axis.
    corrector.SetNumberOfControlPoints([max(1, round(e / first_spacing_mm)) + 3 for e in extent_mm])
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


def normalize_iqr(
    image: sitk.Image, mask: sitk.Image, *, erode_mm: float = 0.1, min_csf_mm3: float = 0.002
) -> sitk.Image:
    """Map intensities to ``(x - median) / IQR + 1``, with the median and IQR taken over tissue alone.

    Tissue is ``mask`` without CSF, eroded by ``erode_mm`` so partial-volume voxels at the surface and around the
    ventricles do not enter. CSF is the dark regions (at or below a log-Otsu threshold) larger than
    ``min_csf_mm3``; smaller dark specks are noise, and eroding around each would riddle the tissue. Nothing is
    clipped: a template in these units sets the clip range.

    Parameters
    ----------
    image : sitk.Image
        Bias-corrected brain image.
    mask : sitk.Image
        UInt8 hole-filled brain mask.
    erode_mm : float
        Erosion of the tissue before the statistics.
    min_csf_mm3 : float
        Smallest dark region treated as CSF.

    Returns
    -------
    sitk.Image
        Float32 image with the input's geometry.
    """
    floated = sitk.Cast(image, sitk.sitkFloat32)
    log = sitk.Log(sitk.Clamp(floated, lowerBound=0.0) + 1.0)
    inside = sitk.Cast(mask, sitk.sitkUInt8) > 0
    dark = sitk.And(inside, sitk.OtsuThreshold(log, 1, 0))
    voxel_mm3 = float(np.prod(image.GetSpacing()))
    csf = sitk.RelabelComponent(sitk.ConnectedComponent(dark), minimumObjectSize=int(min_csf_mm3 / voxel_mm3) + 1) > 0
    tissue = sitk.And(inside, sitk.Not(csf))
    # exact Euclidean erosion; ITK's ball kernel of the same radius is about 1.5 times larger in volume
    to_background = sitk.SignedMaurerDistanceMap(
        sitk.Not(tissue), insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )
    tissue = sitk.And(tissue, to_background > erode_mm * (1 + 1e-6))
    values = sitk.GetArrayViewFromImage(floated)
    selected = values[sitk.GetArrayViewFromImage(tissue) > 0]
    if selected.size == 0:
        raise ValueError("no tissue left after removing CSF and eroding")
    q25, q50, q75 = np.percentile(selected, [25, 50, 75])
    if q75 <= q25:
        raise ValueError(f"tissue interquartile range is empty at {q50}")
    out = sitk.GetImageFromArray(((values - q50) / (q75 - q25) + 1).astype(np.float32))
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
