"""Catch acquisition metadata that names the wrong axis directions, before a registration converges on it.

A brain whose header is rotated by a multiple of 90° registers to a template as a deformed, wrong brain without any
error. Fitting the brain rigidly under each of the 24 axis rotations and comparing their mutual information exposes
it. A left-right mirror cannot be caught this way, since a mirrored brain matches its template as well as the brain
does; only a declared expected orientation can catch that.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

CHECK_SPACING_MM = 0.2


@dataclass(frozen=True)
class OrientationScore:
    """Fit of the brain to the template with its directions replaced by ``orientation``.

    Attributes
    ----------
    orientation : str
        SimpleITK axis code of the brain's directions for this fit.
    mutual_information : float
        Mattes mutual information over the template's tissue after the rigid fit; higher is better.
    """

    orientation: str
    mutual_information: float


def orientation_code(image: sitk.Image) -> str:
    """SimpleITK axis code of ``image``'s directions, e.g. "LPS" for the identity."""
    code: str = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())
    return code


def with_direction(image: sitk.Image, direction: tuple[float, ...]) -> sitk.Image:
    """Replace ``image``'s directions, choosing the origin that keeps its physical center in place."""
    center_index = [(n - 1) / 2 for n in image.GetSize()]
    center = np.array(image.TransformContinuousIndexToPhysicalPoint(center_index))
    out = sitk.Image(image)
    out.SetDirection(direction)
    offset = np.array(out.TransformContinuousIndexToPhysicalPoint(center_index)) - np.array(out.GetOrigin())
    out.SetOrigin(tuple(center - offset))
    return out


def axis_rotations() -> list[np.ndarray]:
    """Return the 24 signed permutation matrices with determinant +1, identity first."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            m[range(3), perm] = signs
            if np.linalg.det(m) > 0:
                out.append(m)
    out.sort(key=lambda m: not np.array_equal(m, np.eye(3)))
    return out


def score_orientations(
    brain: sitk.Image,
    brain_mask: sitk.Image,
    template: sitk.Image,
    template_mask: sitk.Image,
    *,
    spacing_mm: float = CHECK_SPACING_MM,
) -> list[OrientationScore]:
    """Rigidly fit the brain to the template under each axis rotation of its directions, metadata's first.

    Parameters
    ----------
    brain, brain_mask : sitk.Image
        Brain image and its UInt8 tissue mask, on one grid.
    template, template_mask : sitk.Image
        Template image and its UInt8 tissue mask, on one grid.
    spacing_mm : float
        Spacing both are block-averaged to first; coarse enough for 24 fits to take seconds.

    Returns
    -------
    list of OrientationScore
        One per rotation; the first is the brain's own directions.
    """
    fixed, fixed_mask = _coarse(template, spacing_mm), _coarse_mask(template_mask, spacing_mm)
    moving, moving_mask = _coarse(brain, spacing_mm), _coarse_mask(brain_mask, spacing_mm)
    direction = np.reshape(moving.GetDirection(), (3, 3))
    scores = []
    for rotation in axis_rotations():
        rotated = tuple((rotation @ direction).ravel())
        image, mask = with_direction(moving, rotated), with_direction(moving_mask, rotated)
        scores.append(OrientationScore(orientation_code(image), _rigid_fit(fixed, fixed_mask, image, mask)))
    return scores


def check_orientation(
    brain: sitk.Image,
    brain_mask: sitk.Image,
    template: sitk.Image,
    template_mask: sitk.Image,
    *,
    expected: str | None = None,
    min_ratio: float = 2.0,
) -> list[OrientationScore]:
    """Raise if another axis orientation fits the template clearly better than the brain's metadata.

    Parameters
    ----------
    brain, brain_mask, template, template_mask : sitk.Image
        As for :func:`score_orientations`.
    expected : str or None
        Orientation code the acquisition should have; a mismatch warns, catching the mirror the fits cannot.
    min_ratio : float
        How many times the metadata's mutual information a rotation must reach to be called the right one. Correct
        metadata fits several times better than any rotation, and a rotated header several times worse.

    Returns
    -------
    list of OrientationScore
        The scores, metadata's first.
    """
    own = orientation_code(brain)
    if expected is not None and expected.upper() != own:
        warnings.warn(f"brain directions are {own}, expected {expected.upper()}", stacklevel=2)
    scores = score_orientations(brain, brain_mask, template, template_mask)
    best = max(scores[1:], key=lambda s: s.mutual_information)
    if best.mutual_information > scores[0].mutual_information * min_ratio:
        raise ValueError(
            f"brain directions {own} fit the template with mutual information {scores[0].mutual_information:.3f}, "
            f"{best.orientation} with {best.mutual_information:.3f}; if the metadata is wrong, pass "
            f"orientation={best.orientation!r}"
        )
    return scores


def _coarse(image: sitk.Image, spacing_mm: float) -> sitk.Image:
    factor = max(1, round(spacing_mm / image.GetSpacing()[0]))
    shrunk: sitk.Image = sitk.BinShrink(sitk.Cast(image, sitk.sitkFloat32), [factor] * 3)
    return shrunk


def _coarse_mask(mask: sitk.Image, spacing_mm: float) -> sitk.Image:
    kept: sitk.Image = sitk.Cast(_coarse(mask, spacing_mm) > 0.5, sitk.sitkUInt8)
    return kept


def _rigid_fit(fixed: sitk.Image, fixed_mask: sitk.Image, moving: sitk.Image, moving_mask: sitk.Image) -> float:
    initial = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_mask, sitk.sitkFloat32),
        sitk.Cast(moving_mask, sitk.sitkFloat32),
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricFixedMask(fixed_mask)
    reg.SetMetricSamplingStrategy(reg.REGULAR)
    reg.SetMetricSamplingPercentage(0.25, 1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-3, 200, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1, 0])
    reg.SetInitialTransform(initial, inPlace=False)
    fitted = reg.Execute(fixed, moving)
    # every template tissue voxel, so all orientations are scored on the same samples
    final = sitk.ImageRegistrationMethod()
    final.SetMetricAsMattesMutualInformation(32)
    final.SetMetricFixedMask(fixed_mask)
    final.SetInterpolator(sitk.sitkLinear)
    final.SetInitialTransform(fitted)
    return -float(final.MetricEvaluate(fixed, moving))
