"""Tests for catching wrong axis directions in a brain's metadata."""

import numpy as np
import pytest
import SimpleITK as sitk

from aind_lightsheet_registration.orientation import (
    axis_rotations,
    check_orientation,
    orientation_code,
    with_direction,
)
from tests.test_register import _ellipsoid

# a phantom shares more of its shape with its rotations than a brain does, so it separates them less
PHANTOM_RATIO = 1.3


def _phantom(origin=(0.0, 0.0, 0.0)) -> tuple[sitk.Image, sitk.Image]:
    """An ellipsoid with unequal axes, a bright and a dark inclusion, so no rotation maps it onto itself."""
    shape = (64, 64, 64)
    tissue = _ellipsoid(shape, (32, 32, 32), (18, 24, 30))
    values = np.where(tissue, 0.4, 0.0)
    values[_ellipsoid(shape, (26, 40, 20), (6, 6, 6))] = 1.0
    values[_ellipsoid(shape, (38, 28, 44), (4, 8, 5))] = 0.1
    image = sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(values.astype(np.float32)), 1.0)
    mask = sitk.GetImageFromArray(tissue.astype(np.uint8))
    for img in (image, mask):
        img.SetSpacing((0.1,) * 3)
        img.SetOrigin(origin)
    return image, mask


def test_axis_rotations_are_the_24_proper_rotations_identity_first():
    rotations = axis_rotations()
    assert len({m.tobytes() for m in rotations}) == 24
    assert np.array_equal(rotations[0], np.eye(3))
    assert all(np.isclose(np.linalg.det(m), 1) for m in rotations)


def test_correct_metadata_passes_and_its_score_comes_first():
    template, template_mask = _phantom()
    brain, brain_mask = _phantom((0.7, -0.3, 0.2))
    scores = check_orientation(brain, brain_mask, template, template_mask, min_ratio=PHANTOM_RATIO)
    assert scores[0].orientation == "LPS"
    assert scores[0].mutual_information > PHANTOM_RATIO * max(s.mutual_information for s in scores[1:])


def test_a_rotated_header_is_refused_naming_the_orientation_that_fits():
    template, template_mask = _phantom()
    wrong = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)  # 180 degrees about the third axis
    brain, brain_mask = (with_direction(img, wrong) for img in _phantom((0.7, -0.3, 0.2)))
    assert orientation_code(brain) == "RAS"
    with pytest.raises(ValueError, match="orientation='LPS'"):
        check_orientation(brain, brain_mask, template, template_mask, min_ratio=PHANTOM_RATIO)


def test_an_unexpected_orientation_warns():
    template, template_mask = _phantom()
    with pytest.warns(UserWarning, match="expected RPI"):
        check_orientation(template, template_mask, template, template_mask, expected="rpi")


def test_with_direction_keeps_the_center():
    image = sitk.Image(10, 12, 14, sitk.sitkFloat32)
    image.SetSpacing((0.1, 0.2, 0.3))
    flipped = with_direction(image, (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
    center = [(n - 1) / 2 for n in image.GetSize()]
    np.testing.assert_allclose(
        flipped.TransformContinuousIndexToPhysicalPoint(center), image.TransformContinuousIndexToPhysicalPoint(center)
    )
