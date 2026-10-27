"""Tests for moving images between the pipeline and Zarr frames."""

import numpy as np
import pytest
import SimpleITK as sitk

from aind_lightsheet_registration.frames import move_image


def _image() -> sitk.Image:
    image = sitk.GetImageFromArray(np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6))
    image.SetSpacing((0.025, 0.025, 0.025))
    image.SetOrigin((-1.5, -1.5, 1.5))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, -1))
    return image


def test_move_image_carries_every_voxel_to_its_mapped_point():
    image = _image()
    transform = sitk.AffineTransform(3)
    transform.SetMatrix((-1, 0, 0, 0, 1, 0, 0, 0, 1))  # an axis flip as well as a shift
    transform.SetTranslation((10.0, 1.5, -1.5))
    moved = move_image(image, transform)
    for index in [(0, 0, 0), (5, 4, 3), (2, 1, 3)]:
        expected = transform.TransformPoint(image.TransformIndexToPhysicalPoint(index))
        assert moved.TransformIndexToPhysicalPoint(index) == pytest.approx(expected)
    assert sitk.GetArrayViewFromImage(moved).tolist() == sitk.GetArrayViewFromImage(image).tolist()


def test_a_shearing_map_is_refused():
    transform = sitk.AffineTransform(3)
    transform.SetMatrix((1, 0.2, 0, 0, 1, 0, 0, 0, 1))
    with pytest.raises(ValueError, match="shears"):
        move_image(_image(), transform)
