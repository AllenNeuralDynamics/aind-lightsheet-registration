"""Tests for masking, bias correction, normalization and the initial transform."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from aind_lightsheet_registration.prepare import (
    brain_mask,
    center_of_mass_transform,
    dilate_mask,
    n4_correct,
    normalize_percentiles,
    write_transform,
)

SPACING = (0.025, 0.025, 0.025)


def _ellipsoid(center=(20, 24, 28), radii=(14, 18, 22), shape=(40, 48, 56)) -> np.ndarray:
    """Boolean (z, y, x) ellipsoid."""
    grid = np.indices(shape)
    radius2 = np.zeros(shape)
    for g, c, r in zip(grid, center, radii, strict=True):
        radius2 += ((g - c) / r) ** 2
    return radius2 <= 1


def _image(array: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing(SPACING)
    image.SetOrigin((1.0, -2.0, 0.5))
    return image


@pytest.fixture
def brain():
    """Bright noisy ellipsoid on a dim noisy background, with its true extent."""
    inside = _ellipsoid()
    rng = np.random.default_rng(0)
    values = np.where(inside, 1000.0, 100.0) + rng.normal(0, 30, inside.shape)
    return _image(values), inside


class TestBrainMask:
    """Tissue is the bright side of the threshold, whatever the method."""

    @pytest.mark.parametrize("method", ["otsu", "li", "triangle", "huang"])
    def test_recovers_the_tissue(self, brain, method):
        image, inside = brain
        mask = brain_mask(image, method=method)
        assert mask.GetPixelID() == sitk.sitkUInt8
        assert np.count_nonzero(sitk.GetArrayFromImage(mask).astype(bool) ^ inside) < 0.001 * inside.sum()

    def test_keeps_only_the_largest_component_and_fills_holes(self, brain):
        image, inside = brain
        values = sitk.GetArrayFromImage(image)
        values[2:5, 2:5, 2:5] = 1000.0  # detached speck
        values[18:23, 22:27, 26:31] = 100.0  # dark interior pocket
        mask = sitk.GetArrayFromImage(brain_mask(_image(values))).astype(bool)
        assert not mask[2:5, 2:5, 2:5].any()
        assert mask[18:23, 22:27, 26:31].all()
        assert np.count_nonzero(mask ^ inside) < 0.001 * inside.sum()


def test_dilate_mask_grows_by_the_radius():
    mask = sitk.Cast(_image(_ellipsoid()), sitk.sitkUInt8)
    grown = sitk.GetArrayFromImage(dilate_mask(mask, 2 * SPACING[0])).astype(bool)
    edge = 28 + 22  # last tissue voxel along x through the center
    assert grown[20, 24, edge + 2] and not grown[20, 24, edge + 3]
    assert grown[_ellipsoid()].all()


def test_n4_removes_a_smooth_bias(brain):
    image, inside = brain
    z, _, x = np.indices(inside.shape)
    bias = np.exp(0.4 * (x - 28) / 22 + 0.2 * (z - 20) / 14)
    biased = _image(sitk.GetArrayFromImage(image) * bias)
    mask = sitk.Cast(_image(inside), sitk.sitkUInt8)
    corrected = n4_correct(biased, mask, spline_distance_mm=0.5)

    def spread(img):
        values = sitk.GetArrayFromImage(img)[inside]
        return np.std(values) / np.mean(values)

    assert spread(biased) > 0.15
    assert spread(corrected) < 0.05
    assert corrected.GetOrigin() == image.GetOrigin()
    assert corrected.GetSpacing() == image.GetSpacing()


def test_normalize_percentiles_maps_in_mask_range_to_unit_interval(brain):
    image, inside = brain
    mask = sitk.Cast(_image(inside), sitk.sitkUInt8)
    out = normalize_percentiles(image, mask)
    values = sitk.GetArrayFromImage(out)
    tissue = values[inside]
    assert values.min() == 0.0 and values.max() == 1.0
    assert np.mean(tissue == 0.0) == pytest.approx(0.005, abs=0.001)
    assert np.mean(tissue == 1.0) == pytest.approx(0.005, abs=0.001)
    assert np.all(values[~inside] == 0.0)
    assert out.GetOrigin() == image.GetOrigin()


def test_normalize_percentiles_rejects_a_constant_mask_region():
    image = _image(np.ones((8, 8, 8)))
    with pytest.raises(ValueError, match="both"):
        normalize_percentiles(image, sitk.Cast(image, sitk.sitkUInt8))


class TestCenterOfMassTransform:
    """The translation maps fixed points to moving points."""

    def test_maps_fixed_centroid_to_moving_centroid(self):
        fixed = sitk.Cast(_image(_ellipsoid()), sitk.sitkUInt8)
        moving = sitk.Cast(_image(_ellipsoid(center=(17, 26, 33))), sitk.sitkUInt8)
        transform = center_of_mass_transform(fixed, moving)
        fixed_center = fixed.TransformContinuousIndexToPhysicalPoint((28, 24, 20))
        moving_center = moving.TransformContinuousIndexToPhysicalPoint((33, 26, 17))
        np.testing.assert_allclose(transform.TransformPoint(fixed_center), moving_center, atol=1e-9)

    def test_resampling_with_it_overlays_the_masks(self):
        fixed = sitk.Cast(_image(_ellipsoid()), sitk.sitkUInt8)
        moving = sitk.Cast(_image(_ellipsoid(center=(17, 26, 33))), sitk.sitkUInt8)
        transform = center_of_mass_transform(fixed, moving)
        aligned = sitk.Resample(moving, fixed, transform, sitk.sitkNearestNeighbor)
        assert np.array_equal(sitk.GetArrayFromImage(aligned), sitk.GetArrayFromImage(fixed))

    def test_rejects_an_empty_mask(self):
        empty = sitk.Image(8, 8, 8, sitk.sitkUInt8)
        with pytest.raises(ValueError, match="empty"):
            center_of_mass_transform(empty, empty)


def test_write_transform_round_trips(tmp_path):
    transform = sitk.AffineTransform(3)
    transform.SetTranslation((0.1, -0.2, 0.3))
    path = write_transform(transform, tmp_path / "init.mat")
    np.testing.assert_allclose(sitk.ReadTransform(str(path)).TransformPoint((1, 2, 3)), (1.1, 1.8, 3.3))
    with pytest.raises(ValueError, match=r"\.mat"):
        write_transform(transform, tmp_path / "init.txt")


ANTS_BIN = Path(os.environ.get("ANTSPATH", "")) / "antsApplyTransformsToPoints"


@pytest.mark.skipif(not ANTS_BIN.is_file(), reason="needs ANTSPATH with antsApplyTransformsToPoints")
def test_ants_reads_the_initial_transform_in_the_same_direction(tmp_path):
    """ANTs applies the written transform to points as SimpleITK does, fixed to moving."""
    fixed = sitk.Cast(_image(_ellipsoid()), sitk.sitkUInt8)
    moving = sitk.Cast(_image(_ellipsoid(center=(17, 26, 33))), sitk.sitkUInt8)
    transform = center_of_mass_transform(fixed, moving)
    path = write_transform(transform, tmp_path / "init.mat")
    point = fixed.TransformContinuousIndexToPhysicalPoint((28, 24, 20))
    (tmp_path / "in.csv").write_text("x,y,z,t\n" + ",".join(map(str, point)) + ",0\n")
    subprocess.run(
        [ANTS_BIN, "-d", "3", "-i", tmp_path / "in.csv", "-o", tmp_path / "out.csv", "-t", path],
        check=True,
        capture_output=True,
    )
    mapped = np.loadtxt(tmp_path / "out.csv", delimiter=",", skiprows=1)[:3]
    np.testing.assert_allclose(mapped, transform.TransformPoint(point), atol=1e-6)
