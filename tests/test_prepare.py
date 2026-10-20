"""Tests for masking, bias correction, normalization and the initial transform."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from aind_lightsheet_registration.prepare import (
    block_average,
    block_average_mask,
    brain_mask,
    center_of_mass_transform,
    dilate_mask,
    n4_correct,
    normalize_iqr,
    normalize_percentiles,
    set_orientation,
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

    def test_log_intensity_keeps_dim_tissue_beside_a_bright_structure(self):
        """A structure far brighter than the rest of the tissue captures a linear threshold but not a log one."""
        inside = _ellipsoid()
        bright = _ellipsoid(center=(20, 24, 42), radii=(8, 10, 8))
        rng = np.random.default_rng(1)
        values = np.where(bright, 2000.0, np.where(inside, 100.0, 1.0)) * rng.lognormal(0, 0.1, inside.shape)
        image = _image(values)
        linear = sitk.GetArrayFromImage(brain_mask(image, log_intensity=False)).astype(bool)
        logged = sitk.GetArrayFromImage(brain_mask(image)).astype(bool)
        assert linear.sum() < 0.5 * inside.sum()
        assert np.count_nonzero(logged ^ inside) < 0.001 * inside.sum()

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
    corrected = n4_correct(biased, mask, final_spline_spacing_mm=0.2)

    def spread(img):
        values = sitk.GetArrayFromImage(img)[inside]
        return np.std(values) / np.mean(values)

    assert spread(biased) > 0.15
    assert spread(corrected) < 0.05
    assert corrected.GetOrigin() == image.GetOrigin()
    assert corrected.GetSpacing() == image.GetSpacing()


def test_n4_keeps_a_sharp_step_while_removing_a_gradient():
    """A field as fine as the phantom's step would flatten it; 0.2 mm on this 1.4 mm phantom is 1.6 mm on a brain."""
    inside = _ellipsoid()
    z, _, x = np.indices(inside.shape)
    step = np.where(x < 28, 1.0, 1.6)
    values = np.where(inside, 1000.0 * step, 100.0) * np.exp(0.4 * (z - 20) / 14)
    values += np.random.default_rng(0).normal(0, 30, inside.shape)
    mask = sitk.Cast(_image(inside), sitk.sitkUInt8)
    corrected = sitk.GetArrayFromImage(n4_correct(_image(values), mask, final_spline_spacing_mm=0.2))

    def ratio(a, b):
        return np.median(corrected[inside & b]) / np.median(corrected[inside & a])

    assert ratio((x >= 20) & (x < 26), (x >= 30) & (x < 36)) == pytest.approx(1.6, rel=0.02)
    assert ratio((z <= 12) & (x < 26), (z >= 28) & (x < 26)) == pytest.approx(1.0, abs=0.05)


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


def test_normalize_iqr_takes_statistics_from_tissue_without_csf_or_surface():
    """A dim surface shell and a large dark ventricle would pull the median and widen the IQR if they entered."""
    rng = np.random.default_rng(0)
    shape, spacing = (60, 60, 60), 0.025
    inside = _ellipsoid(shape=shape, center=(30, 30, 30), radii=(26, 26, 26))
    core = _ellipsoid(shape=shape, center=(30, 30, 30), radii=(21, 21, 21))
    ventricle = _ellipsoid(shape=shape, center=(30, 30, 30), radii=(9, 9, 9))  # 0.05 mm3, above min_csf_mm3
    values = np.where(inside, 20.0, 0.0)  # the surface shell outside the core stays dim
    values[core] = rng.uniform(80, 120, core.sum())
    values[ventricle] = 2.0
    image = _image(values)
    image.SetSpacing((spacing,) * 3)
    mask = sitk.Cast(_image(inside), sitk.sitkUInt8)
    mask.CopyInformation(image)
    out = sitk.GetArrayFromImage(normalize_iqr(image, mask, erode_mm=5 * spacing))
    # tissue is the core less 5 voxels at each boundary; uniform on [80, 120] has median 100 and IQR 20
    assert np.median(out[core & ~ventricle]) == pytest.approx(1.0, abs=0.02)
    q25, q75 = np.percentile(out[core & ~ventricle], [25, 75])
    assert q75 - q25 == pytest.approx(1.0, abs=0.03)


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


def test_block_average_keeps_block_centers_and_means():
    values = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    image = _image(values)
    image.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, -1))
    out = block_average(image, 2)
    assert out.GetSize() == (4, 3, 2) and out.GetSpacing() == pytest.approx((0.05,) * 3)
    np.testing.assert_allclose(sitk.GetArrayFromImage(out)[0, 0, 0], values[:2, :2, :2].mean())
    center = image.TransformContinuousIndexToPhysicalPoint((0.5, 0.5, 0.5))
    np.testing.assert_allclose(out.TransformIndexToPhysicalPoint((0, 0, 0)), center, atol=1e-9)
    mask = sitk.Cast(_image((values % 8 < 5).astype(np.float32)), sitk.sitkUInt8)
    kept = sitk.GetArrayFromImage(block_average_mask(mask, 2))
    np.testing.assert_array_equal(kept[0, 0], [1, 1, 0, 0])


def test_set_orientation_replaces_directions_about_the_center():
    image = _image(np.zeros((4, 6, 8)))
    image.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    out = set_orientation(image, "lps")
    assert out.GetDirection() == pytest.approx((1, 0, 0, 0, 1, 0, 0, 0, 1))
    center = [(n - 1) / 2 for n in image.GetSize()]
    np.testing.assert_allclose(
        out.TransformContinuousIndexToPhysicalPoint(center), image.TransformContinuousIndexToPhysicalPoint(center)
    )
    assert (
        sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(set_orientation(image, "RAI").GetDirection())
        == "RAI"
    )
