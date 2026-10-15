"""Tests for reading a level in the acquisition frame and finishing its downsampling."""

import numpy as np
import pytest
import SimpleITK as sitk

from aind_lightsheet_registration.reader import finish_downsampling, read_level, smooth_to_box

# zarr axes run S→I, A→P, L→R, so SimpleITK's (x, y, z) index axes point R, P, I in LPS.
EXPECTED_DIRECTION = (-1, 0, 0, 0, 1, 0, 0, 0, -1)


class TestReadLevel:
    """Level choice, values and frame of :func:`read_level`."""

    def test_reads_coarsest_level_within_target(self, pyramid, nd_metadata):
        uri, levels = pyramid
        read = read_level(uri, nd_metadata, 0.008)
        assert read.level == 2
        np.testing.assert_array_equal(sitk.GetArrayFromImage(read.image), levels[2])
        np.testing.assert_allclose(read.image.GetSpacing(), (0.0072, 0.0072, 0.008))
        np.testing.assert_allclose(read.image.GetDirection(), EXPECTED_DIRECTION, atol=1e-12)

    def test_levels_share_one_frame(self, pyramid, nd_metadata):
        """A level voxel's center is the center of the base voxels it averages."""
        uri, _ = pyramid
        base = read_level(uri, nd_metadata, 0.002).image
        coarse = read_level(uri, nd_metadata, 0.008).image
        assert base.GetOrigin() == (0.0, 0.0, 0.0)
        for index in [(0, 0, 0), (2, 1, 3)]:
            base_index = [i * 4 + 1.5 for i in index]
            np.testing.assert_allclose(
                coarse.TransformIndexToPhysicalPoint(index),
                base.TransformContinuousIndexToPhysicalPoint(base_index),
                atol=1e-12,
            )


class TestFinishDownsampling:
    """Smoothing and resampling in :func:`finish_downsampling`."""

    @staticmethod
    def _impulse(spacing, size=(41, 41, 41)):
        image = sitk.Image(list(size), sitk.sitkFloat32)
        image.SetSpacing(spacing)
        image.SetDirection(EXPECTED_DIRECTION)
        image[[n // 2 for n in size]] = 1.0
        return image

    def test_smoothing_adds_the_missing_box_variance(self):
        """Level variance s²/12 plus the added σ² reaches the 25 µm box's 25²/12, though σ is a third of a voxel."""
        spacing = (0.0144, 0.0144, 0.016)
        smoothed = smooth_to_box(self._impulse(spacing), 0.025)
        array = sitk.GetArrayFromImage(smoothed)
        for profile, step in [(array[:, 20, 20], spacing[2]), (array[20, 20, :], spacing[0])]:
            x = (np.arange(41) - 20) * step
            added = np.sum(profile * x**2) / np.sum(profile)
            assert added + step**2 / 12 == pytest.approx(0.025**2 / 12, rel=1e-3)

    def test_keeps_corner_and_direction(self):
        image = self._impulse((0.0144, 0.0144, 0.016), size=(50, 60, 40))
        out = finish_downsampling(image, 0.025)
        assert out.GetSpacing() == (0.025, 0.025, 0.025)
        assert out.GetSize() == (29, 35, 26)
        assert out.GetDirection() == image.GetDirection()
        np.testing.assert_allclose(
            out.TransformContinuousIndexToPhysicalPoint((-0.5, -0.5, -0.5)),
            image.TransformContinuousIndexToPhysicalPoint((-0.5, -0.5, -0.5)),
            atol=1e-12,
        )

    def test_preserves_a_linear_ramp(self):
        """Smoothing and linear resampling leave a ramp unchanged away from the edges."""
        ramp = np.broadcast_to(np.arange(60, dtype=np.float32)[np.newaxis, np.newaxis, :], (60, 60, 60))
        image = sitk.GetImageFromArray(np.ascontiguousarray(ramp))
        image.SetSpacing((0.0144, 0.0144, 0.016))
        out = finish_downsampling(image, 0.025)
        for index in [(10, 10, 10), (20, 15, 12)]:
            point = out.TransformIndexToPhysicalPoint(index)
            expected = image.TransformPhysicalPointToContinuousIndex(point)[0]
            assert out[index] == pytest.approx(expected, abs=1e-3)
