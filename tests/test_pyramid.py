"""Tests for level choice and the smoothing that finishes the downsampling."""

import math

import pytest

from aind_lightsheet_registration.pyramid import box_sigma, choose_level, extra_sigma, level_center_offset

# SmartSPIM fused levels in SimpleITK (x, y, z) order, mm.
SMARTSPIM = [(0.0018 * 2**k, 0.0018 * 2**k, 0.002 * 2**k) for k in range(4)]


def test_box_sigma_matches_uniform_variance():
    assert box_sigma(math.sqrt(12)) == pytest.approx(1.0)


def test_extra_sigma_from_coarsest_smartspim_level_to_25um():
    sx, sy, sz = (s * 1000 for s in extra_sigma(SMARTSPIM[3], 0.025))
    assert sz == pytest.approx(5.55, abs=0.01)
    assert sx == sy == pytest.approx(5.90, abs=0.01)


def test_extra_sigma_is_zero_at_the_target():
    assert extra_sigma((0.025, 0.025, 0.025), 0.025) == (0.0, 0.0, 0.0)


def test_extra_sigma_rejects_a_coarser_level():
    with pytest.raises(ValueError, match="coarser"):
        extra_sigma((0.03, 0.02, 0.02), 0.025)


@pytest.mark.parametrize(("target", "level"), [(0.025, 3), (0.010, 2), (0.002, 0), (0.0036, 0)])
def test_choose_level_takes_coarsest_level_within_target(target, level):
    assert choose_level(SMARTSPIM, target) == level


def test_choose_level_rejects_target_finer_than_full_resolution():
    with pytest.raises(ValueError, match="No level"):
        choose_level(SMARTSPIM, 0.001)


def test_level_center_offset_is_half_a_block_less_half_a_voxel():
    assert level_center_offset(SMARTSPIM[0], SMARTSPIM[3]) == pytest.approx((3.5, 3.5, 3.5))
    assert level_center_offset(SMARTSPIM[0], SMARTSPIM[0]) == (0.0, 0.0, 0.0)
