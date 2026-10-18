"""Tests for the calibrated tile-row seam profile."""

import numpy as np
from scipy.ndimage import gaussian_filter

from aind_lightsheet_registration.seams import WINDOW_MM, SeamProfile, estimate_seam_profile, seam_windows

SPACING, STEP_MM = 0.025, 2.592


def _ripple(d_mm: np.ndarray) -> np.ndarray:
    return -0.05 * np.exp(-(((d_mm - 0.15) / 0.08) ** 2)) + 0.04 * np.exp(-(((d_mm - 0.5) / 0.1) ** 2))


def test_profile_pooled_over_brains_recovers_the_ripple_and_not_the_anatomy():
    rng = np.random.default_rng(0)
    truth = SeamProfile(np.arange(-1, 2, 0.005), _ripple(np.arange(-1, 2, 0.005)))
    starts = [r * STEP_MM for r in range(1, 6)]
    shape = (20, round(15 / SPACING), 40)
    y_mm = np.arange(shape[1]) * SPACING
    windows = []
    for _ in range(6):
        anatomy = np.exp(gaussian_filter(rng.normal(size=shape), (0, 3, 3)))  # log std about 0.1
        brain = 1000 * anatomy * np.exp(truth.along(y_mm, np.array(starts)))[None, :, None]
        windows += seam_windows(brain, np.ones(shape, bool), SPACING, starts)
    est = estimate_seam_profile(windows, SPACING)
    inner = (est.distance_mm > WINDOW_MM[0] + 0.2) & (est.distance_mm < WINDOW_MM[1] - 0.2)
    expected = _ripple(est.distance_mm[inner])
    assert np.abs(est.log_gain[inner] - expected).max() < 0.012
    assert est.log_gain[0] == 0 and est.log_gain[-1] == 0


def test_profile_acts_only_within_its_window_of_each_row():
    p = SeamProfile(np.array([-0.5, 0.0, 0.5]), np.array([0.0, -0.1, 0.0]))
    y = np.array([0.4, 1.0, 1.6, 2.0, 3.0])
    out = p.along(y, np.array([1.0, 3.0]))
    np.testing.assert_allclose(out, [0.0, -0.1, 0.0, 0.0, -0.1])
