"""Tests for reading the focus-ramp planes and fitting their gain on synthetic tiles."""

import struct

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

from aind_lightsheet_registration.focus import (
    FocusRamp,
    estimate_focus_ramp,
    plane_depths_um,
    ramp_depths_um,
    read_focus_ramp,
)
from aind_lightsheet_registration.fuse import fuse_mosaic
from aind_lightsheet_registration.tiles import TileLayout

from .test_tiles import _cut, _metadata, _scene


def _flattened(z: list[float]) -> str:
    """A LabVIEW-flattened 2 x n array of (stage z, focus), every byte hex-escaped."""
    raw = struct.pack(">ii", 2, len(z)) + struct.pack(f">{2 * len(z)}d", *z, *[100.0] * len(z))
    return "".join(f"\\{b:02X}" for b in raw)


def _tile_settings(z: list[float]) -> str:
    return (
        "[Stage]\nX = 1\n[Focus Compensation]\n"
        f'Focus.0 = "{_flattened([])}"\nFocus.1 = "{_flattened([9.0, 8.0])}"\nFocus.2 = "{_flattened(z)}"\n'
        "[Other]\nFocus.2 = nothing\n"
    )


def test_ramp_planes_are_read_for_the_channel_and_taken_below_the_stack_start():
    meta = _metadata()
    meta["wavelength_config"] = {"445": {}, "488": {}, "639": {}}
    for t in meta["tile_config"].values():
        t["Z"] = "1800"
    text = _tile_settings([2300.5, 1800.0, 2800.0])
    assert read_focus_ramp(text, "445", ["445", "488", "639"]).size == 0
    np.testing.assert_array_equal(ramp_depths_um(text, meta, "639"), [0.0, 500.5, 1000.0])
    meta["tile_config"]["0"]["Z"] = "1700"
    meta["tile_config"]["1"]["Z"] = "1700"  # the 639 nm entry of the first tile
    with pytest.raises(ValueError, match="start at different depths"):
        ramp_depths_um(text, meta, "639")


def _dip(past_um: np.ndarray, period_um: float) -> np.ndarray:
    """Log gain sloping down into each ramp plane and recovering just after it."""
    return -0.06 * np.exp(-(period_um - past_um) / 15) - 0.02 * np.exp(-past_um / 6)


def test_ramp_gain_is_recovered_from_tiles_with_different_anatomy():
    nz = 200
    layout = TileLayout.from_metadata(_metadata(3, 3), level=1, shape=(nz, 60, 60))
    depth = plane_depths_um(layout)
    assert depth[:2] == pytest.approx([1.0, 5.0])  # pairs of 2 um planes, from the first plane's center
    period, ramp = 100.0, np.arange(30.0, depth[-1], 100.0)
    truth = FocusRamp(ramp, 4.0, _dip(np.arange(25) * 4.0 + 2.0, period))
    gain = np.exp(truth.along(depth))[:, None, None]
    rng = np.random.default_rng(0)
    tiles = {}
    for t in layout.tiles:  # each tile its own texture and depth trend
        texture = gaussian_filter(rng.normal(size=(nz, 60, 60)), 1.5)
        trend = 0.3 * np.sin(depth / 400 + t.row)[:, None, None]
        tiles[t.name] = (np.exp(texture / texture.std() * 0.2 + trend) * 300 * gain).astype(np.float32)
    fit = estimate_focus_ramp(layout, lambda t: tiles[t.name], ramp, highpass_um=200.0)
    assert fit.bin_um == 4.0 and fit.period_um == period
    expected = truth.log_gain - np.median(truth.log_gain)
    # the running-median high-pass spans a period and a bit, so it ripples with phase by a fraction of the dip
    np.testing.assert_allclose(fit.log_gain, expected, atol=0.01)
    assert np.argmin(fit.log_gain) == np.argmin(expected)
    inner = (depth > ramp[0] + 100) & (depth < ramp[-1])
    np.testing.assert_allclose(
        fit.along(depth)[inner], truth.along(depth)[inner] - np.median(truth.log_gain), atol=0.01
    )


def test_ramp_gain_is_zero_away_from_the_ramp(tmp_path):
    ramp = FocusRamp(np.array([100.0, 200.0, 300.0]), 10.0, np.linspace(-0.1, 0.0, 10))
    g = ramp.along(np.array([20.0, 55.0, 105.0, 395.0, 405.0]))
    np.testing.assert_allclose(g, [0.0, ramp.log_gain[5], ramp.log_gain[0], ramp.log_gain[9], 0.0])
    ramp.save(tmp_path / "f.json")
    back = FocusRamp.load(tmp_path / "f.json")
    np.testing.assert_array_equal(back.log_gain, ramp.log_gain)
    assert back.bin_um == ramp.bin_um


def test_fusion_divides_out_the_plane_gain():
    layout = TileLayout.from_metadata(_metadata(), level=1, shape=(30, 60, 70))
    scene = _scene(layout)
    gain = 0.05 * np.sin(np.arange(30) / 3)
    tiles = {
        k: v * np.exp(gain)[:, None, None].astype(np.float32)
        for k, v in _cut(layout, scene, {t.name: (0, 0, 0) for t in layout.tiles}).items()
    }
    fused = sitk.GetArrayFromImage(fuse_mosaic(layout, lambda t: tiles[t.name], plane_log_gain=gain))
    nz, my, mx = layout.mosaic_shape
    np.testing.assert_allclose(fused, scene[5 : 5 + nz, 5 : 5 + my, 5 : 5 + mx], rtol=1e-4)
