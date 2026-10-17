"""Tests for the tile layout, stitching, fusion and the nominal-to-stitched field on synthetic mosaics."""

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as nd_shift

from aind_lightsheet_registration.fuse import fuse_mosaic, place_on_reference
from aind_lightsheet_registration.stitch import (
    measure_pairs,
    nominal_to_stitched_field,
    solve_offsets,
    solve_residuals,
)
from aind_lightsheet_registration.tiles import TileLayout, choose_tile_level

SHAPE = (30, 60, 70)  # z, y, x at level 1
STEP = (48, 56)  # y, x
PIXEL_UM, Z_UM = 1.8, 2.0


def _metadata(n_rows=2, n_cols=2):
    """A SmartSPIM metadata.json for a raster grid, stage units 0.1 µm, column 1 first in every row."""
    tiles, k = {}, 0
    for r in range(n_rows):
        for c in range(n_cols):
            for laser in ("488", "639"):
                tiles[str(k)] = {
                    "X": str(round(400000 + c * STEP[1] * 2 * PIXEL_UM * 10)),
                    "Y": str(round(300000 + r * STEP[0] * 2 * PIXEL_UM * 10)),
                    "Laser": laser,
                    "Side": str(0 if c < n_cols / 2 else 1),
                }
                k += 1
    return {
        "session_config": {"um/pix": str(PIXEL_UM), "Z step (um)": str(Z_UM)},
        "wavelength_config": {
            "488": {"power_left": "50", "power_right": "60"},
            "639": {"power_left": "90.00", "power_right": "100.00"},
        },
        "tile_config": tiles,
    }


@pytest.fixture
def layout():
    return TileLayout.from_metadata(_metadata(), level=1, shape=SHAPE)


def test_layout_reads_grid_from_metadata(layout):
    assert layout.voxel_um == (4.0, 3.6, 3.6)
    assert layout.step_px == STEP
    assert layout.overlap_px == (12, 14)
    assert (layout.n_rows, layout.n_cols) == (2, 2)
    assert layout.mosaic_shape == (30, 108, 126)
    assert [(t.row, t.col, t.side) for t in layout.tiles] == [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]
    assert layout.relative_power[("639", 1)] == pytest.approx(100 / 90)
    assert [k for k, _, _ in layout.neighbor_pairs()] == ["AP", "LR", "AP", "LR"]
    assert TileLayout.from_json(layout.to_json()) == layout


def test_tile_level_is_coarsest_no_coarser_than_target():
    assert choose_tile_level(_metadata(), 25.0) == 3  # 14.4 um
    assert choose_tile_level({"session_config": {"um/pix": "4.0"}}, 25.0) == 2  # 16 um


def _scene(layout, seed=0):
    rng = np.random.default_rng(seed)
    nz, my, mx = layout.mosaic_shape
    texture = gaussian_filter(rng.normal(size=(nz + 10, my + 10, mx + 10)), 1.5)
    return np.exp(2 * texture / texture.std()) * 200 + 50


def _cut(layout, scene, offsets):
    """Tiles as a stage error records them: tile t shows the scene moved by ``offsets[t]``."""
    nz, ny, nx = layout.shape
    out = {}
    for t in layout.tiles:
        y0, x0 = t.row * layout.step_px[0] + 5, t.col * layout.step_px[1] + 5
        moved = nd_shift(scene, offsets[t.name], order=1, mode="nearest")
        out[t.name] = moved[5 : 5 + nz, y0 : y0 + ny, x0 : x0 + nx].astype(np.float32)
    return out


def test_stitching_recovers_tile_offsets(layout):
    true = dict(zip([t.name for t in layout.tiles], [(0, 0, 0), (1, -1, 1), (-1, 1, 0), (0, 1, -1)], strict=True))
    tiles = _cut(layout, _scene(layout), true)
    pairs = measure_pairs(layout, lambda t: tiles[t.name])
    assert len(pairs) == 4 and not any(p.at_edge for p in pairs)
    offsets = solve_offsets(layout, pairs)
    # fusing undoes each move with the opposite offset; offsets are median-centered
    expected = -np.array(list(true.values()), float)
    expected -= np.median(expected, axis=0)
    np.testing.assert_allclose(np.array(list(offsets.values())), expected, atol=0.25)
    assert max(solve_residuals(offsets, pairs)) < 0.25


def test_unplaceable_tile_is_an_error(layout):
    tiles = _cut(layout, _scene(layout), {t.name: (0, 0, 0) for t in layout.tiles})
    pairs = [p for p in measure_pairs(layout, lambda t: tiles[t.name]) if layout.tiles[3] not in (p.earlier, p.later)]
    with pytest.raises(ValueError, match="no reliable overlap"):
        solve_offsets(layout, pairs)


def test_nominal_fusion_with_gains_reproduces_the_scene(layout):
    scene = _scene(layout)
    tiles = _cut(layout, scene, {t.name: (0, 0, 0) for t in layout.tiles})
    _, ny, nx = layout.shape
    yy, xx = np.mgrid[:ny, :nx]
    gains = {t.name: 0.3 * np.sin(yy / 17 + t.col) + 0.002 * xx * (t.row + 1) for t in layout.tiles}
    lit = {k: v * np.exp(gains[k]).astype(np.float32) for k, v in tiles.items()}
    fused = fuse_mosaic(layout, lambda t: lit[t.name], log_gain=lambda t: gains[t.name])
    nz, my, mx = layout.mosaic_shape
    np.testing.assert_allclose(sitk.GetArrayFromImage(fused), scene[5 : 5 + nz, 5 : 5 + my, 5 : 5 + mx], rtol=1e-4)
    assert fused.GetSpacing() == pytest.approx((0.0036, 0.0036, 0.004))


def test_placement_finds_the_shift():
    rng = np.random.default_rng(1)
    ref = np.exp(gaussian_filter(rng.normal(size=(40, 50, 60)), 2) * 8).astype(np.float32)
    small = ref[3:33, 7:47, 2:52]
    placed, shift = place_on_reference(sitk.GetImageFromArray(small), sitk.GetImageFromArray(ref))
    assert shift == (3, 7, 2)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(placed)[3:33, 7:47, 2:52], small)


def test_field_moves_points_by_the_blended_offset(layout):
    offsets = {t.name: (1.0, -2.0, 0.5) for t in layout.tiles}
    nz, my, mx = layout.mosaic_shape
    ref = sitk.Image([mx // 4, my // 4, nz // 4], sitk.sitkFloat32)
    ref.SetSpacing([0.0144] * 3)
    ref.SetDirection([-1, 0, 0, 0, 1, 0, 0, 0, -1])
    field = sitk.GetArrayFromImage(nominal_to_stitched_field(layout, offsets, ref, ap_weights=None))
    # (z, y, x) voxels of (4.0, 3.6, 3.6) um, in mm along LPS through the direction matrix
    expected = np.array([-0.5 * 0.0036, -2.0 * 0.0036, -1.0 * 0.004])
    np.testing.assert_allclose(field.reshape(-1, 3), np.broadcast_to(expected, (field.size // 3, 3)), atol=1e-12)


def test_field_blends_offsets_across_an_overlap(layout):
    offsets = {t.name: (0.0, 0.0, 0.0 if t.col == 0 else 4.0) for t in layout.tiles}
    nz, my, mx = layout.mosaic_shape
    ref = sitk.Image([mx, my, nz], sitk.sitkFloat32)
    ref.SetSpacing([0.0036] * 3)
    dx = sitk.GetArrayFromImage(nominal_to_stitched_field(layout, offsets, ref, ap_weights=None))[0, 10, :, 0]
    ov_start = layout.step_px[1]
    assert dx[ov_start - 5] == 0
    assert dx[layout.shape[2] + 5] == pytest.approx(4 * 0.0036)
    assert np.all(np.diff(dx[ov_start : layout.shape[2]]) > 0)
