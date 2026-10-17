"""Tests for destriping, the midline mirror map, the illumination model and the annotation transform chain."""

import csv
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

from aind_lightsheet_registration.annotations import MappedPoint, brain_to_ccf_transforms, write_mapped_csv
from aind_lightsheet_registration.destripe import destripe_stack
from aind_lightsheet_registration.illumination import PARAM_NAMES, SheetModel, _history
from aind_lightsheet_registration.mirror import mirror_indices
from aind_lightsheet_registration.tiles import Tile, TileLayout


def test_destripe_removes_horizontal_stripes_and_keeps_tissue():
    rng = np.random.default_rng(0)
    tissue = np.exp(gaussian_filter(rng.normal(size=(3, 200, 250)), (0, 6, 6)) * 20) * 300 + 200
    stripes = np.exp(0.3 * rng.normal(size=(1, 200, 1)) * (rng.random((1, 200, 1)) < 0.15))

    def stripe_left(unprotect_finest):
        out = destripe_stack(tissue * stripes, level=3, unprotect_finest=unprotect_finest)
        assert out.shape == tissue.shape and out.dtype == np.float32
        residual = np.log(out[1]) - np.log(tissue[1])
        return np.std(residual.mean(axis=1)), np.std(residual - residual.mean(axis=1, keepdims=True))

    pipeline_stripes, _ = stripe_left(0)
    stripes_left, texture_change = stripe_left(2)
    assert stripes_left < 0.4 * np.std(np.log(stripes))
    assert stripes_left < 0.5 * pipeline_stripes  # the pipeline's protection keeps strong stripes
    assert texture_change < 0.03


def test_mirror_of_a_symmetric_brain_reflects_about_its_midline():
    shape, center_x = (48, 80, 128), 70  # brain off the array's center
    z, y, x = np.indices(shape, dtype=float)
    r2 = ((z - 24) / 18) ** 2 + ((y - 40) / 30) ** 2 + ((x - center_x) / 44) ** 2
    rng = np.random.default_rng(0)
    half = gaussian_filter(rng.normal(size=(*shape[:2], 2 * center_x + 1)), 2)
    texture = (half + half[:, :, ::-1])[:, :, : shape[2]]  # symmetric about center_x
    array = np.where(r2 < 1, np.exp(3 * texture) * 1000, 1.0).astype(np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing([0.05] * 3)
    mask = sitk.GetImageFromArray((r2 < 1).astype(np.uint8))
    mask.CopyInformation(image)
    mirror = mirror_indices(image, mask)
    inside = r2 < 0.6
    for axis, expected in ((2, 2 * center_x - x), (1, y), (0, z)):
        assert np.median(np.abs(mirror[axis][inside] - expected[inside])) < 0.5


def _layout() -> TileLayout:
    tiles = tuple(Tile(f"{r}_{c}", r, c, int(c > 0)) for r in range(2) for c in range(2))
    return TileLayout(
        tiles=tiles,
        level=3,
        voxel_um=(16.0, 14.4, 14.4),
        shape=(10, 40, 50),
        step_px=(36, 45),
        relative_power={("639", 0): 1.0, ("639", 1): 1.25},
    )


def test_history_counts_earlier_exposures_through_each_sheet():
    history = _history(_layout())
    assert history[0] == []
    assert history[1] == [((0, 0, 1), 1.0)]
    assert history[3] == [((0, 0, 1), 1.0), ((0, 1, 1), 1.0), ((1, 1, 0), 1.25)]


def test_log_illumination_without_bleaching_depends_only_on_side():
    q = dict.fromkeys(PARAM_NAMES, 0.0)
    for s in (0, 1):
        q.update({f"logw_{s}": np.log(5.0), f"H_{s}": 1.8, f"logsig_{s}": np.log(0.2), f"a2_{s}": -0.1})
    q.update(logk=-50.0, gain1=np.log(1.1))
    model = SheetModel(_layout(), np.array([q[n] for n in PARAM_NAMES]))
    maps = [model.log_illumination(t) for t in model.layout.tiles]
    assert maps[0].shape == (40, 50)
    np.testing.assert_allclose(maps[2], maps[0], atol=1e-12)
    np.testing.assert_allclose(maps[1] - maps[0], np.log(1.1), atol=1e-12)
    # a quadratic falloff along the beam, peaked at the field's center
    profile = maps[0][20]
    assert profile[25] > profile[0] and profile[25] > profile[-1]


def test_bleaching_dims_later_tiles():
    q = dict.fromkeys(PARAM_NAMES, 0.0)
    for s in (0, 1):
        q.update({f"logw_{s}": np.log(5.0), f"H_{s}": 1.8, f"logsig_{s}": np.log(0.2)})
    q.update(logk=np.log(0.2))
    model = SheetModel(_layout(), np.array([q[n] for n in PARAM_NAMES]))
    first, last = (model.log_illumination(model.layout.tiles[i]) for i in (0, 3))
    assert np.all(last < first)


def test_brain_to_ccf_chain_inverts_both_registrations_affine_first(tmp_path):
    reg, t2c = Path("reg"), Path("t2c")
    chain = brain_to_ccf_transforms(reg, t2c, point_fields=[Path("field.nii.gz")])
    assert chain == [
        "field.nii.gz",
        "[reg/reg_0GenericAffine.mat,1]",
        "reg/reg_1InverseWarp.nii.gz",
        "[t2c/syn_0GenericAffine.mat,1]",
        "t2c/syn_1InverseWarp.nii.gz",
    ]


def test_mapped_points_csv(tmp_path):
    p = MappedPoint("probe A", 1, "tip", np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), 385, "VISp")
    write_mapped_csv([p], tmp_path / "out.csv")
    with open(tmp_path / "out.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["layer"] == "probe A" and rows[0]["point"] == "1" and rows[0]["acronym"] == "VISp"
    assert float(rows[0]["ccf_z_mm"]) == pytest.approx(6.0)
