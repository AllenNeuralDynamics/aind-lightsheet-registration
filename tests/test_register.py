"""Tests for preparing a brain from its pyramid and registering it to a template."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from aind_registration_utils.ants_cli import LinearConfig, RegistrationConfig, SynConfig
from aind_registration_utils.chains import SynTransform, transform_points

from aind_lightsheet_registration.register import (
    SIDECAR_NAME,
    PreparedBrain,
    Template,
    prepare_brain,
    register_to_template,
)
from tests.conftest import write_pyramid

ANTS_PATH = Path(os.environ.get("ANTSPATH", ""))
needs_ants = pytest.mark.skipif(
    not (ANTS_PATH / "antsRegistration").is_file(), reason="needs ANTSPATH with antsRegistration"
)


def _ellipsoid(shape, center, radii) -> np.ndarray:
    grid = np.indices(shape)
    radius2 = np.zeros(shape)
    for g, c, r in zip(grid, center, radii, strict=True):
        radius2 += ((g - c) / r) ** 2
    return radius2 <= 1


def test_prepare_brain_writes_registration_inputs(tmp_path, nd_metadata):
    """A 50 µm pyramid read at 100 µm: level 1 is used and the mask finds the tissue despite a bright structure."""
    shape = (64, 80, 96)
    tissue = _ellipsoid(shape, (32, 40, 48), (22, 30, 38))
    bright = _ellipsoid(shape, (32, 40, 70), (8, 10, 10))
    base = np.where(bright, 3000, np.where(tissue, 150, 2)).astype(np.uint16)
    path = tmp_path / "brain.ome.zarr"
    write_pyramid(path, base, 3, base_spacing_um=(50.0, 50.0, 50.0))

    out = prepare_brain(str(path), nd_metadata, tmp_path / "prepared", target_spacing_mm=0.1)

    record = json.loads(out.front_end.read_text())
    assert record["level"] == 1
    # A one-voxel shell is a tenth of this ellipsoid at 100 µm; a linear threshold would keep only the blob.
    assert record["mask_volume_mm3"] == pytest.approx(tissue.sum() * 0.05**3, rel=0.1)
    normalized = sitk.ReadImage(str(out.normalized))
    assert normalized.GetSpacing() == pytest.approx((0.1, 0.1, 0.1))
    assert normalized.GetDirection() == pytest.approx((-1, 0, 0, 0, 1, 0, 0, 0, -1))
    values = sitk.GetArrayViewFromImage(normalized)
    assert values.min() == 0.0 and values.max() == 1.0
    assert out == PreparedBrain.in_dir(tmp_path / "prepared")


def _phantom(origin, spacing=0.1, shape=(48, 48, 48)) -> tuple[sitk.Image, sitk.Image]:
    """An ellipsoid with an off-center blob, so a rigid fit has one answer, and its mask."""
    tissue = _ellipsoid(shape, (24, 24, 24), (16, 18, 20))
    blob = _ellipsoid(shape, (20, 30, 16), (4, 4, 4))
    values = np.where(blob, 1.0, np.where(tissue, 0.5, 0.0)).astype(np.float32)
    image = sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(values), 1.0)
    mask = sitk.GetImageFromArray(tissue.astype(np.uint8))
    for img in (image, mask):
        img.SetSpacing((spacing,) * 3)
        img.SetOrigin(origin)
    return image, mask


@needs_ants
def test_register_to_template_recovers_a_shift(tmp_path):
    """The brain is the template moved 0.6, -0.4, 0.3 mm; the transforms carry template landmarks onto it."""
    shift = np.array([0.6, -0.4, 0.3])
    template_image, template_mask = _phantom((0.0, 0.0, 0.0))
    brain_image, brain_mask = _phantom(tuple(shift))
    template = Template(tmp_path / "template.nii.gz", tmp_path / "template_mask.nii.gz")
    sitk.WriteImage(template_image, str(template.image))
    sitk.WriteImage(template_mask, str(template.mask))
    brain = PreparedBrain.in_dir(tmp_path / "brain")
    brain.mask.parent.mkdir()
    sitk.WriteImage(brain_image, str(brain.normalized))
    sitk.WriteImage(brain_mask, str(brain.mask))

    config = RegistrationConfig(
        linear=LinearConfig(iterations=[100, 50], shrink_factors=[2, 1], smoothing_sigmas=[1.0, 0.0]),
        syn=SynConfig(iterations=[20, 10], shrink_factors=[2, 1], smoothing_sigmas=[1.0, 0.0]),
        random_seed=1,
    )
    result = register_to_template(brain, template, tmp_path / "reg", config=config, threads=1, ants_path=ANTS_PATH)

    assert result.sidecar.name == SIDECAR_NAME
    landmarks = np.array(
        [template_image.TransformContinuousIndexToPhysicalPoint(i) for i in [(16, 30, 20), (24, 24, 24)]]
    )
    mapped = transform_points(landmarks, [(SynTransform.from_sidecar(result.sidecar), "fixed_to_moving")])
    np.testing.assert_allclose(mapped, landmarks + shift, atol=0.05)
