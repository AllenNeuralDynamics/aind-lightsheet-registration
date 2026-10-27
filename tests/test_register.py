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
    WORKING_SPACING_MM,
    LabelTerm,
    PreparedBrain,
    Template,
    prepare_brain,
    register_to_template,
    registration_config,
    write_label_terms,
    write_registration_inputs,
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

    out = prepare_brain(
        str(path), nd_metadata, tmp_path / "prepared", target_spacing_mm=0.1, normalization="percentile"
    )

    record = json.loads(out.front_end.read_text())
    assert record["level"] == 1 and record["normalization"] == "percentile"
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
    result = register_to_template(
        brain, template, tmp_path / "reg", config=config, working_spacing_mm=0.1, threads=1, ants_path=ANTS_PATH
    )

    assert result.sidecar.name == SIDECAR_NAME
    landmarks = np.array(
        [template_image.TransformContinuousIndexToPhysicalPoint(i) for i in [(16, 30, 20), (24, 24, 24)]]
    )
    mapped = transform_points(landmarks, [(SynTransform.from_sidecar(result.sidecar), "fixed_to_moving")])
    np.testing.assert_allclose(mapped, landmarks + shift, atol=0.05)


def test_registration_config_smooths_over_0_1_mm_and_holds_the_total_field():
    config = registration_config()
    assert config.syn.update_field_variance**0.5 * WORKING_SPACING_MM == pytest.approx(0.1)
    assert config.syn.radius_or_bins * WORKING_SPACING_MM == pytest.approx(0.2)
    assert config.syn.total_field_variance**0.5 * WORKING_SPACING_MM == pytest.approx(0.05)
    assert config.syn.shrink_factors[0] * WORKING_SPACING_MM == pytest.approx(1.2)
    assert config.syn.shrink_factors[-1] == 1 and len(config.syn.iterations) == 7


def _write_pair(tmp_path, brain_spacing, template_spacing):
    template_image, template_mask = _phantom((0.0, 0.0, 0.0), spacing=template_spacing)
    brain_image, brain_mask = _phantom((0.0, 0.0, 0.0), spacing=brain_spacing)
    template = Template(tmp_path / "template.nii.gz", tmp_path / "template_mask.nii.gz")
    sitk.WriteImage(template_image, str(template.image))
    sitk.WriteImage(template_mask, str(template.mask))
    brain = PreparedBrain.in_dir(tmp_path / "brain")
    brain.mask.parent.mkdir()
    sitk.WriteImage(brain_image, str(brain.normalized))
    sitk.WriteImage(brain_mask, str(brain.mask))
    return brain, template


@pytest.mark.parametrize(
    ("brain_spacing", "template_spacing", "message"),
    [(0.03, 0.05, "whole multiple"), (0.1, 0.05, "whole multiple"), (0.025, 0.1, "template spacing")],
)
def test_register_to_template_refuses_mismatched_grids(tmp_path, brain_spacing, template_spacing, message):
    brain, template = _write_pair(tmp_path, brain_spacing, template_spacing)
    with pytest.raises(ValueError, match=message):
        register_to_template(brain, template, tmp_path / "reg")


def test_iqr_template_is_used_as_stored_and_the_brain_clipped_to_its_range(tmp_path):
    brain, template = _write_pair(tmp_path, 0.05, 0.05)
    image = sitk.ReadImage(str(brain.normalized)) * 4.0 - 1.0  # about [-1, 3], wider than the template's [0, 1]
    sitk.WriteImage(image, str(brain.normalized))
    fixed, moving, *_ = write_registration_inputs(brain, template, tmp_path / "inputs", working_spacing_mm=0.05)
    assert fixed == template.image
    stored = sitk.GetArrayFromImage(sitk.ReadImage(str(template.image)))
    values = sitk.GetArrayFromImage(sitk.ReadImage(str(moving)))
    assert (values.min(), values.max()) == pytest.approx((stored.min(), stored.max()), abs=1e-6)

    clipped = Template(template.image, template.mask, clip=(-0.5, 2.0))
    _, moving, *_ = write_registration_inputs(brain, clipped, tmp_path / "inputs", working_spacing_mm=0.05)
    values = sitk.GetArrayFromImage(sitk.ReadImage(str(moving)))
    assert (values.min(), values.max()) == pytest.approx((-0.5, 2.0), abs=1e-6)


def test_percentile_template_is_rescaled(tmp_path):
    brain, template = _write_pair(tmp_path, 0.05, 0.05)
    sitk.WriteImage(sitk.ReadImage(str(template.image)) * 50.0, str(template.image))
    percentile = Template(template.image, template.mask, normalization="percentile")
    fixed, *_ = write_registration_inputs(brain, percentile, tmp_path / "inputs", working_spacing_mm=0.05)
    values = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed)))
    assert fixed != template.image and (values.min(), values.max()) == (0.0, 1.0)


def test_register_to_template_refuses_a_brain_normalized_for_another_template(tmp_path):
    brain, template = _write_pair(tmp_path, 0.05, 0.05)
    brain.front_end.write_text(json.dumps({"normalization": "percentile"}))
    with pytest.raises(ValueError, match="normalized as 'percentile'"):
        register_to_template(brain, template, tmp_path / "reg", working_spacing_mm=0.05)


def test_label_terms_are_distance_maps_on_the_registration_grids(tmp_path):
    """A brain label drawn at 25 µm lands on the 50 µm moving grid, negative inside and saturating outside."""
    fixed_image, fixed_label = _phantom((0.0, 0.0, 0.0), spacing=0.05)
    fine_label = sitk.GetImageFromArray(_ellipsoid((96, 96, 96), (48, 48, 48), (32, 36, 40)).astype(np.uint8))
    fine_label.SetSpacing((0.025,) * 3)
    moving_image, _ = _phantom((0.0, 0.0, 0.0), spacing=0.05)
    paths = {n: tmp_path / f"{n}.nii.gz" for n in ("fixed", "moving", "fixed_label", "brain_label")}
    for image, name in [(fixed_image, "fixed"), (moving_image, "moving"), (fixed_label, "fixed_label")]:
        sitk.WriteImage(image, str(paths[name]))
    sitk.WriteImage(fine_label, str(paths["brain_label"]))
    term = LabelTerm("ac", paths["fixed_label"], paths["brain_label"], weight=0.2)

    (metric,) = write_label_terms([term], paths["fixed"], paths["moving"], tmp_path)

    assert (metric.metric, metric.weight) == ("MeanSquares", 0.2)
    moving_map = sitk.ReadImage(metric.moving)
    assert moving_map.GetSize() == moving_image.GetSize()
    values = sitk.GetArrayFromImage(moving_map)
    assert values[24, 24, 24] < -0.9 and values[0, 0, 0] > 0.9


def test_an_empty_label_is_refused(tmp_path):
    image, label = _phantom((0.0, 0.0, 0.0), spacing=0.05)
    paths = [tmp_path / f"{n}.nii.gz" for n in ("image", "label", "empty")]
    sitk.WriteImage(image, str(paths[0]))
    sitk.WriteImage(label, str(paths[1]))
    sitk.WriteImage(label * 0, str(paths[2]))
    with pytest.raises(ValueError, match="brain label 'ac'"):
        write_label_terms([LabelTerm("ac", paths[1], paths[2])], paths[0], paths[0], tmp_path)


def test_a_label_value_is_only_inside_or_outside(tmp_path):
    image, tissue = _phantom((0.0, 0.0, 0.0), spacing=0.05)
    paths = {n: tmp_path / f"{n}.nii.gz" for n in ("image", "one", "five")}
    sitk.WriteImage(image, str(paths["image"]))
    sitk.WriteImage(tissue, str(paths["one"]))
    sitk.WriteImage(tissue * 5, str(paths["five"]))
    (metric,) = write_label_terms(
        [LabelTerm("ac", paths["one"], paths["five"])], paths["image"], paths["image"], tmp_path
    )
    maps = [sitk.GetArrayFromImage(sitk.ReadImage(m)) for m in (metric.fixed, metric.moving)]
    np.testing.assert_array_equal(*maps)


def test_a_label_outside_tissue_is_refused(tmp_path):
    image, tissue = _phantom((0.0, 0.0, 0.0), spacing=0.05)
    outside = sitk.GetImageFromArray(_ellipsoid((48, 48, 48), (3, 3, 3), (3, 3, 3)).astype(np.uint8))
    outside.CopyInformation(tissue)
    paths = {n: tmp_path / f"{n}.nii.gz" for n in ("image", "tissue", "outside")}
    for img, name in [(image, "image"), (tissue, "tissue"), (outside, "outside")]:
        sitk.WriteImage(img, str(paths[name]))
    term = LabelTerm("ac", paths["tissue"], paths["outside"])
    with pytest.raises(ValueError, match="brain label 'ac'.*right frame"):
        write_label_terms(
            [term],
            paths["image"],
            paths["image"],
            tmp_path,
            fixed_tissue=paths["tissue"],
            moving_tissue=paths["tissue"],
        )
