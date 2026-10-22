"""Tests for measuring the bulb shadow against a template on synthetic volumes."""

import json

import numpy as np
import pytest
import SimpleITK as sitk
from aind_registration_utils.chains import _Step
from scipy.ndimage import gaussian_filter

from aind_lightsheet_registration import olfactory as olf_module
from aind_lightsheet_registration.olfactory import (
    BulbShadowModel,
    bulb_shadow,
    correct_bulb_shadow,
    from_depth_first,
    remove_shadow,
    to_depth_first,
    zero_level,
)
from aind_lightsheet_registration.register import PreparedBrain, Template

SPACING = 0.05
TOP = 20  # first olfactory plane; the fissure is the dark plane above it
SHADOW = -0.3


def _template() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cortex over a bulb in the front half (y < 24), with a dark fissure between them."""
    shape = (60, 48, 48)
    rng = np.random.default_rng(0)
    template = (1.0 + 0.3 * gaussian_filter(rng.normal(size=shape), 1.5)).astype(np.float32)
    olfactory = np.zeros(shape, bool)
    olfactory[TOP:, :24] = True
    template[TOP - 1, :24] = -0.8  # near the normalized floor, as pigment is
    return template, np.ones(shape, bool), olfactory


def _shadowed(template: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """``template`` attenuated by ``tau`` at every depth below the fissure."""
    off = BulbShadowModel().log_offset
    att = np.zeros(template.shape)
    att[TOP:] = tau
    return ((template + off) * np.exp(att) - off).astype(np.float32)


def _patch() -> np.ndarray:
    tau = np.zeros((48, 48))
    tau[:24, 8:40] = SHADOW
    return tau


def test_shallow_shadow_is_recovered_and_nothing_above_the_fissure():
    template, mask, olfactory = _template()
    brain = _shadowed(template, _patch())
    shadow = bulb_shadow(brain, template, mask, olfactory, SPACING)
    assert np.all(shadow.fissure[:24] == TOP - 1)
    att = shadow.log_attenuation
    assert att.max() <= 0
    assert np.all(att[: TOP - 1] == 0)
    assert np.abs(att[TOP + 1, 30:]).max() < 0.01  # it spreads beyond the bulb only as it blurs with depth
    # at 0.1 mm the gain is 1 and the lateral blur, 0.06 mm, is small against the patch
    np.testing.assert_allclose(att[TOP + 1, 4:16, 18:30], SHADOW, atol=0.03)
    # deeper the model scales the blurred map by the fitted gain
    deep = TOP + 15
    p = BulbShadowModel()
    assert att[deep, 10, 24] == pytest.approx(SHADOW * np.interp(0.75, p.gain_depth_mm, p.gain), abs=0.05)
    corrected = remove_shadow(brain, shadow.log_attenuation)
    shallow = np.s_[TOP : TOP + 3, 4:16, 18:30]
    np.testing.assert_allclose(corrected[shallow], template[shallow], rtol=0.05)


def test_zero_level_restores_shadow_the_template_still_carries():
    template, mask, olfactory = _template()
    tau_brain, tau_template = _patch(), 0.5 * _patch()
    under_corrected = _shadowed(template, tau_template)
    brain = _shadowed(template, tau_brain)
    measured = bulb_shadow(brain, under_corrected, mask, olfactory, SPACING).measured
    assert np.nanmedian(measured[4:16, 18:30]) == pytest.approx(SHADOW / 2, abs=0.02)
    zero = zero_level([np.where(np.isfinite(measured), tau_brain, np.nan)], [measured], SPACING)
    shadow = bulb_shadow(brain, under_corrected, mask, olfactory, SPACING, zero)
    np.testing.assert_allclose(shadow.log_attenuation[TOP + 1, 4:16, 18:30], SHADOW, atol=0.03)


def test_depth_first_round_trip():
    image = sitk.GetImageFromArray(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    image.SetDirection((-1, 0, 0, 0, 0, 1, 0, -1, 0))
    flipped = to_depth_first(image)
    assert sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(flipped.GetDirection()) == "RPI"
    back = from_depth_first(flipped, image)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(back), sitk.GetArrayFromImage(image))
    assert back.GetDirection() == pytest.approx(image.GetDirection())


class _Shifted:
    """A registration that only translates: template points land ``shift`` mm away in the brain."""

    def __init__(self, shift: tuple[float, float, float]) -> None:
        self.shift = shift

    def _steps(self, direction: str) -> list[_Step]:
        sign = 1 if direction == "fixed_to_moving" else -1
        return [_Step(sitk.TranslationTransform(3, [sign * v for v in self.shift]), None, "shift")]


def _image(array: np.ndarray, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> sitk.Image:
    """Depth-first ``array`` as an RPI image."""
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((SPACING,) * 3)
    image.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, -1))
    image.SetOrigin(origin)
    return image


def test_registered_brain_is_corrected_on_its_own_grid(tmp_path, monkeypatch):
    template, mask, olfactory = _template()
    shift = (0.1, 0.05, -0.15)
    monkeypatch.setattr(olf_module.SynTransform, "from_sidecar", lambda path: _Shifted(shift))
    files = {}
    for name, array in (
        ("template", template),
        ("mask", mask.astype(np.uint8)),
        ("olfactory", olfactory.astype(np.uint8)),
        ("zero", np.zeros(template.shape, np.float32)),
    ):
        files[name] = tmp_path / f"{name}.nii.gz"
        # stored in another orientation, as a CCF-aligned template is
        sitk.WriteImage(sitk.DICOMOrient(_image(array), "PIR"), str(files[name]))
    brain_path = tmp_path / "brain.nii.gz"
    sitk.WriteImage(_image(_shadowed(template, _patch()), shift), str(brain_path))
    brain = PreparedBrain(brain_path, brain_path, brain_path, brain_path, tmp_path / "front_end.json")
    out = correct_bulb_shadow(
        brain,
        tmp_path / "sidecar.json",
        Template(files["template"], files["mask"]),
        files["olfactory"],
        files["zero"],
        tmp_path / "bulb",
    )
    assert out.normalized == tmp_path / "bulb" / "brain_normalized_bulb.nii.gz"
    assert out.mask == brain.mask
    corrected = sitk.GetArrayFromImage(sitk.ReadImage(str(out.normalized)))
    shallow = np.s_[TOP : TOP + 3, 4:16, 18:30]
    np.testing.assert_allclose(corrected[shallow], template[shallow], rtol=0.05)
    np.testing.assert_array_equal(corrected[: TOP - 1], _shadowed(template, _patch())[: TOP - 1])
    record = json.loads((tmp_path / "bulb" / "bulb_shadow.json").read_text())
    assert record["sharp_map_median"] < 0
