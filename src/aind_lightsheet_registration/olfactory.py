"""The shadow over the olfactory bulb, measured in one brain against an already-corrected template.

Pigment in the meninges at the fissure between cortex and bulb attenuates the light from tissue below it. Tissue at
depth ``h`` below the fissure sees the attenuation map blurred laterally by ``blur_per_depth * h``, and a per-depth
gain scales the blur model where it alone over-corrects. The map is measured just below the brain's own fissure as
its log residual against the template, minus its residual in the cortex above (the brain's unshadowed offset). The
gain is below 1 inside that band, so the corrected template keeps part of its shadow there and the residual misses
it; a zero-level map from the template's build brains (:func:`zero_level`) adds it back.

Every array is on the template's grid with axis 0 running down the detection axis, superior to inferior, so depth
below the fissure grows with the index; :func:`to_depth_first` and :func:`from_depth_first` reorient an image.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from aind_registration_utils.chains import SynTransform, resample
from scipy import ndimage as ndi

from aind_lightsheet_registration.register import PreparedBrain, Template, _clip, _range

DEPTH_FIRST = "RPI"  # SimpleITK code whose third axis, numpy's first, points inferior


def to_depth_first(image: sitk.Image) -> sitk.Image:
    """``image`` reoriented so numpy axis 0 runs superior to inferior."""
    reoriented: sitk.Image = sitk.DICOMOrient(image, DEPTH_FIRST)
    return reoriented


def from_depth_first(image: sitk.Image, like: sitk.Image) -> sitk.Image:
    """``image`` reoriented back to ``like``'s axes."""
    code = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(like.GetDirection())
    reoriented: sitk.Image = sitk.DICOMOrient(image, code)
    return reoriented


@dataclass(frozen=True)
class BulbShadowModel:
    """Parameters of the shadow model; the defaults are those fitted on the 2026-10 iDISCO template's inputs.

    Attributes
    ----------
    log_offset : float
        Added to normalized intensity before the log, so that zero signal maps to about zero.
    blur_per_depth : float
        Lateral blur sigma per unit depth below the fissure.
    gain_depth_mm, gain : tuple of float
        Gain on the blurred map at these depths, interpolated between them.
    min_cover_mm : float
        Least non-olfactory tissue above the bulb for a column to be shadowed.
    shallow_mm : tuple of float
        Depth band below the fissure that measures the map.
    base_mm : tuple of float
        Band of cortex above the bulb's top that gives the brain's unshadowed offset.
    fissure_search_mm, own_fissure_mm : float
        Half-window for the template's fissure around the bulb's top, and for the brain's around the template's.
    stripe_highpass_mm, stripe_x_mm : float
        High-pass sigma in z and y, and x half-window, of the sheet stripes removed before the map is measured.
    """

    log_offset: float = 1.477
    blur_per_depth: float = 0.6
    gain_depth_mm: tuple[float, ...] = (0.175, 0.45, 0.8, 1.5)
    gain: tuple[float, ...] = (1.0, 0.445, 0.426, 0.538)
    min_cover_mm: float = 0.15
    shallow_mm: tuple[float, float] = (0.05, 0.4)
    base_mm: tuple[float, float] = (0.1, 0.5)
    base_smooth_mm: float = 0.2
    map_smooth_mm: float = 0.05
    fissure_search_mm: float = 0.3
    own_fissure_mm: float = 0.15
    onset_mm: float = 0.1
    stripe_highpass_mm: float = 0.1
    stripe_x_mm: float = 1.5


@dataclass(frozen=True)
class BulbShadow:
    """One brain's shadow.

    Attributes
    ----------
    log_attenuation : numpy.ndarray
        Float32 volume, at most 0; zero above the fissure, and reaching neighboring columns as it blurs with depth.
    measured : numpy.ndarray
        Destriped shallow-band map against the template before the zero level, NaN outside the shadowed columns.
    fissure : numpy.ndarray
        Index of the fissure down each (y, x) column.
    """

    log_attenuation: np.ndarray
    measured: np.ndarray
    fissure: np.ndarray


@dataclass(frozen=True)
class _Columns:
    """Which (y, x) columns the bulb shadow can reach, and where their olfactory tissue starts."""

    olfactory: np.ndarray
    covered: np.ndarray
    top: np.ndarray
    nearest: tuple[np.ndarray, np.ndarray]


def _columns(mask: np.ndarray, olfactory: np.ndarray, spacing: float, min_cover_mm: float) -> _Columns:
    olf = np.asarray(ndi.binary_opening(mask & olfactory), bool)
    nz = mask.shape[0]
    has = olf.any(0)
    top = np.where(has, np.argmax(olf, 0), nz)
    zz = np.arange(nz)[:, None, None]
    covered = has & ((mask & ~olf & (zz < top)).sum(0) * spacing >= min_cover_mm)
    if not covered.any():
        raise ValueError("no olfactory column has cortex above it")
    iy, ix = np.asarray(ndi.distance_transform_edt(~covered, return_distances=False, return_indices=True))
    return _Columns(olf, covered, top, (iy, ix))


def _fissure(image: np.ndarray, mask: np.ndarray, cols: _Columns, around: np.ndarray, half: int) -> np.ndarray:
    """Darkest tissue voxel down each covered column within ``half`` voxels of ``around``; nearest-filled elsewhere."""
    nz = image.shape[0]
    f = around.copy()
    for y, x in np.argwhere(cols.covered):
        a0, a1 = max(around[y, x] - half, 0), min(around[y, x] + half + 1, nz)
        f[y, x] = a0 + int(np.argmin(np.where(mask[a0:a1, y, x], image[a0:a1, y, x], np.inf)))
    iy, ix = cols.nearest
    return np.asarray(ndi.median_filter(np.where(cols.covered, f, f[iy, ix]), 3))


def _masked_smooth(t: np.ndarray, sigma: float | tuple[float, ...], where: np.ndarray) -> np.ndarray:
    w = np.where(where & np.isfinite(t), 1.0, 0.0)
    weight = np.maximum(ndi.gaussian_filter(w, sigma), 1e-6)
    return np.asarray(ndi.gaussian_filter(np.nan_to_num(t) * w, sigma) / weight)


def _band_median(r: np.ndarray, sel: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():  # columns without the band
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanmedian(np.where(sel, r, np.nan), axis=0))


def bulb_shadow(
    brain: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    olfactory: np.ndarray,
    spacing_mm: float,
    zero_level: np.ndarray | None = None,
    model: BulbShadowModel | None = None,
) -> BulbShadow:
    """Measure the bulb shadow in ``brain``; dividing ``brain + offset`` by its attenuation's exponential removes it.

    Parameters
    ----------
    brain : numpy.ndarray
        The brain resampled onto the template, normalized as the template is; depth-first (see module docstring).
    template : numpy.ndarray
        The shadow-corrected template.
    mask : numpy.ndarray
        Template tissue mask.
    olfactory : numpy.ndarray
        Template voxels in the olfactory bulb, accessory bulb, anterior olfactory nucleus or taenia tecta.
    spacing_mm : float
        Isotropic voxel size.
    zero_level : numpy.ndarray or None
        (y, x) map from :func:`zero_level`, added to the measured map; zero when None.
    model : BulbShadowModel or None
        Parameters; the defaults when None.

    Returns
    -------
    BulbShadow
        The attenuation and what it was measured from.
    """
    p = model or BulbShadowModel()
    sp = spacing_mm
    mask = mask.astype(bool)
    cols = _columns(mask, olfactory.astype(bool), sp, p.min_cover_mm)
    nz = mask.shape[0]
    zz = np.arange(nz)[:, None, None]
    z_sigma = p.map_smooth_mm / sp

    def smooth_z(a: np.ndarray) -> np.ndarray:
        return np.asarray(ndi.gaussian_filter1d(a, z_sigma, axis=0))

    fis_t = _fissure(smooth_z(template), mask, cols, cols.top, round(p.fissure_search_mm / sp))
    fis = _fissure(smooth_z(brain), mask, cols, fis_t, round(p.own_fissure_mm / sp))
    lb = np.log(np.maximum(brain + p.log_offset, 1e-3))
    lt = np.log(np.maximum(template + p.log_offset, 1e-3))
    lo, hi = (round(b / sp) for b in p.base_mm)
    base_band = mask & ~cols.olfactory & cols.covered[None] & (zz >= cols.top[None] - hi) & (zz <= cols.top[None] - lo)
    h = (zz - fis[None]) * sp
    shallow = mask & cols.covered[None] & (h >= p.shallow_mm[0]) & (h < p.shallow_mm[1])

    zero = np.zeros(mask.shape[1:]) if zero_level is None else np.nan_to_num(zero_level)

    def measure(log_brain: np.ndarray) -> np.ndarray:
        r = np.where(mask, log_brain - lt, np.nan)
        base = _masked_smooth(_band_median(r, base_band), p.base_smooth_mm / sp, cols.covered)
        t = _masked_smooth(_band_median(r, shallow), p.map_smooth_mm / sp, cols.covered)
        return np.where(cols.covered, t - base, np.nan)

    def sharp(measured: np.ndarray) -> np.ndarray:
        tau0 = np.minimum(np.where(cols.covered, measured + zero, 0.0), 0.0)
        return np.asarray(ndi.gaussian_filter(tau0, z_sigma))

    def depth_blur(tau: np.ndarray) -> np.ndarray:
        """Each voxel takes ``tau`` blurred laterally by ``blur_per_depth`` times its depth below the fissure."""
        fz = fis.astype(float) + 1
        depth = np.clip(zz - fz[None], 0, None) * sp
        levels = np.r_[0, np.geomspace(0.025, max(p.blur_per_depth * 3.0, 0.05), 12)]
        blurred = np.stack([ndi.gaussian_filter(tau, s / sp) if s > 0 else tau for s in levels])
        want = np.clip(p.blur_per_depth * depth, 0, levels[-1])
        i_hi = np.clip(np.searchsorted(levels, want), 1, len(levels) - 1)
        f = (want - levels[i_hi - 1]) / (levels[i_hi] - levels[i_hi - 1])
        yy = np.broadcast_to(np.arange(mask.shape[1])[None, :, None], mask.shape)
        xx = np.broadcast_to(np.arange(mask.shape[2])[None, None, :], mask.shape)
        t = (1 - f) * blurred[i_hi - 1, yy, xx] + f * blurred[i_hi, yy, xx]
        onset = round(p.onset_mm / sp)
        return np.asarray(t * np.clip(zz - fz[None] + onset, 0, onset) / onset)

    def stripes(log_brain: np.ndarray) -> np.ndarray:
        """Sheet stripes, constant along x: the x-average of the z-y high-pass of the residual."""
        w = mask.astype(float)
        r = np.where(mask, log_brain - lt, 0.0)
        s = p.stripe_highpass_mm / sp
        hp = (r - _masked_smooth(r, (s, s, 0.0), mask)) * w
        n = 2 * round(p.stripe_x_mm / sp) + 1
        st = ndi.uniform_filter1d(hp, n, axis=2) / np.maximum(ndi.uniform_filter1d(w, n, axis=2), 1e-6)
        return np.where(mask, st, 0.0)

    # stripes cross the shadow's edges, so measure them after a first correction and the map after removing them
    measured = measure(lb - stripes(lb - depth_blur(sharp(measure(lb)))))
    depth = np.clip(zz - fis[None] - 1, 0, None) * sp
    shadow = depth_blur(sharp(measured)) * np.interp(depth, p.gain_depth_mm, p.gain)
    return BulbShadow(np.where(mask, shadow, 0.0).astype(np.float32), measured, fis)


def zero_level(
    sharp_maps: list[np.ndarray], measured: list[np.ndarray], spacing_mm: float, smooth_mm: float = 0.2
) -> np.ndarray:
    """Fit the (y, x) map that :func:`bulb_shadow` adds to a brain's measured map, from the template's build brains.

    Parameters
    ----------
    sharp_maps : list of numpy.ndarray
        Each build brain's absolute sharp map, as the template build fitted it.
    measured : list of numpy.ndarray
        The same brains' :attr:`BulbShadow.measured` against the finished template.
    spacing_mm : float
        Isotropic voxel size.
    smooth_mm : float
        Smoothing of the mean difference.

    Returns
    -------
    numpy.ndarray
        The mean difference, smoothed, NaN where no brain measured a map.
    """
    diff = np.stack([t - m for t, m in zip(sharp_maps, measured, strict=True)])
    with warnings.catch_warnings():  # columns no brain measured
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(diff, axis=0)
    known = np.isfinite(mean)
    return np.where(known, _masked_smooth(mean, smooth_mm / spacing_mm, known), np.nan)


def remove_shadow(brain: np.ndarray, shadow: np.ndarray, model: BulbShadowModel | None = None) -> np.ndarray:
    """``brain`` with the log attenuation ``shadow`` divided out, in its own normalized units."""
    offset = (model or BulbShadowModel()).log_offset
    return np.where(shadow < 0, (brain + offset) * np.exp(-shadow) - offset, brain).astype(np.float32)


def correct_bulb_shadow(
    brain: PreparedBrain,
    sidecar: Path,
    template: Template,
    olfactory: Path,
    zero: Path,
    out_dir: Path,
    model: BulbShadowModel | None = None,
) -> PreparedBrain:
    """Measure a registered brain's bulb shadow on the template and divide it out of the brain's normalized image.

    Parameters
    ----------
    brain : PreparedBrain
        The brain as registered.
    sidecar : Path
        Its registration's transform sidecar.
    template : Template
        The shadow-corrected template it was registered to, with isotropic voxels.
    olfactory : Path
        Template-grid mask of the olfactory structures (see :func:`bulb_shadow`).
    zero : Path
        Template-grid volume holding the :func:`zero_level` map in every plane along the detection axis.
    out_dir : Path
        Directory for the shadow on both grids and the corrected image; created if missing.
    model : BulbShadowModel or None
        Parameters; the defaults when None.

    Returns
    -------
    PreparedBrain
        ``brain`` with the corrected image as its normalized one.
    """
    p = model or BulbShadowModel()
    out_dir.mkdir(parents=True, exist_ok=True)
    tpl = sitk.ReadImage(str(template.image), sitk.sitkFloat32)
    spacing = tpl.GetSpacing()
    if max(spacing) - min(spacing) > 1e-6:
        raise ValueError(f"template voxels {spacing} mm are not isotropic")
    image = sitk.ReadImage(str(brain.normalized), sitk.sitkFloat32)
    reg = SynTransform.from_sidecar(sidecar)
    on_template = resample(_clip(image, template.clip or _range(tpl)), tpl, [(reg, "fixed_to_moving")])

    def depth_first(img: sitk.Image) -> np.ndarray:
        return sitk.GetArrayFromImage(to_depth_first(img))

    with warnings.catch_warnings():  # columns without a zero level
        warnings.simplefilter("ignore", RuntimeWarning)
        zero_map = np.nanmedian(depth_first(sitk.ReadImage(str(zero), sitk.sitkFloat32)), axis=0)
    shadow = bulb_shadow(
        depth_first(on_template),
        depth_first(tpl),
        depth_first(sitk.ReadImage(str(template.mask), sitk.sitkUInt8)) > 0,
        depth_first(sitk.ReadImage(str(olfactory), sitk.sitkUInt8)) > 0,
        spacing[0],
        zero_map,
        p,
    )
    flipped = sitk.GetImageFromArray(shadow.log_attenuation)
    flipped.CopyInformation(to_depth_first(tpl))
    on_tpl = from_depth_first(flipped, tpl)
    on_brain = resample(on_tpl, image, [(reg, "moving_to_fixed")])
    corrected = sitk.GetImageFromArray(
        remove_shadow(sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(on_brain), p)
    )
    corrected.CopyInformation(image)
    paths = {"template": out_dir / "shadow_template.nii.gz", "brain": out_dir / "shadow_brain.nii.gz"}
    normalized = out_dir / "brain_normalized_bulb.nii.gz"
    for img, path in ((on_tpl, paths["template"]), (on_brain, paths["brain"]), (corrected, normalized)):
        sitk.WriteImage(img, str(path))
    covered = np.isfinite(shadow.measured)
    tau = np.minimum(shadow.measured + np.nan_to_num(zero_map), 0)[covered]
    record = {
        "brain": str(brain.normalized),
        "sidecar": str(sidecar),
        "template": str(template.image),
        "olfactory": str(olfactory),
        "zero_level": str(zero),
        "model": {k: list(v) if isinstance(v, tuple) else v for k, v in vars(p).items()},
        "covered_columns": int(covered.sum()),
        "sharp_map_median": float(np.median(tau)),
        "sharp_map_p10": float(np.percentile(tau, 10)),
        "shadow": {k: str(v) for k, v in paths.items()},
    }
    (out_dir / "bulb_shadow.json").write_text(json.dumps(record, indent=2) + "\n")
    return replace(brain, normalized=normalized)
