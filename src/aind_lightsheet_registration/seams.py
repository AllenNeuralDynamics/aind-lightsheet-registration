"""Calibrated tile-row seam profile along A→P.

The sheet model leaves a few-percent ripple at every tile-row seam that is the same in every brain and extends past
the overlap, where one brain's data cannot separate it from anatomy. It is estimated once from the fusions of several
brains, as the median high-passed A→P log profile around each seam, and divided out of new brains at every seam.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from aind_lightsheet_registration.tiles import TileLayout

# profile window around each row's anterior edge, mm
WINDOW_MM = (-0.6, 1.0)
# A→P high-pass that leaves the ripple and removes anatomy and slow shading, mm
HIGHPASS_MM = 1.0
# ramp to zero at the window's ends so applying the profile adds no step, mm
TAPER_MM = 0.15


@dataclass
class SeamProfile:
    """Log gain against A→P distance (mm) from a tile row's anterior edge; zero outside the sampled range."""

    distance_mm: np.ndarray
    log_gain: np.ndarray

    def save(self, path: str) -> None:
        """Write to ``path`` (.npz)."""
        np.savez(path, distance_mm=self.distance_mm, log_gain=self.log_gain)

    @classmethod
    def load(cls, path: str) -> SeamProfile:
        """Read a profile written by :meth:`save`."""
        f = np.load(path)
        return cls(f["distance_mm"], f["log_gain"])

    def along(self, y_mm: np.ndarray, row_starts_mm: np.ndarray) -> np.ndarray:
        """Sum the log gain of rows starting at ``row_starts_mm``, at A→P positions ``y_mm``."""
        out = np.zeros_like(y_mm, dtype=float)
        for start in row_starts_mm:
            out += np.interp(y_mm - start, self.distance_mm, self.log_gain, left=0.0, right=0.0)
        return out


def seam_windows(
    image: np.ndarray, mask: np.ndarray, spacing_mm: float, seam_starts_mm: list[float]
) -> list[np.ndarray]:
    """High-passed A→P log profiles of ``image`` around each seam, sampled every ``spacing_mm`` across WINDOW_MM.

    ``image`` and ``mask`` are (z, y, x) with y along A→P and isotropic ``spacing_mm``. A window more than a fifth
    outside the tissue is dropped.
    """
    tissue = mask & (image > 0.3 * np.median(image[mask]))  # leaves out ventricles and the dim rim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columns and rows with no tissue
        col = np.nanmedian(np.where(tissue, np.log(np.maximum(image, 1e-6)), np.nan), axis=0)  # (y, x)
    ok = np.isfinite(col)
    size = 2 * round(HIGHPASS_MM / spacing_mm / 2) + 1
    smooth = median_filter(np.where(ok, col, np.nanmedian(col)), size=(size, 1), mode="nearest")
    hp = np.where(ok, col - smooth, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.nanmedian(hp, axis=1)
    prof[ok.sum(1) < 40] = np.nan
    lo, hi = (round(v / spacing_mm) for v in WINDOW_MM)
    out = []
    for start in seam_starts_mm:
        s = round(start / spacing_mm)
        if s + lo >= 0 and s + hi <= prof.size:
            w = prof[s + lo : s + hi]
            if np.isfinite(w).mean() > 0.8:
                out.append(np.nan_to_num(w))
    return out


def estimate_seam_profile(windows: list[np.ndarray], spacing_mm: float) -> SeamProfile:
    """Median over seam windows from :func:`seam_windows`, tapered to zero at both ends."""
    g = np.median(np.stack(windows), axis=0)
    n = max(round(TAPER_MM / spacing_mm), 1)
    taper = np.ones(g.size)
    taper[:n] = np.linspace(0, 1, n)
    taper[-n:] = np.linspace(1, 0, n)
    return SeamProfile(WINDOW_MM[0] + spacing_mm * np.arange(g.size), g * taper)


def interior_seam_starts_mm(layout: TileLayout, placement_y_px: int, spacing_mm: float) -> list[float]:
    """A→P position (mm, in the placed fusion) of each row's anterior edge, without the first and last seams.

    The first and last seams fall on the bulb-cortex junction and the cerebellum, whose anatomy would enter the
    profile.
    """
    step_mm = layout.step_px[0] * layout.voxel_um[1] / 1000
    return [placement_y_px * spacing_mm + r * step_mm for r in range(2, layout.n_rows - 1)]


def row_starts_mm(layout: TileLayout) -> np.ndarray:
    """A→P position (mm, in the mosaic) of the anterior edge of every row that has a seam."""
    return np.arange(1, layout.n_rows) * layout.step_px[0] * layout.voxel_um[1] / 1000
