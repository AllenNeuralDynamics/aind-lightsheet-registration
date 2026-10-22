"""Whole-plane dips at the focus-ramp planes.

Before imaging, the operator sets the detection focus at planes a fixed z step apart and the microscope interpolates
between them; the planes near each one come out a few percent dimmer, sloping down into it and recovering sharply
after. The planes are in the raw asset's ``TileSettings.ini``. The gain depends only on the distance past the last
ramp plane, so pooling the tiles' high-passed per-plane brightness over every ramp period separates it from anatomy,
which does not repeat with the ramp.
"""

from __future__ import annotations

import json
import re
import struct
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from aind_lightsheet_registration.tiles import Tile, TileLayout


def read_focus_ramp(tile_settings: str, channel: str, wavelengths: list[str]) -> np.ndarray:
    r"""Stage z (µm) of ``channel``'s focus-ramp planes, ascending.

    Each ``Focus.N`` entry of ``[Focus Compensation]`` is a LabVIEW-flattened 2 × n array: ``\XX`` hex escapes, then
    big-endian int32 dimensions, then big-endian doubles, the first row stage z and the second the focus setting.
    ``N`` indexes the session's wavelength list.

    Parameters
    ----------
    tile_settings : str
        The file's text, read as latin-1.
    channel : str
        Excitation wavelength.
    wavelengths : list of str
        The session's wavelengths in order, as in ``metadata.json``'s ``wavelength_config``.

    Returns
    -------
    numpy.ndarray
        Ramp planes' stage z; empty when the channel has no ramp.
    """
    start = tile_settings.index("[Focus Compensation]")
    end = tile_settings.find("\n[", start + 1)
    section = tile_settings[start : end if end >= 0 else None]
    found = re.search(rf'^Focus\.{wavelengths.index(channel)} = "(.*)"', section, re.MULTILINE)
    if found is None:
        raise ValueError(f"no focus table for {channel} nm")
    raw = re.sub(rb"\\([0-9A-Fa-f]{2})", lambda m: bytes([int(m.group(1), 16)]), found.group(1).encode("latin1"))
    rows, cols = struct.unpack(">ii", raw[:8])
    values = struct.unpack(f">{rows * cols}d", raw[8 : 8 + 8 * rows * cols])
    return np.sort(np.array(values[:cols]))


def ramp_depths_um(tile_settings: str, metadata: dict[str, Any], channel: str) -> np.ndarray:
    """Depth (µm) of each ramp plane below the first plane of ``channel``'s stacks, which must all start together."""
    starts = {float(t["Z"]) for t in metadata["tile_config"].values() if t["Laser"] == channel}
    if len(starts) != 1:
        raise ValueError(f"{channel} nm stacks start at different depths: {sorted(starts)}")
    ramp = read_focus_ramp(tile_settings, channel, list(metadata["wavelength_config"]))
    return np.asarray(ramp - starts.pop(), float)


def plane_depths_um(layout: TileLayout) -> np.ndarray:
    """Depth (µm) of each tile plane's center below the first full-resolution plane's center."""
    vz = layout.voxel_um[0]
    step = vz / 2**layout.level
    return np.asarray((np.arange(layout.shape[0]) + 0.5) * vz - step / 2, float)


@dataclass(frozen=True)
class FocusRamp:
    """Log gain against distance past the last focus-ramp plane.

    Attributes
    ----------
    ramp_um : numpy.ndarray
        Ramp planes' depths below the stacks' first plane (µm).
    bin_um : float
        Width of a distance bin (µm).
    log_gain : numpy.ndarray
        Log gain per bin, from 0 past a ramp plane; median zero.
    """

    ramp_um: np.ndarray
    bin_um: float
    log_gain: np.ndarray

    @property
    def period_um(self) -> float:
        """Median spacing of the ramp planes."""
        return float(np.median(np.diff(self.ramp_um)))

    def along(self, depth_um: np.ndarray) -> np.ndarray:
        """Log gain at ``depth_um``.

        It is zero more than half a period before the first ramp plane or a period after the last. Planes before
        the first ramp plane take the distance past the plane a period before it, since the dip
        starts ahead of each ramp plane.
        """
        depth = np.asarray(depth_um, float)
        i = np.searchsorted(self.ramp_um, depth, side="right") - 1
        past = np.where(i >= 0, depth - self.ramp_um[np.maximum(i, 0)], depth - (self.ramp_um[0] - self.period_um))
        valid = (depth >= self.ramp_um[0] - self.period_um / 2) & (past < self.bin_um * len(self.log_gain))
        k = np.clip((past // self.bin_um).astype(int), 0, len(self.log_gain) - 1)
        return np.where(valid, self.log_gain[k], 0.0)

    def save(self, path: Path) -> None:
        """Write as JSON."""
        record = {"ramp_um": self.ramp_um.tolist(), "bin_um": self.bin_um, "log_gain": self.log_gain.tolist()}
        path.write_text(json.dumps(record, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> FocusRamp:
        """Read what :meth:`save` wrote."""
        r = json.loads(path.read_text())
        return cls(np.array(r["ramp_um"], float), float(r["bin_um"]), np.array(r["log_gain"], float))


def estimate_focus_ramp(
    layout: TileLayout,
    load: Callable[[Tile], np.ndarray],
    ramp_um: np.ndarray,
    *,
    highpass_um: float = 1000.0,
    tissue_floor: float = 40.0,
    min_tissue_fraction: float = 0.2,
) -> FocusRamp:
    """Fit the ramp gain from every tile's per-plane brightness.

    Each tile's median log intensity per plane over its tissue is high-passed by a running median ``highpass_um``
    long, the tiles are pooled by their median, and the pooled profile is binned by distance past the last ramp plane
    at one tile plane per bin, taking each bin's median over the periods. Planes within half the high-pass of the
    stack's ends are left out, where the running median follows the dip.

    Parameters
    ----------
    layout : TileLayout
        The grid.
    load : callable
        Returns a tile's dark-subtracted stack.
    ramp_um : numpy.ndarray
        From :func:`ramp_depths_um`; at least two planes.
    highpass_um : float
        Length of the running median removed from each tile's profile.
    tissue_floor : float
        Intensity below which voxels are background.
    min_tissue_fraction : float
        Fewest tissue voxels a tile needs to count, and a tenth of that for one of its planes.

    Returns
    -------
    FocusRamp
        The gain.
    """
    if len(ramp_um) < 2:
        raise ValueError("a focus ramp needs at least two planes")
    vz = layout.voxel_um[0]
    size = 2 * round(highpass_um / vz / 2) + 1
    profiles = []
    for t in layout.tiles:
        log = np.log(np.clip(load(t), 1, None))
        tissue = log > np.log(tissue_floor)
        if tissue.mean() < min_tissue_fraction:
            continue
        with warnings.catch_warnings():  # planes without tissue
            warnings.simplefilter("ignore", RuntimeWarning)
            plane = np.nanmedian(np.where(tissue, log, np.nan), axis=(1, 2))
        plane[tissue.mean(axis=(1, 2)) < min_tissue_fraction / 10] = np.nan
        ok = np.isfinite(plane)
        if ok.sum() < size:
            continue
        trend = median_filter(np.where(ok, plane, np.nanmedian(plane)), size=size, mode="nearest")
        profiles.append(np.where(ok, plane - trend, np.nan))
    if not profiles:
        raise ValueError("no tile has enough tissue to measure the focus ramp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pooled = np.nanmedian(profiles, axis=0)
    depth = plane_depths_um(layout)
    period = float(np.median(np.diff(ramp_um)))
    n_bins = int(np.ceil(np.max(np.diff(ramp_um)) / vz))
    i = np.searchsorted(ramp_um, depth, side="right") - 1
    past = depth - ramp_um[np.maximum(i, 0)]
    interior = (i >= 0) & (depth >= highpass_um / 2) & (depth <= depth[-1] - highpass_um / 2) & np.isfinite(pooled)
    if not np.any(interior):
        raise ValueError(f"the stacks are too short for a {highpass_um:g} um high-pass and a {period:g} um period")
    bins = np.minimum((past // vz).astype(int), n_bins - 1)
    gain = np.array(
        [np.median(pooled[interior & (bins == b)]) if np.any(interior & (bins == b)) else 0.0 for b in range(n_bins)]
    )
    return FocusRamp(np.asarray(ramp_um, float), vz, gain - np.median(gain))
