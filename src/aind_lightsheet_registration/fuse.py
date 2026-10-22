"""Fuse tiles into a mosaic, take it to a working resolution and place it on a reference brain's grid.

Placing on the reference keeps every mask, transform and annotation made on the reference valid for the fusion.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import SimpleITK as sitk

from aind_lightsheet_registration.reader import finish_downsampling
from aind_lightsheet_registration.seams import SeamProfile, row_starts_mm
from aind_lightsheet_registration.stitch import Offsets, SideShear, blend_weights, move_tile
from aind_lightsheet_registration.tiles import Tile, TileLayout


def fuse_mosaic(
    layout: TileLayout,
    load: Callable[[Tile], np.ndarray],
    *,
    log_gain: Callable[[Tile], np.ndarray] | None = None,
    offsets: Offsets | None = None,
    ap_weights: tuple[float, float] | None = None,
    seam_profile: SeamProfile | None = None,
    shear: SideShear | None = None,
) -> sitk.Image:
    """Blend the tiles at their nominal positions, or sheared and moved by ``shear`` and ``offsets``.

    Parameters
    ----------
    layout : TileLayout
        The grid.
    load : callable
        Returns a tile's dark-subtracted stack.
    log_gain : callable or None
        Returns a tile's (y, x) log illumination; the tile is divided by its exponential.
    offsets : dict or None
        Tile offsets (z, y, x voxels) from :func:`~aind_lightsheet_registration.stitch.solve_offsets`, applied with
        linear interpolation; None keeps the nominal grid.
    ap_weights : tuple of float or None
        A→P blending; see :func:`~aind_lightsheet_registration.stitch.blend_weights`.
    seam_profile : SeamProfile or None
        Calibrated tile-row seam profile, divided out of the blended mosaic at every row's nominal anterior edge.
    shear : SideShear or None
        Per-side z-y shear from :func:`~aind_lightsheet_registration.stitch.measure_side_shear`, applied with the
        offsets in one interpolation.

    Returns
    -------
    sitk.Image
        Float32 mosaic with the level's spacing in mm, identity direction and origin 0.
    """
    _, ny, nx = layout.shape
    num = np.zeros(layout.mosaic_shape, np.float32)
    den = np.zeros(layout.mosaic_shape[1:], np.float32)
    for tile in layout.tiles:
        y0, x0 = tile.row * layout.step_px[0], tile.col * layout.step_px[1]
        w = blend_weights(layout, tile, np.arange(y0, y0 + ny), np.arange(x0, x0 + nx), ap_weights=ap_weights)
        w = w.astype(np.float32)
        stack = load(tile)
        if log_gain is not None:  # in the tile's own frame, before it moves
            stack = stack * np.exp(-log_gain(tile)).astype(np.float32)
        if offsets is not None or shear is not None:
            offset = offsets[tile.name] if offsets is not None else (0.0, 0.0, 0.0)
            stack = move_tile(stack, offset, shear.of(tile) if shear is not None else 0.0)
        num[:, y0 : y0 + ny, x0 : x0 + nx] += stack * w
        den[y0 : y0 + ny, x0 : x0 + nx] += w
    num /= np.maximum(den, 1e-6)
    if seam_profile is not None:
        y_mm = np.arange(num.shape[1]) * layout.voxel_um[1] / 1000
        num *= np.exp(-seam_profile.along(y_mm, row_starts_mm(layout))).astype(np.float32)[None, :, None]
    image = sitk.GetImageFromArray(num)
    image.SetSpacing([v / 1000 for v in layout.voxel_um[::-1]])
    return image


def place_on_reference(
    image: sitk.Image, reference: sitk.Image, *, shift: tuple[int, int, int] | None = None
) -> tuple[sitk.Image, tuple[int, int, int]]:
    """Copy ``image`` onto ``reference``'s grid at the integer voxel shift maximizing their log-intensity correlation.

    Parameters
    ----------
    image : sitk.Image
        Fusion at ``reference``'s spacing.
    reference : sitk.Image
        Brain whose grid the result takes.
    shift : tuple of int or None
        Voxel shift (z, y, x) to use instead of searching.

    Returns
    -------
    tuple
        The placed float32 image with ``reference``'s geometry, and the shift (z, y, x) of ``image`` within it.
    """
    mine = sitk.GetArrayFromImage(image)
    target = sitk.GetArrayFromImage(reference)
    if shift is None:
        shape = [max(a, b) for a, b in zip(mine.shape, target.shape, strict=True)]

        def zscore_log(a: np.ndarray) -> np.ndarray:
            pad = np.zeros(shape, np.float32)
            pad[: a.shape[0], : a.shape[1], : a.shape[2]] = np.log1p(np.clip(a, 0, None))
            return np.asarray(pad - pad.mean())

        xc = np.fft.irfftn(
            np.fft.rfftn(zscore_log(target)) * np.conj(np.fft.rfftn(zscore_log(mine))), shape, axes=(0, 1, 2)
        )
        peak = np.unravel_index(np.argmax(xc), shape)
        found = [int(s) if s <= n // 2 else int(s) - n for s, n in zip(peak, shape, strict=True)]
        shift = (found[0], found[1], found[2])
    out = np.zeros(target.shape, np.float32)
    src = tuple(slice(max(0, -s), min(m, n - s)) for s, m, n in zip(shift, mine.shape, target.shape, strict=True))
    dst = tuple(slice(max(0, s), max(0, s) + (sl.stop - sl.start)) for s, sl in zip(shift, src, strict=True))
    out[dst] = mine[src]
    placed = sitk.GetImageFromArray(out)
    placed.CopyInformation(reference)
    return placed, shift


def fuse_to_reference(
    layout: TileLayout,
    load: Callable[[Tile], np.ndarray],
    reference: sitk.Image,
    *,
    log_gain: Callable[[Tile], np.ndarray] | None = None,
    offsets: Offsets | None = None,
    ap_weights: tuple[float, float] | None = None,
    shift: tuple[int, int, int] | None = None,
    seam_profile: SeamProfile | None = None,
    shear: SideShear | None = None,
) -> tuple[sitk.Image, tuple[int, int, int]]:
    """:func:`fuse_mosaic`, taken to ``reference``'s spacing and placed on its grid.

    Parameters
    ----------
    layout, load, log_gain, offsets, ap_weights, seam_profile, shear
        As for :func:`fuse_mosaic`.
    reference : sitk.Image
        Brain whose grid the result takes; isotropic.
    shift : tuple of int or None
        Placement to use instead of searching; see :func:`place_on_reference`.

    Returns
    -------
    tuple
        The placed fusion and its shift (z, y, x).
    """
    mosaic = fuse_mosaic(
        layout,
        load,
        log_gain=log_gain,
        offsets=offsets,
        ap_weights=ap_weights,
        seam_profile=seam_profile,
        shear=shear,
    )
    return place_on_reference(finish_downsampling(mosaic, reference.GetSpacing()[0]), reference, shift=shift)
