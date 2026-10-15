"""Choosing a pyramid level and the smoothing that finishes the downsampling.

AIND light-sheet pyramids are built by block-averaging each level 2×2×2 from the one below, so every voxel of a
level is a box average over its own extent. A box of width ``w`` has the variance of a Gaussian with
σ = w / √12, which lets a level be brought to any coarser target by adding the Gaussian that makes up the
difference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

BOX_SIGMA_PER_WIDTH = 1 / math.sqrt(12)


def box_sigma(width: float) -> float:
    """Gaussian σ with the variance of a box of ``width``.

    Parameters
    ----------
    width : float
        Box width, e.g. a voxel spacing.

    Returns
    -------
    float
        σ in the same unit.
    """
    return width * BOX_SIGMA_PER_WIDTH


def extra_sigma(level_spacing: Sequence[float], target_spacing: float) -> tuple[float, ...]:
    """Per-axis Gaussian σ that takes a box-filtered level to a ``target_spacing`` box.

    Parameters
    ----------
    level_spacing : sequence of float
        Spacing of the level read, per axis.
    target_spacing : float
        Isotropic spacing to reach, in the same unit.

    Returns
    -------
    tuple of float
        σ per axis, in the same unit.

    Raises
    ------
    ValueError
        If the level is coarser than the target on any axis.
    """
    if any(s > target_spacing for s in level_spacing):
        raise ValueError(f"Level spacing {tuple(level_spacing)} is coarser than the target {target_spacing}")
    return tuple(box_sigma(math.sqrt(target_spacing**2 - s**2)) for s in level_spacing)


def choose_level(level_spacings: Sequence[Sequence[float]], target_spacing: float) -> int:
    """Index of the coarsest level no coarser than ``target_spacing`` on every axis.

    Parameters
    ----------
    level_spacings : sequence of sequence of float
        Spacing of each level, finest first.
    target_spacing : float
        Isotropic spacing to reach, in the same unit.

    Returns
    -------
    int
        Index into ``level_spacings``.

    Raises
    ------
    ValueError
        If even the finest level is coarser than the target.
    """
    candidates = [i for i, spacing in enumerate(level_spacings) if all(s <= target_spacing for s in spacing)]
    if not candidates:
        raise ValueError(f"No level is as fine as {target_spacing}; the finest is {tuple(level_spacings[0])}")
    return max(candidates)


def level_center_offset(base_spacing: Sequence[float], level_spacing: Sequence[float]) -> tuple[float, ...]:
    """Continuous base-level index of a level's first voxel center, per axis.

    A level voxel averages the base voxels beneath it, so its center sits half a block minus half a base voxel
    from the first base voxel's center. Placing every level with this offset keeps them in one physical frame.

    Parameters
    ----------
    base_spacing, level_spacing : sequence of float
        Spacing of the full-resolution level and of the level read, per axis.

    Returns
    -------
    tuple of float
        Offset in base-level voxels, per axis.
    """
    return tuple((level / base - 1) / 2 for base, level in zip(base_spacing, level_spacing, strict=True))
