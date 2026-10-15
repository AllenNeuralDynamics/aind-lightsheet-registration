"""Read an AIND light-sheet OME-Zarr at a working resolution, in its acquisition frame.

The frame comes from the acquisition metadata's axis directions, in mm, with the center of the first
full-resolution voxel at the origin. Every level is placed in that same frame, so a full-resolution voxel index maps
to the same physical point whichever level was read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import SimpleITK as sitk
from aind_zarr_utils.zarr import zarr_to_sitk, zarr_to_sitk_stub
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from aind_lightsheet_registration.pyramid import choose_level, extra_sigma, level_center_offset

Spacing = tuple[float, float, float]


@dataclass(frozen=True)
class LevelRead:
    """A pyramid level read for a target resolution.

    Attributes
    ----------
    image : sitk.Image
        The level, in mm, in the acquisition frame.
    level : int
        Zarr dataset index read; 0 is full resolution.
    level_spacings : tuple of Spacing
        Spacing of every level in mm, in SimpleITK axis order, finest first.
    """

    image: sitk.Image
    level: int
    level_spacings: tuple[Spacing, ...]


def read_level(
    uri: str,
    nd_metadata: dict[str, Any],
    target_spacing_mm: float,
    *,
    compute_kwargs: dict[str, Any] | None = None,
) -> LevelRead:
    """Read the coarsest level no coarser than ``target_spacing_mm`` on any axis.

    Parameters
    ----------
    uri : str
        Path or URI of the fused OME-Zarr channel.
    nd_metadata : dict
        Asset metadata holding ``acquisition``, whose axes give the directions.
    target_spacing_mm : float
        Working resolution the level will be brought to.
    compute_kwargs : dict or None
        Passed to dask's ``compute()``, e.g. ``{"num_workers": 16}`` for S3.

    Returns
    -------
    LevelRead
        The level and the spacings it was chosen from.
    """
    node = next(iter(Reader(parse_url(uri))()))
    opened = (node, node.metadata)
    stubs = [zarr_to_sitk_stub(uri, nd_metadata, level=k, opened_zarr=opened)[0] for k in range(len(node.data))]
    spacings = tuple(_spacing(stub) for stub in stubs)
    level = choose_level(spacings, target_spacing_mm)
    offset = level_center_offset(spacings[0], spacings[level])
    origin = stubs[0].TransformContinuousIndexToPhysicalPoint(offset)
    image = zarr_to_sitk(
        uri, nd_metadata, level=level, set_origin=origin, opened_zarr=opened, compute_kwargs=compute_kwargs
    )
    return LevelRead(image=image, level=level, level_spacings=spacings)


def smooth_to_box(image: sitk.Image, target_spacing_mm: float) -> sitk.Image:
    """Add the Gaussian that takes a box-filtered level to a ``target_spacing_mm`` box, on the level's grid.

    Parameters
    ----------
    image : sitk.Image
        A pyramid level, e.g. :attr:`LevelRead.image`.
    target_spacing_mm : float
        Isotropic spacing the result will be resampled to.

    Returns
    -------
    sitk.Image
        Float32 image with the input's geometry.
    """
    smoothed: sitk.Image = sitk.Cast(image, sitk.sitkFloat32)
    sigma = extra_sigma(_spacing(image), target_spacing_mm)
    if not any(s > 0 for s in sigma):
        return smoothed
    # σ here is about a third of a voxel. ITK's discrete Gaussian keeps the requested variance there only with a
    # tight truncation error (its default of 0.01 loses 11-15%); the recursive Gaussian loses over 70%.
    return sitk.DiscreteGaussian(
        smoothed, variance=[s**2 for s in sigma], maximumKernelWidth=64, maximumError=1e-5, useImageSpacing=True
    )


def finish_downsampling(image: sitk.Image, target_spacing_mm: float) -> sitk.Image:
    """Smooth a box-filtered level to a ``target_spacing_mm`` box and resample it isotropically.

    The output keeps the input's direction and outer corner, so it stays in the acquisition frame.

    Parameters
    ----------
    image : sitk.Image
        A pyramid level, e.g. :attr:`LevelRead.image`.
    target_spacing_mm : float
        Isotropic output spacing.

    Returns
    -------
    sitk.Image
        Float32 image at ``target_spacing_mm``.
    """
    spacing = _spacing(image)
    size = [max(1, round(n * s / target_spacing_mm)) for n, s in zip(image.GetSize(), spacing, strict=True)]
    first_center = [-0.5 + 0.5 * target_spacing_mm / s for s in spacing]
    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetSpacing([target_spacing_mm] * 3)
    reference.SetDirection(image.GetDirection())
    reference.SetOrigin(image.TransformContinuousIndexToPhysicalPoint(first_center))
    smoothed = smooth_to_box(image, target_spacing_mm)
    return sitk.Resample(smoothed, reference, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def _spacing(image: sitk.Image) -> Spacing:
    sx, sy, sz = image.GetSpacing()
    return (sx, sy, sz)
