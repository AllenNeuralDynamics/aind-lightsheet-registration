"""Move images from the SmartSPIM pipeline's registration frame into the stitched Zarr's acquisition frame.

The pipeline registers a copy of the fusion whose header it rewrote (``prep_n4bias.nii.gz`` and anything drawn on
it), while this package works in the frame :mod:`~aind_lightsheet_registration.reader` reads. Both frames place the
same full-resolution voxel grid, so one affine maps either onto the other and voxels never need resampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import SimpleITK as sitk
from aind_zarr_utils.pipeline_transformed import base_and_pipeline_anatomical_stub


def _index_to_physical(image: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    direction = np.array(image.GetDirection()).reshape(3, 3)
    return direction @ np.diag(image.GetSpacing()), np.array(image.GetOrigin())


def pipeline_to_zarr_transform(
    zarr_uri: str, nd_metadata: dict[str, Any], processing: dict[str, Any]
) -> sitk.AffineTransform:
    """Return the point map from a fusion's pipeline frame to its Zarr frame.

    Parameters
    ----------
    zarr_uri : str
        Path or URI of the fused OME-Zarr channel the pipeline registered.
    nd_metadata : dict
        Asset metadata holding ``acquisition``.
    processing : dict
        The stitched asset's ``processing.json``, whose pipeline version selects the header the pipeline wrote.

    Returns
    -------
    sitk.AffineTransform
        Maps a pipeline-frame point to the Zarr-frame point of the same voxel.
    """
    zarr_stub, pipeline_stub, _ = base_and_pipeline_anatomical_stub(zarr_uri, nd_metadata, processing)
    zarr_matrix, zarr_origin = _index_to_physical(zarr_stub)
    pipeline_matrix, pipeline_origin = _index_to_physical(pipeline_stub)
    matrix = zarr_matrix @ np.linalg.inv(pipeline_matrix)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix.ravel().tolist())
    transform.SetTranslation((zarr_origin - matrix @ pipeline_origin).tolist())
    return transform


def move_image(image: sitk.Image, transform: sitk.AffineTransform) -> sitk.Image:
    """Rewrite an image's header so its voxels sit where ``transform`` sends them; the voxels are untouched.

    Parameters
    ----------
    image : sitk.Image
        Image in the transform's source frame, e.g. a mask drawn on the pipeline's prepared image.
    transform : sitk.AffineTransform
        Point map between frames, from :func:`pipeline_to_zarr_transform`.

    Returns
    -------
    sitk.Image
        The same voxels with the header of the target frame.

    Raises
    ------
    ValueError
        If the map shears the voxel axes, which a header cannot express.
    """
    matrix = np.array(transform.GetMatrix()).reshape(3, 3)
    index_matrix, origin = _index_to_physical(image)
    moved = matrix @ index_matrix
    spacing = np.linalg.norm(moved, axis=0)
    direction = moved / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6):
        raise ValueError("the frame map shears the voxel axes; resample instead")
    out = sitk.Image(image)
    out.SetSpacing(spacing.tolist())
    out.SetDirection(direction.ravel().tolist())
    out.SetOrigin(transform.TransformPoint(origin.tolist()))
    return out
