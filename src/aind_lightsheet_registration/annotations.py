"""Map Neuroglancer point annotations on a fused light-sheet asset into CCF and look up their structures.

Points become level-0 Zarr indices, then LPS mm in the frame :func:`~aind_lightsheet_registration.reader.read_level`
gives (the raw Zarr header's level-0 stub, first voxel's center at the origin). ``antsApplyTransformsToPoints``
then applies, left to right: optional point fields (e.g. nominal → stitched), the inverse brain → template
registration and the inverse template → CCF registration.
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class MappedPoint:
    """One annotation point in brain and CCF space.

    Attributes
    ----------
    layer : str
        Neuroglancer layer name.
    position : int
        1-based position of the point within its layer.
    description : str
        The point's Neuroglancer description, or empty.
    brain_mm : numpy.ndarray
        LPS mm in the brain's acquisition frame.
    ccf_mm : numpy.ndarray
        LPS mm in CCF.
    structure_id : int
        CCF structure; 0 is root (outside the brain), -1 outside the label volume.
    acronym : str
        Structure acronym, or ``"root"``/``"outside"``.
    """

    layer: str
    position: int
    description: str
    brain_mm: np.ndarray
    ccf_mm: np.ndarray
    structure_id: int
    acronym: str


def neuroglancer_points_in_brain(
    neuroglancer: dict[str, Any], zarr_uri: str, nd_metadata: dict[str, Any]
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Annotation points of every layer in LPS mm, with their descriptions.

    Parameters
    ----------
    neuroglancer : dict
        Parsed Neuroglancer state.
    zarr_uri : str
        The fused channel the annotations were made on.
    nd_metadata : dict
        Asset metadata holding ``acquisition``.

    Returns
    -------
    dict
        Layer → ((n, 3) LPS mm, n descriptions).
    """
    from aind_zarr_utils.annotations import annotation_indices_to_anatomical
    from aind_zarr_utils.points import Points
    from aind_zarr_utils.zarr import zarr_to_sitk_stub

    stub, _ = zarr_to_sitk_stub(zarr_uri, nd_metadata, level=0)
    points = Points.from_neuroglancer(neuroglancer)
    brain = annotation_indices_to_anatomical(stub, dict(points.values))
    descriptions = points.descriptions or {}
    return {
        layer: (np.asarray(xyz), [d or "" for d in (descriptions.get(layer) or [None] * len(xyz))])
        for layer, xyz in brain.items()
    }


def transform_points(points_mm: np.ndarray, transforms: list[str], *, ants_path: Path | None = None) -> np.ndarray:
    """Apply ANTs transforms to LPS mm points, left to right.

    Parameters
    ----------
    points_mm : numpy.ndarray
        (n, 3) points.
    transforms : list of str
        ``antsApplyTransformsToPoints`` ``-t`` arguments, e.g. ``"[reg_0GenericAffine.mat,1]"``.
    ants_path : Path or None
        Directory holding the ANTs binaries; ``None`` uses ``PATH``.

    Returns
    -------
    numpy.ndarray
        (n, 3) transformed points.
    """
    exe = str(ants_path / "antsApplyTransformsToPoints") if ants_path else "antsApplyTransformsToPoints"
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        np.savetxt(src, np.c_[points_mm, np.zeros(len(points_mm))], delimiter=",", header="x,y,z,t", comments="")
        argv = [exe, "-d", "3", "-i", str(src), "-o", str(dst)]
        for t in transforms:
            argv += ["-t", t]
        subprocess.run(argv, check=True, capture_output=True)
        return np.loadtxt(dst, delimiter=",", skiprows=1, ndmin=2)[:, :3]


def brain_to_ccf_transforms(
    registration_dir: Path, template_to_ccf_dir: Path, *, point_fields: list[Path] | None = None
) -> list[str]:
    """Build the point chain from brain mm to CCF mm.

    Both registrations have the brain-side image moving, so their forward transforms map template points to brain
    points (and CCF points to template points); brain → CCF is their inverse, affine first.

    Parameters
    ----------
    registration_dir : Path
        Brain → template registration with ``reg_0GenericAffine.mat`` and ``reg_1InverseWarp.nii.gz``.
    template_to_ccf_dir : Path
        Template → CCF registration with ``syn_0GenericAffine.mat`` and ``syn_1InverseWarp.nii.gz``.
    point_fields : list of Path or None
        Displacement fields applied first, e.g. nominal → stitched.

    Returns
    -------
    list of str
        ``-t`` arguments in application order.
    """
    return [
        *(str(p) for p in point_fields or []),
        f"[{registration_dir / 'reg_0GenericAffine.mat'},1]",
        str(registration_dir / "reg_1InverseWarp.nii.gz"),
        f"[{template_to_ccf_dir / 'syn_0GenericAffine.mat'},1]",
        str(template_to_ccf_dir / "syn_1InverseWarp.nii.gz"),
    ]


def map_to_ccf(
    points: dict[str, tuple[np.ndarray, list[str]]],
    transforms: list[str],
    ccf_labels: Path,
    structures_csv: Path,
    *,
    ants_path: Path | None = None,
) -> list[MappedPoint]:
    """Map points to CCF and look up the structure under each.

    Parameters
    ----------
    points : dict
        From :func:`neuroglancer_points_in_brain`.
    transforms : list of str
        From :func:`brain_to_ccf_transforms`.
    ccf_labels : Path
        CCF annotation volume in LPS mm.
    structures_csv : Path
        CCF structure table with ``id`` and ``acronym`` columns.
    ants_path : Path or None
        Directory holding the ANTs binaries.

    Returns
    -------
    list of MappedPoint
        In layer order.
    """
    layers = list(points)
    brain = np.vstack([points[k][0] for k in layers])
    ccf = transform_points(brain, transforms, ants_path=ants_path)
    labels = sitk.ReadImage(str(ccf_labels))
    label_array = sitk.GetArrayViewFromImage(labels)
    with open(structures_csv, newline="") as fh:
        acronyms = {int(r["id"]): r["acronym"] for r in csv.DictReader(fh)}

    def structure(p: np.ndarray) -> int:
        i, j, k = labels.TransformPhysicalPointToIndex([float(v) for v in p])
        if not all(0 <= a < n for a, n in zip((i, j, k), labels.GetSize(), strict=True)):
            return -1
        return int(label_array[k, j, i])

    out, row = [], 0
    for layer in layers:
        xyz, descriptions = points[layer]
        for n, (p, desc) in enumerate(zip(xyz, descriptions, strict=True)):
            sid = structure(ccf[row])
            name = acronyms.get(sid, "outside" if sid == -1 else "root" if sid == 0 else "?")
            out.append(MappedPoint(layer, n + 1, desc, p, ccf[row], sid, name))
            row += 1
    return out


def write_mapped_csv(points: list[MappedPoint], path: Path) -> None:
    """Write mapped points as CSV: layer, point, description, brain and CCF LPS mm, structure id and acronym."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["layer", "point", "description", "brain_x_mm", "brain_y_mm", "brain_z_mm"]
            + ["ccf_x_mm", "ccf_y_mm", "ccf_z_mm", "structure_id", "acronym"]
        )
        for p in points:
            w.writerow(
                [p.layer, p.position, p.description, *np.round(p.brain_mm, 5), *np.round(p.ccf_mm, 5)]
                + [p.structure_id, p.acronym]
            )
