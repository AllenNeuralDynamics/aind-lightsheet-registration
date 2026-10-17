"""Prepare a light-sheet brain from its OME-Zarr and register it to a template with ANTs.

The template is fixed and the brain moving, so the transforms map template points to brain points and resample the
brain into the template.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import SimpleITK as sitk
from aind_registration_utils.ants_cli import (
    RegistrationConfig,
    SynRegistrationResult,
    build_registration,
    run_syn_registration,
)

from aind_lightsheet_registration.prepare import (
    ThresholdMethod,
    brain_mask,
    center_of_mass_transform,
    dilate_mask,
    n4_correct,
    normalize_percentiles,
    write_transform,
)
from aind_lightsheet_registration.reader import finish_downsampling, read_level

SIDECAR_NAME = "ls_to_template_transform_information.json"


@dataclass(frozen=True)
class Template:
    """A registration template in mm.

    Attributes
    ----------
    image : Path
        Template image.
    mask : Path
        UInt8 tissue mask on the template's grid.
    """

    image: Path
    mask: Path


@dataclass(frozen=True)
class PreparedBrain:
    """Files written by :func:`prepare_brain`, all in the acquisition frame at the working resolution.

    Attributes
    ----------
    downsampled : Path
        The brain smoothed and resampled to the working resolution.
    corrected : Path
        ``downsampled`` after N4.
    normalized : Path
        ``corrected`` scaled to [0, 1]; the registration's moving image.
    mask : Path
        Tissue mask.
    front_end : Path
        JSON record of how the files were made.
    """

    downsampled: Path
    corrected: Path
    normalized: Path
    mask: Path
    front_end: Path

    @classmethod
    def in_dir(cls, out_dir: Path) -> PreparedBrain:
        """Paths :func:`prepare_brain` writes into ``out_dir``.

        Parameters
        ----------
        out_dir : Path
            Directory given to :func:`prepare_brain`.

        Returns
        -------
        PreparedBrain
            Paths, whether or not they exist yet.
        """
        return cls(
            downsampled=out_dir / "brain_downsampled.nii.gz",
            corrected=out_dir / "brain_n4.nii.gz",
            normalized=out_dir / "brain_normalized.nii.gz",
            mask=out_dir / "brain_mask.nii.gz",
            front_end=out_dir / "front_end.json",
        )


def prepare_brain(
    uri: str,
    nd_metadata: dict[str, Any],
    out_dir: Path,
    *,
    target_spacing_mm: float = 0.025,
    mask_method: ThresholdMethod = "otsu",
    compute_kwargs: dict[str, Any] | None = None,
) -> PreparedBrain:
    """Read a fused channel at ``target_spacing_mm``, mask it, bias-correct it and normalize it.

    Parameters
    ----------
    uri : str
        Path or URI of the fused OME-Zarr channel.
    nd_metadata : dict
        Asset metadata holding ``acquisition``, whose axes give the directions.
    out_dir : Path
        Directory for the outputs; created if missing.
    target_spacing_mm : float
        Isotropic working resolution.
    mask_method : {"otsu", "li", "triangle", "huang"}
        Threshold for :func:`~aind_lightsheet_registration.prepare.brain_mask`.
    compute_kwargs : dict or None
        Passed to dask's ``compute()``, e.g. ``{"num_workers": 16}`` for S3.

    Returns
    -------
    PreparedBrain
        Paths of the written files.
    """
    read = read_level(uri, nd_metadata, target_spacing_mm, compute_kwargs=compute_kwargs)
    record = {
        "uri": uri,
        "level": read.level,
        "level_spacing_mm": list(read.level_spacings[read.level]),
        "frame": "LPS, mm; origin at the center of the first full-resolution voxel",
    }
    return prepare_image(
        finish_downsampling(read.image, target_spacing_mm), out_dir, mask_method=mask_method, record=record
    )


def prepare_image(
    image: sitk.Image,
    out_dir: Path,
    *,
    mask_method: ThresholdMethod = "otsu",
    record: dict[str, Any] | None = None,
) -> PreparedBrain:
    """Mask, bias-correct and normalize a brain already at its working resolution, e.g. a corrected fusion.

    Parameters
    ----------
    image : sitk.Image
        Isotropic brain image; its geometry is kept.
    out_dir : Path
        Directory for the outputs; created if missing.
    mask_method : {"otsu", "li", "triangle", "huang"}
        Threshold for :func:`~aind_lightsheet_registration.prepare.brain_mask`.
    record : dict or None
        Provenance merged into ``front_end.json``.

    Returns
    -------
    PreparedBrain
        Paths of the written files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = PreparedBrain.in_dir(out_dir)
    downsampled = sitk.Cast(image, sitk.sitkFloat32)
    mask = brain_mask(downsampled, method=mask_method)
    corrected = n4_correct(downsampled, mask)
    normalized = normalize_percentiles(corrected, mask)
    for result, path in [
        (downsampled, paths.downsampled),
        (corrected, paths.corrected),
        (normalized, paths.normalized),
        (mask, paths.mask),
    ]:
        sitk.WriteImage(result, str(path))
    spacing = downsampled.GetSpacing()[0]
    full_record = {
        **(record or {}),
        "target_spacing_mm": spacing,
        "origin_mm": list(downsampled.GetOrigin()),
        "direction": list(downsampled.GetDirection()),
        "size": list(downsampled.GetSize()),
        "mask_method": mask_method,
        "mask_log_intensity": True,
        "mask_volume_mm3": float(sitk.GetArrayViewFromImage(mask).sum()) * spacing**3,
    }
    paths.front_end.write_text(json.dumps(full_record, indent=2) + "\n")
    return paths


def register_to_template(
    brain: PreparedBrain,
    template: Template,
    out_dir: Path,
    *,
    config: RegistrationConfig | None = None,
    mask_dilation_mm: float = 0.3,
    threads: int | None = None,
    ants_path: Path | None = None,
) -> SynRegistrationResult:
    """Register a prepared brain to ``template``, starting from their masks' centroids.

    Parameters
    ----------
    brain : PreparedBrain
        Output of :func:`prepare_brain`.
    template : Template
        Fixed image and its mask.
    out_dir : Path
        Directory for the inputs written for ANTs and the transforms.
    config : RegistrationConfig or None
        ANTs settings; ``None`` uses :meth:`RegistrationConfig.lightsheet_25um`.
    mask_dilation_mm : float
        Growth of both masks before SyN, so the tissue boundary is registered.
    threads : int or None
        ITK thread count; ``None`` leaves ITK's default.
    ants_path : Path or None
        Directory holding the ANTs binaries, ahead of ``$ANTSPATH`` and ``PATH``.

    Returns
    -------
    SynRegistrationResult
        Transform paths, the sidecar and the ANTs command.
    """
    if config is None:
        config = RegistrationConfig.lightsheet_25um()
    inputs = out_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    template_image = sitk.ReadImage(str(template.image), sitk.sitkFloat32)
    template_mask = sitk.ReadImage(str(template.mask), sitk.sitkUInt8)
    brain_tissue = sitk.ReadImage(str(brain.mask), sitk.sitkUInt8)
    fixed = inputs / "template_normalized.nii.gz"
    fixed_mask = inputs / "template_mask_dilated.nii.gz"
    moving_mask = inputs / "brain_mask_dilated.nii.gz"
    sitk.WriteImage(normalize_percentiles(template_image, template_mask), str(fixed))
    sitk.WriteImage(dilate_mask(template_mask, mask_dilation_mm), str(fixed_mask))
    sitk.WriteImage(dilate_mask(brain_tissue, mask_dilation_mm), str(moving_mask))
    initial = write_transform(
        center_of_mass_transform(template_mask, brain_tissue), inputs / "initial_center_of_mass.mat"
    )

    reg = build_registration(
        fixed,
        brain.normalized,
        out_dir / "reg_",
        config,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        initial_transform=initial,
        ants_path=ants_path,
    )
    result = run_syn_registration(reg, threads=threads, sidecar_name=SIDECAR_NAME, config=config)
    (out_dir / "inputs.json").write_text(
        json.dumps(
            {
                "template": {k: str(v) for k, v in asdict(template).items()},
                "brain": {k: str(v) for k, v in asdict(brain).items()},
                "mask_dilation_mm": mask_dilation_mm,
            },
            indent=2,
        )
        + "\n"
    )
    return result
