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
    LinearConfig,
    RegistrationConfig,
    SynConfig,
    SynRegistrationResult,
    build_registration,
    run_syn_registration,
)

from aind_lightsheet_registration.prepare import (
    Normalization,
    ThresholdMethod,
    block_average,
    block_average_mask,
    brain_mask,
    center_of_mass_transform,
    dilate_mask,
    n4_correct,
    normalize_iqr,
    normalize_percentiles,
    write_transform,
)
from aind_lightsheet_registration.reader import finish_downsampling, read_level

SIDECAR_NAME = "ls_to_template_transform_information.json"
# working resolution of :func:`template_build_config`; its update-field smoothing is in voxels at this spacing
TEMPLATE_BUILD_SPACING_MM = 0.05


def template_build_config() -> RegistrationConfig:
    """ANTs settings templates are built and registered with, for brains and templates at 50 µm.

    SyN smooths each update with sigma 2 voxels (0.1 mm; ANTs takes the variance in voxels squared) and the total
    field not at all, matches over a CC radius of 4 voxels (0.2 mm), and starts from a 1.2 mm level. Narrower
    smoothing and windows, as on a 25 µm grid, let SyN fit noise and residual shading, which shows as a mottled log
    Jacobian.
    """
    return RegistrationConfig(
        linear=LinearConfig(
            iterations=[1000, 500, 250],
            shrink_factors=[4, 2, 1],
            smoothing_sigmas=[0.1, 0.05, 0.0],
            smoothing_units="mm",
        ),
        syn=SynConfig(
            gradient_step=0.2,
            update_field_variance=4.0,
            total_field_variance=0.0,
            metric="CC",
            radius_or_bins=4,
            iterations=[100, 100, 100, 70, 40, 40, 20],
            shrink_factors=[24, 16, 12, 8, 4, 2, 1],
            smoothing_sigmas=[0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0],
            smoothing_units="mm",
        ),
        interpolation="Linear",
        random_seed=1,
    )


@dataclass(frozen=True)
class Template:
    """A registration template in mm.

    Attributes
    ----------
    image : Path
        Template image.
    mask : Path
        UInt8 tissue mask on the template's grid.
    normalization : {"iqr", "percentile"}
        How the template's intensities are scaled, and so how a brain is normalized to match. An "iqr" template
        is used as stored; a "percentile" template is rescaled by in-mask percentiles like the brain.
    clip : tuple of float or None
        Range an "iqr" brain is clipped to, so CSF and background land on the template's values; ``None`` uses
        the template's own minimum and maximum.
    """

    image: Path
    mask: Path
    normalization: Normalization = "iqr"
    clip: tuple[float, float] | None = None


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
        ``corrected`` normalized to the template's convention; the registration's moving image.
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
    normalization: Normalization = "iqr",
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
    normalization : {"iqr", "percentile"}
        The convention of the template the brain will be registered to.
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
        finish_downsampling(read.image, target_spacing_mm),
        out_dir,
        mask_method=mask_method,
        normalization=normalization,
        record=record,
    )


def prepare_image(
    image: sitk.Image,
    out_dir: Path,
    *,
    mask_method: ThresholdMethod = "otsu",
    normalization: Normalization = "iqr",
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
    normalization : {"iqr", "percentile"}
        The convention of the template the brain will be registered to.
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
    normalized = normalize_iqr(corrected, mask) if normalization == "iqr" else normalize_percentiles(corrected, mask)
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
        "normalization": normalization,
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
    working_spacing_mm: float = TEMPLATE_BUILD_SPACING_MM,
    mask_dilation_mm: float = 0.4,
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
        ANTs settings; ``None`` uses :func:`template_build_config`.
    working_spacing_mm : float
        Spacing the brain is block-averaged to before registering; a whole multiple of its own, and the template's.
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
        config = template_build_config()
    fixed, moving, fixed_mask, moving_mask, initial = write_registration_inputs(
        brain, template, out_dir / "inputs", working_spacing_mm=working_spacing_mm, mask_dilation_mm=mask_dilation_mm
    )
    reg = build_registration(
        fixed,
        moving,
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
                "template": {k: v if k == "clip" else str(v) for k, v in asdict(template).items()},
                "brain": {k: str(v) for k, v in asdict(brain).items()},
                "mask_dilation_mm": mask_dilation_mm,
                "working_spacing_mm": working_spacing_mm,
            },
            indent=2,
        )
        + "\n"
    )
    return result


def write_registration_inputs(
    brain: PreparedBrain,
    template: Template,
    inputs: Path,
    *,
    working_spacing_mm: float = TEMPLATE_BUILD_SPACING_MM,
    mask_dilation_mm: float = 0.4,
) -> tuple[Path, Path, Path, Path, Path]:
    """Write the images, masks and initial transform ANTs registers, after checking brain and template agree.

    Returns
    -------
    tuple of Path
        Fixed image, moving image, fixed mask, moving mask and initial transform.
    """
    inputs.mkdir(parents=True, exist_ok=True)
    template_image = sitk.ReadImage(str(template.image), sitk.sitkFloat32)
    template_mask = sitk.ReadImage(str(template.mask), sitk.sitkUInt8)
    brain_image = sitk.ReadImage(str(brain.normalized), sitk.sitkFloat32)
    brain_tissue = sitk.ReadImage(str(brain.mask), sitk.sitkUInt8)
    factor = _block_factor(brain_image.GetSpacing(), working_spacing_mm)
    if not all(abs(s - working_spacing_mm) < 1e-4 for s in template_image.GetSpacing()):
        raise ValueError(f"template spacing {template_image.GetSpacing()} mm is not {working_spacing_mm} mm")
    if brain.front_end.exists():
        prepared_as = json.loads(brain.front_end.read_text()).get("normalization", "percentile")
        if prepared_as != template.normalization:
            raise ValueError(f"brain normalized as {prepared_as!r}, template is {template.normalization!r}")
    fixed_mask = inputs / "template_mask_dilated.nii.gz"
    moving = inputs / "brain_normalized_working.nii.gz"
    moving_mask = inputs / "brain_mask_dilated.nii.gz"
    if template.normalization == "iqr":
        fixed = template.image
        brain_image = _clip(brain_image, template.clip or _range(template_image))
    else:
        fixed = inputs / "template_normalized.nii.gz"
        sitk.WriteImage(normalize_percentiles(template_image, template_mask), str(fixed))
    sitk.WriteImage(block_average(brain_image, factor), str(moving))
    brain_tissue = block_average_mask(brain_tissue, factor)
    sitk.WriteImage(dilate_mask(template_mask, mask_dilation_mm), str(fixed_mask))
    sitk.WriteImage(dilate_mask(brain_tissue, mask_dilation_mm), str(moving_mask))
    initial = write_transform(
        center_of_mass_transform(template_mask, brain_tissue), inputs / "initial_center_of_mass.mat"
    )
    return fixed, moving, fixed_mask, moving_mask, initial


def _block_factor(spacing: tuple[float, ...], working_spacing_mm: float) -> int:
    ratios = [working_spacing_mm / s for s in spacing]
    factor = round(ratios[0])
    if factor < 1 or any(abs(r - factor) > 1e-3 for r in ratios):
        raise ValueError(f"working spacing {working_spacing_mm} mm is not a whole multiple of {spacing} mm")
    return factor


def _range(image: sitk.Image) -> tuple[float, float]:
    values = sitk.GetArrayViewFromImage(image)
    return float(values.min()), float(values.max())


def _clip(image: sitk.Image, bounds: tuple[float, float]) -> sitk.Image:
    clipped: sitk.Image = sitk.Clamp(image, lowerBound=bounds[0], upperBound=bounds[1])
    return clipped
