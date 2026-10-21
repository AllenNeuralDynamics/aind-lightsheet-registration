"""Command line: fetch raw tiles, correct and fuse them, register the fusion, and map annotations to CCF."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import SimpleITK as sitk
import tyro

from aind_lightsheet_registration.annotations import (
    brain_to_ccf_transforms,
    map_to_ccf,
    neuroglancer_points_in_brain,
    write_mapped_csv,
)
from aind_lightsheet_registration.destripe import destripe_stack
from aind_lightsheet_registration.fuse import fuse_to_reference
from aind_lightsheet_registration.illumination import fit_sheet_model
from aind_lightsheet_registration.mirror import mirror_indices
from aind_lightsheet_registration.prepare import Normalization, brain_mask, set_orientation
from aind_lightsheet_registration.register import (
    WORKING_SPACING_MM,
    Template,
    prepare_image,
    register_to_template,
)
from aind_lightsheet_registration.seams import SeamProfile, estimate_seam_profile, interior_seam_starts_mm, seam_windows
from aind_lightsheet_registration.stitch import measure_pairs, nominal_to_stitched_field, solve_offsets, solve_residuals
from aind_lightsheet_registration.tiles import Tile, TileSet, fetch_tiles

DESTRIPED = "destriped"
# the SmartSPIM pipeline's A->P blending: the earlier row's weight across the overlap, measured by fitting its fusion
# to individually placed tiles
PIPELINE_AP_WEIGHTS = (0.62, 0.40)


@dataclass
class Fetch:
    """Download one channel's raw tiles at the level nearest a target resolution, with layout and darkfield."""

    raw_asset: str
    """Raw SmartSPIM asset, e.g. s3://aind-open-data/SmartSPIM_<subject>_<date>."""
    tiles_dir: Path
    """Directory for the tiles."""
    target_spacing_um: float = 25.0
    channel: str = "639"
    channel_dir: str = "Ex_639_Em_667"

    def run(self) -> None:
        """Run the step."""
        ts = fetch_tiles(
            self.raw_asset,
            self.tiles_dir,
            target_spacing_um=self.target_spacing_um,
            channel=self.channel,
            channel_dir=self.channel_dir,
        )
        lay = ts.layout
        print(f"{len(lay.tiles)} tiles, {lay.n_rows} rows x {lay.n_cols} columns, level {lay.level}, {lay.voxel_um} um")


def _destripe_one(args: tuple[Path, str, int, int]) -> str:
    directory, name, level, unprotect = args
    out = directory / f"{name}_{DESTRIPED}.npy"
    if not out.exists():
        np.save(out, destripe_stack(np.load(directory / f"{name}.npy"), level, unprotect_finest=unprotect))
    return name


@dataclass
class Correct:
    """Destripe, stitch, fit the illumination model and fuse, onto a reference brain's grid.

    Writes into ``out_dir``: offsets.json, illumination.npz, fused_raw, mask, fused_nominal (corrected, tiles at
    their stage positions), fused_stitched (corrected, tiles moved by the offsets) and nominal_to_stitched, a point
    field from the annotated pipeline fusion to fused_stitched.
    """

    tiles_dir: Path
    """Directory written by fetch."""
    reference: Path
    """Pipeline fusion at the working resolution, whose grid the outputs take."""
    out_dir: Path
    unprotect_finest: int = 2
    """Finest wavelet levels destriped without foreground protection."""
    workers: int = 12
    annotated_ap_weights: tuple[float, float] = PIPELINE_AP_WEIGHTS
    """A->P blending of the fusion that annotations were made on, for the point field."""
    model: Literal["none", "sheet"] = "none"
    """Illumination model: none, which leaves tile shading in place to match the template, or the full sheet and
    bleaching model."""
    seam_profile: Path | None = None
    """Tile-row seam profile (.tsv) from calibrate-seams, divided out at every seam."""
    restitch: bool = False
    """Measure tile offsets again even if offsets.json exists."""

    def run(self) -> None:
        """Run the step."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = TileSet.open(self.tiles_dir)
        lay = ts.layout
        jobs = [(self.tiles_dir, t.name, lay.level, self.unprotect_finest) for t in lay.tiles]
        with ProcessPoolExecutor(self.workers) as pool:
            for name in pool.map(_destripe_one, jobs):
                print(f"destriped {name}", flush=True)

        def load(t: Tile) -> np.ndarray:
            return ts.load(t, DESTRIPED)

        offsets_path = self.out_dir / "offsets.json"
        if offsets_path.exists() and not self.restitch:
            offsets = {k: (v[0], v[1], v[2]) for k, v in json.loads(offsets_path.read_text()).items()}
            print(f"reusing {offsets_path}", flush=True)
        else:
            pairs = measure_pairs(lay, load)
            for p in pairs:
                flag = " AT SEARCH EDGE" if p.at_edge else ""
                print(
                    f"{p.label:16s} shift (z, y, x) {np.round(p.shift, 2)} corr {p.correlation:.2f}{flag}", flush=True
                )
            offsets = solve_offsets(lay, pairs)
            resid = solve_residuals(offsets, pairs)
            print(f"solve residual median {np.median(resid):.2f}, max {np.max(resid):.2f} voxels", flush=True)
            offsets_path.write_text(json.dumps(offsets, indent=1) + "\n")

        reference = sitk.ReadImage(str(self.reference), sitk.sitkFloat32)
        raw, shift = fuse_to_reference(lay, load, reference)
        print(f"mosaic placed on the reference at voxel shift (z, y, x) {shift}", flush=True)
        mask = brain_mask(raw)
        sitk.WriteImage(raw, str(self.out_dir / "fused_raw.nii.gz"))
        sitk.WriteImage(mask, str(self.out_dir / "mask.nii.gz"))
        log_gain = None
        if self.model == "sheet":
            model = fit_sheet_model(lay, load, offsets, raw, mask, mirror_indices(raw, mask), placement=shift)
            model.save(str(self.out_dir / "illumination.npz"))
            log_gain = model.log_illumination

        seams = SeamProfile.load(self.seam_profile) if self.seam_profile else None
        if seams is not None:
            seams.check_layout(lay)
        nominal, _ = fuse_to_reference(lay, load, reference, log_gain=log_gain, shift=shift, seam_profile=seams)
        sitk.WriteImage(nominal, str(self.out_dir / "fused_nominal.nii.gz"))
        stitched, _ = fuse_to_reference(
            lay, load, reference, log_gain=log_gain, offsets=offsets, shift=shift, seam_profile=seams
        )
        sitk.WriteImage(stitched, str(self.out_dir / "fused_stitched.nii.gz"))
        field = nominal_to_stitched_field(
            lay, offsets, reference, ap_weights=self.annotated_ap_weights, placement=shift
        )
        sitk.WriteImage(field, str(self.out_dir / "nominal_to_stitched.nii.gz"))
        (self.out_dir / "correct.json").write_text(
            json.dumps(
                {
                    "tiles_dir": str(self.tiles_dir),
                    "reference": str(self.reference),
                    "placement_zyx": shift,
                    "unprotect_finest": self.unprotect_finest,
                    "annotated_ap_weights": self.annotated_ap_weights,
                    "model": self.model,
                    "seam_profile": str(self.seam_profile) if self.seam_profile else None,
                },
                indent=1,
            )
            + "\n"
        )


@dataclass
class CalibrateSeams:
    """Estimate the tile-row seam profile from several brains run through correct with the sheet model.

    The ripple is a property of the microscope, so one profile serves every brain imaged the same way; pool at least a
    handful so anatomy at the seams averages out.
    """

    corrected_dirs: list[Path]
    """Output directories of correct, without a seam profile."""
    out: Path
    """Profile to write (.tsv, with a .json sidecar beside it)."""

    def run(self) -> None:
        """Run the step."""
        windows: list[np.ndarray] = []
        spacing, layout = None, None
        for d in self.corrected_dirs:
            record = json.loads((d / "correct.json").read_text())
            if record.get("model") != "sheet" or record.get("seam_profile"):
                raise SystemExit(f"{d}: needs correct with the sheet model and no seam profile")
            lay = TileSet.open(Path(record["tiles_dir"])).layout
            if layout is not None and (lay.step_px, lay.overlap_px, lay.voxel_um) != (
                layout.step_px,
                layout.overlap_px,
                layout.voxel_um,
            ):
                raise SystemExit(f"{d}: tile layout differs from {self.corrected_dirs[0]}")
            layout = lay
            image = sitk.ReadImage(str(d / "fused_stitched.nii.gz"), sitk.sitkFloat32)
            spacing = image.GetSpacing()[0]
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "mask.nii.gz"))) > 0
            starts = interior_seam_starts_mm(lay, record["placement_zyx"][1], spacing)
            found = seam_windows(sitk.GetArrayFromImage(image), mask, spacing, starts)
            print(f"{d}: {len(found)} of {len(starts)} seams usable", flush=True)
            windows += found
        if spacing is None or layout is None or not windows:
            raise SystemExit("no usable seams")
        sources = [json.loads((d / "correct.json").read_text())["tiles_dir"] for d in self.corrected_dirs]
        profile = estimate_seam_profile(
            windows, spacing, layout, {"tile_level": layout.level, "n_seams": len(windows), "sources": sources}
        )
        profile.save(self.out)
        g = profile.log_gain
        print(f"{self.out}: {len(windows)} seams, log gain range [{g.min():+.3f}, {g.max():+.3f}]")


@dataclass
class Register:
    """Mask, N4-correct and normalize a brain image, then register it to a template with ANTs."""

    image: Path
    """Brain at the working resolution, e.g. fused_nominal.nii.gz."""
    out_dir: Path
    template: Path
    """Template at ``working_spacing_mm``."""
    template_mask: Path
    template_normalization: Normalization = "iqr"
    """How the template's intensities are scaled; the brain is normalized to match."""
    template_clip: tuple[float, float] | None = None
    """Clip range for an "iqr" template; defaults to the template's own range."""
    working_spacing_mm: float = WORKING_SPACING_MM
    """Registration grid; the brain is prepared at its own spacing and block-averaged to this."""
    orientation: str | None = None
    """Axis code replacing the image's directions, e.g. "LPS" (SimpleITK convention), when its metadata is wrong."""
    expected_orientation: str | None = None
    """Axis code the acquisition should have; a mismatch warns, the only check that catches a left-right mirror."""
    orientation_check: bool = True
    """Refuse a brain that fits the template far better under another axis rotation of its directions."""
    threads: int | None = None
    ants_path: Path | None = None
    """Directory holding the ANTs binaries."""

    def run(self) -> None:
        """Run the step."""
        image = sitk.ReadImage(str(self.image))
        if self.orientation is not None:
            image = set_orientation(image, self.orientation)
        prepared = prepare_image(
            image,
            self.out_dir / "prepared",
            normalization=self.template_normalization,
            record={"image": str(self.image), "orientation_override": self.orientation},
        )
        result = register_to_template(
            prepared,
            Template(self.template, self.template_mask, self.template_normalization, self.template_clip),
            self.out_dir / "registration",
            working_spacing_mm=self.working_spacing_mm,
            orientation_check=self.orientation_check,
            expected_orientation=self.expected_orientation,
            threads=self.threads,
            ants_path=self.ants_path,
        )
        print(f"registration done in {result.duration_seconds / 60:.1f} min", flush=True)


@dataclass
class MapAnnotations:
    """Map Neuroglancer point annotations on a fused asset to CCF, writing a CSV with each point's structure."""

    neuroglancer: Path
    """Neuroglancer state JSON."""
    zarr_uri: str
    """The fused channel the annotations were made on."""
    nd_metadata: Path
    """The asset's ``metadata.nd.json``."""
    registration_dir: Path
    """Brain -> template registration (reg_ prefix)."""
    template_to_ccf_dir: Path
    """Template -> CCF registration (syn_ prefix)."""
    ccf_labels: Path
    structures_csv: Path
    out_csv: Path
    point_field: list[Path] | None = None
    """Fields applied first, e.g. nominal_to_stitched.nii.gz."""
    ants_path: Path | None = None

    def run(self) -> None:
        """Run the step."""
        import fsspec.config

        # fsspec reads FSSPEC_* variables only at import; public data needs anon explicitly or a stale SSO profile fails
        fsspec.config.conf.setdefault("s3", {}).setdefault("anon", True)
        points = neuroglancer_points_in_brain(
            json.loads(self.neuroglancer.read_text()), self.zarr_uri, json.loads(self.nd_metadata.read_text())
        )
        chain = brain_to_ccf_transforms(self.registration_dir, self.template_to_ccf_dir, point_fields=self.point_field)
        mapped = map_to_ccf(points, chain, self.ccf_labels, self.structures_csv, ants_path=self.ants_path)
        write_mapped_csv(mapped, self.out_csv)
        print(f"{self.out_csv}: {len(mapped)} points in {len(points)} layers")


Command = (
    Annotated[Fetch, tyro.conf.subcommand("fetch")]
    | Annotated[Correct, tyro.conf.subcommand("correct")]
    | Annotated[CalibrateSeams, tyro.conf.subcommand("calibrate-seams")]
    | Annotated[Register, tyro.conf.subcommand("register")]
    | Annotated[MapAnnotations, tyro.conf.subcommand("map-annotations")]
)


def main() -> None:
    """Entry point."""
    tyro.cli(Command).run()
