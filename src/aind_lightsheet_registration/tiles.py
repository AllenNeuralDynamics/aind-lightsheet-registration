"""SmartSPIM raw tiles at one pyramid level, with the grid and exposure history the microscope metadata gives.

The microscope's ``metadata.json`` lists every stack in acquisition order with its stage position (0.1 µm) and the
laser side that lit it; its ``session_config`` gives the pixel size and z step. Tile ``row`` and ``col`` count from the
smallest stage Y (anterior) and X (left), and are 0-based indices.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

STAGE_UNIT_UM = 0.1
TILE_LIST_NAME = "metadata.json"
DARKFIELD_NAME = "DarkMaster_cropped.tif"
TILE_SETTINGS_NAME = "TileSettings.ini"  # focus-ramp planes; absent from older assets
LAYOUT_NAME = "layout.json"


@dataclass(frozen=True)
class Tile:
    """One stack of the imaged channel.

    Attributes
    ----------
    name : str
        ``"<X>_<Y>"`` stage position, as the raw asset names the tile's Zarr.
    row, col : int
        Grid indices along A→P and L→R.
    side : int
        Illumination side: 0 left laser, 1 right.
    """

    name: str
    row: int
    col: int
    side: int


@dataclass(frozen=True)
class TileLayout:
    """The tile grid of one channel at one pyramid level.

    Attributes
    ----------
    tiles : tuple of Tile
        In acquisition order.
    level : int
        Pyramid level; 0 is full resolution.
    voxel_um : tuple of float
        Voxel size (z, y, x) at ``level``.
    shape : tuple of int
        Stack shape (z, y, x) at ``level``.
    step_px : tuple of int
        Tile pitch (y, x) in voxels at ``level``.
    relative_power : dict
        ``(laser, side)`` → that laser's power on that side over the channel's left-side power.
    channel : str
        Excitation wavelength of the imaged channel.
    """

    tiles: tuple[Tile, ...]
    level: int
    voxel_um: tuple[float, float, float]
    shape: tuple[int, int, int]
    step_px: tuple[int, int]
    relative_power: dict[tuple[str, int], float]
    channel: str = "639"

    @classmethod
    def from_metadata(
        cls, metadata: dict[str, Any], *, level: int, shape: tuple[int, int, int], channel: str = "639"
    ) -> TileLayout:
        """Grid of ``channel``'s tiles from the microscope's ``metadata.json``.

        Parameters
        ----------
        metadata : dict
            Parsed ``metadata.json``.
        level : int
            Pyramid level the tiles are read at; each level halves every axis.
        shape : tuple of int
            Stack shape (z, y, x) at ``level``.
        channel : str
            Excitation wavelength.

        Returns
        -------
        TileLayout
            The grid.
        """
        stacks = [t for t in metadata["tile_config"].values() if t["Laser"] == channel]
        xs = sorted({int(t["X"]) for t in stacks})
        ys = sorted({int(t["Y"]) for t in stacks})
        tiles = tuple(
            Tile(name=f"{t['X']}_{t['Y']}", row=ys.index(int(t["Y"])), col=xs.index(int(t["X"])), side=int(t["Side"]))
            for t in stacks
        )
        session = metadata["session_config"]
        factor = 2**level
        voxel_um = (float(session["Z step (um)"]) * factor, *(float(session["um/pix"]) * factor,) * 2)
        step_px = (_pitch_px(ys, voxel_um[1]), _pitch_px(xs, voxel_um[2]))
        wavelengths = metadata["wavelength_config"]
        reference = float(wavelengths[channel]["power_left"])
        relative_power = {
            (laser, side): float(powers["power_left" if side == 0 else "power_right"]) / reference
            for laser, powers in wavelengths.items()
            for side in (0, 1)
        }
        return cls(
            tiles=tiles,
            level=level,
            voxel_um=voxel_um,
            shape=shape,
            step_px=step_px,
            relative_power=relative_power,
            channel=channel,
        )

    @property
    def n_rows(self) -> int:
        """Rows of tiles along A→P."""
        return 1 + max(t.row for t in self.tiles)

    @property
    def n_cols(self) -> int:
        """Columns of tiles along L→R."""
        return 1 + max(t.col for t in self.tiles)

    @property
    def overlap_px(self) -> tuple[int, int]:
        """Overlap (y, x) of neighboring tiles, in voxels."""
        return self.shape[1] - self.step_px[0], self.shape[2] - self.step_px[1]

    @property
    def mosaic_shape(self) -> tuple[int, int, int]:
        """Shape (z, y, x) of the grid with tiles at their nominal positions."""
        nz, ny, nx = self.shape
        return nz, self.step_px[0] * (self.n_rows - 1) + ny, self.step_px[1] * (self.n_cols - 1) + nx

    def at(self, row: int, col: int) -> Tile | None:
        """Find the tile at grid position (``row``, ``col``); None if none was imaged there."""
        return next((t for t in self.tiles if (t.row, t.col) == (row, col)), None)

    def neighbor_pairs(self) -> Iterator[tuple[str, Tile, Tile]]:
        """Overlapping pairs as ``(kind, earlier, later)``, kind ``"AP"`` or ``"LR"``; earlier is anterior or left."""
        for a in self.tiles:
            for kind, (dr, dc) in (("AP", (1, 0)), ("LR", (0, 1))):
                b = self.at(a.row + dr, a.col + dc)
                if b is not None:
                    yield kind, a, b

    def to_json(self) -> str:
        """Serialize for :meth:`from_json`."""
        record = {
            "tiles": [vars(t) for t in self.tiles],
            "level": self.level,
            "voxel_um": self.voxel_um,
            "shape": self.shape,
            "step_px": self.step_px,
            "relative_power": [[laser, side, p] for (laser, side), p in self.relative_power.items()],
            "channel": self.channel,
        }
        return json.dumps(record, indent=1) + "\n"

    @classmethod
    def from_json(cls, text: str) -> TileLayout:
        """Read what :meth:`to_json` wrote.

        Parameters
        ----------
        text : str
            JSON text.

        Returns
        -------
        TileLayout
            The grid.
        """
        r = json.loads(text)
        return cls(
            tiles=tuple(Tile(**t) for t in r["tiles"]),
            level=r["level"],
            voxel_um=tuple(r["voxel_um"]),
            shape=tuple(r["shape"]),
            step_px=tuple(r["step_px"]),
            relative_power={(laser, side): p for laser, side, p in r["relative_power"]},
            channel=r["channel"],
        )


def _pitch_px(stage_coords: list[int], voxel_um: float) -> int:
    if len(stage_coords) < 2:
        return 0
    steps = np.diff(stage_coords) * STAGE_UNIT_UM / voxel_um
    pitch = round(float(np.median(steps)))
    if np.abs(steps - pitch).max() > 0.05:
        raise ValueError(f"tile pitch is not a whole number of voxels: {steps}")
    return pitch


def choose_tile_level(metadata: dict[str, Any], target_spacing_um: float) -> int:
    """Coarsest level whose in-plane voxel is no larger than ``target_spacing_um``.

    Parameters
    ----------
    metadata : dict
        Parsed ``metadata.json``.
    target_spacing_um : float
        Resolution the fusion will be taken to.

    Returns
    -------
    int
        Pyramid level.
    """
    pixel_um = float(metadata["session_config"]["um/pix"])
    level = 0
    while pixel_um * 2 ** (level + 1) <= target_spacing_um:
        level += 1
    return level


@dataclass(frozen=True)
class TileSet:
    """Tiles of one channel saved as ``.npy`` in a directory, with their layout and darkfield.

    Attributes
    ----------
    directory : Path
        Holds ``layout.json``, ``darkfield.json`` and ``<tile>.npy`` per tile.
    layout : TileLayout
        The grid.
    darkfield : float
        Camera offset subtracted from every voxel.
    """

    directory: Path
    layout: TileLayout
    darkfield: float

    @classmethod
    def open(cls, directory: Path) -> TileSet:
        """Open a directory written by :func:`fetch_tiles`.

        Parameters
        ----------
        directory : Path
            Tile directory.

        Returns
        -------
        TileSet
            The tiles.
        """
        layout = TileLayout.from_json((directory / LAYOUT_NAME).read_text())
        darkfield = json.loads((directory / "darkfield.json").read_text())["darkfield"]
        return cls(directory=directory, layout=layout, darkfield=darkfield)

    def path(self, tile: Tile, variant: str = "") -> Path:
        """File of ``tile``; ``variant`` names a processed copy, e.g. ``"destriped"``."""
        return self.directory / f"{tile.name}{'_' + variant if variant else ''}.npy"

    def load(self, tile: Tile, variant: str = "", *, subtract_dark: bool = True) -> np.ndarray:
        """Float32 stack of ``tile``, dark-subtracted and clipped at 0 unless ``subtract_dark`` is False.

        Parameters
        ----------
        tile : Tile
            Which tile.
        variant : str
            Processed copy to read; empty for the raw tile.
        subtract_dark : bool
            Subtract the darkfield and clip at 0.

        Returns
        -------
        numpy.ndarray
            (z, y, x) stack.
        """
        stack = np.load(self.path(tile, variant)).astype(np.float32)
        if subtract_dark:
            np.clip(stack - np.float32(self.darkfield), 0, None, out=stack)
        return np.asarray(stack)


def fetch_tiles(
    raw_asset_uri: str,
    out_dir: Path,
    *,
    target_spacing_um: float = 25.0,
    channel: str = "639",
    channel_dir: str = "Ex_639_Em_667",
    workers: int = 8,
) -> TileSet:
    """Download every tile of ``channel`` at the level :func:`choose_tile_level` picks, plus its layout and darkfield.

    The focus-ramp settings come too where the asset has them. Tiles already in ``out_dir`` are kept. S3 is read
    anonymously.

    Parameters
    ----------
    raw_asset_uri : str
        Raw SmartSPIM asset, e.g. ``s3://aind-open-data/SmartSPIM_<subject>_<date>``.
    out_dir : Path
        Tile directory; created if missing.
    target_spacing_um : float
        Resolution the fusion will be taken to.
    channel : str
        Excitation wavelength.
    channel_dir : str
        The channel's directory under ``SPIM/``.
    workers : int
        Concurrent tile downloads.

    Returns
    -------
    TileSet
        The tiles.
    """
    import fsspec
    import zarr

    out_dir.mkdir(parents=True, exist_ok=True)
    derivatives = f"{raw_asset_uri.rstrip('/')}/SPIM/derivatives"
    storage: dict[str, Any] = {"anon": True} if raw_asset_uri.startswith("s3://") else {}
    for name in (TILE_LIST_NAME, DARKFIELD_NAME, TILE_SETTINGS_NAME):
        if not (out_dir / name).exists():
            fs, path = fsspec.core.url_to_fs(f"{derivatives}/{name}", **storage)
            if name == TILE_SETTINGS_NAME and not fs.exists(path):
                continue
            (out_dir / name).write_bytes(fs.cat_file(path))
    metadata = json.loads((out_dir / TILE_LIST_NAME).read_text())
    level = choose_tile_level(metadata, target_spacing_um)
    names = sorted({f"{t['X']}_{t['Y']}" for t in metadata["tile_config"].values() if t["Laser"] == channel})

    def get(name: str) -> tuple[int, ...]:
        path = out_dir / f"{name}.npy"
        if not path.exists():
            uri = f"{raw_asset_uri.rstrip('/')}/SPIM/{channel_dir}/{name}.ome.zarr/{level}"
            array = zarr.open_array(fsspec.get_mapper(uri, **storage), mode="r")
            np.save(path, np.asarray(array[0, 0]))
        return tuple(np.load(path, mmap_mode="r").shape)

    with ThreadPoolExecutor(workers) as pool:
        shapes = set(pool.map(get, names))
    if len(shapes) != 1:
        raise ValueError(f"tiles differ in shape: {shapes}")
    nz, ny, nx = shapes.pop()
    layout = TileLayout.from_metadata(metadata, level=level, shape=(nz, ny, nx), channel=channel)
    dark = sitk.GetArrayFromImage(sitk.ReadImage(str(out_dir / DARKFIELD_NAME)))
    darkfield = float(np.median(dark))
    (out_dir / LAYOUT_NAME).write_text(layout.to_json())
    (out_dir / "darkfield.json").write_text(json.dumps({"darkfield": darkfield, "source": DARKFIELD_NAME}) + "\n")
    return TileSet(directory=out_dir, layout=layout, darkfield=darkfield)
