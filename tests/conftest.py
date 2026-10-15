"""Synthetic OME-Zarr pyramids built the way AIND light-sheet pyramids are."""

import numpy as np
import pytest
import zarr

# acquisition.json axes for a SmartSPIM-like acquisition: zarr z, y, x run S→I, A→P, L→R.
ND_METADATA = {
    "acquisition": {
        "axes": [
            {"dimension": "0", "name": "Z", "direction": "SUPERIOR_INFERIOR"},
            {"dimension": "1", "name": "Y", "direction": "ANTERIOR_POSTERIOR"},
            {"dimension": "2", "name": "X", "direction": "LEFT_RIGHT"},
        ]
    }
}
BASE_SPACING_UM = (2.0, 1.8, 1.8)  # z, y, x


def block_mean(arr: np.ndarray) -> np.ndarray:
    """Floor of the 2×2×2 block mean over the last three axes, dropping a trailing odd plane."""
    z, y, x = (n // 2 * 2 for n in arr.shape[-3:])
    a = arr[..., :z, :y, :x].astype(np.float64)
    blocks = a.reshape(*a.shape[:-3], z // 2, 2, y // 2, 2, x // 2, 2)
    return np.floor(blocks.mean(axis=(-5, -3, -1))).astype(arr.dtype)


def write_pyramid(path, base: np.ndarray, n_levels: int) -> list[np.ndarray]:
    """Write a 5D (t, c, z, y, x) OME-Zarr v0.4 pyramid in µm and return its levels."""
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    levels = [base[np.newaxis, np.newaxis]]
    for _ in range(1, n_levels):
        levels.append(block_mean(levels[-1]))
    datasets = []
    for k, data in enumerate(levels):
        root.create_array(name=str(k), data=data, chunks=(1, 1, 8, 8, 8))
        scale = [1.0, 1.0, *(s * 2**k for s in BASE_SPACING_UM)]
        datasets.append({"path": str(k), "coordinateTransformations": [{"type": "scale", "scale": scale}]})
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "synthetic",
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
        }
    ]
    return [lvl[0, 0] for lvl in levels]


@pytest.fixture
def nd_metadata():
    """Asset metadata holding the acquisition axes."""
    return ND_METADATA


@pytest.fixture
def pyramid(tmp_path):
    """A 4-level pyramid of a random uint16 volume, with its path and levels."""
    rng = np.random.default_rng(0)
    base = rng.integers(100, 4000, size=(32, 40, 48), dtype=np.uint16)
    path = tmp_path / "channel.ome.zarr"
    return str(path), write_pyramid(path, base, 4)
