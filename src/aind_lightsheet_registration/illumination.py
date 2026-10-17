"""Fit a light-sheet illumination and bleaching model to one acquisition, and give each tile's log illumination.

A tile's dark-subtracted intensity is tissue signal times

    M_t(u, v) = A_side(u) * h_side(v; u) * G_side * exp(-k * dose_t(u, v))

with u along the beam (L→R) and v across it (A→P), both in mm from the field's center. ``A_side`` is quadratic in
log. ``h_side`` is a Gaussian clipped by a soft-edged aperture of half-height H, centered on a line tilted along the
beam. ``G_side`` is the right/left gain. ``dose`` sums every earlier stack of the channel lit through the same sheet,
at its relative power, so each tile is bleached by its neighbors' exposures. A radial vignetting term is left out:
it trades against the quadratic in ``A_side``.

Three kinds of data constrain it: intensity ratios across tile overlaps, ratios between voxels and their mirror
across the midline (which differ in which side lit them), and each side's median A→P profile within a tile.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import shift as nd_shift
from scipy.optimize import least_squares
from scipy.special import erf

from aind_lightsheet_registration.stitch import Offsets
from aind_lightsheet_registration.tiles import Tile, TileLayout

SHEET_PARAMS = ("a1", "a2", "y0", "dy0", "logw", "H", "logsig")
PARAM_NAMES = tuple(f"{n}_{s}" for s in (0, 1) for n in SHEET_PARAMS) + ("logk", "gain1")
# per tile: ((side, rows back, columns across), summed relative power) over earlier stacks
History = list[list[tuple[tuple[int, int, int], float]]]
_TISSUE_FLOOR = 40.0
_SIG_VOXEL, _SIG_PROFILE = 0.15, 0.02


@dataclass
class _Geometry:
    """Tile geometry in µm and mm."""

    layout: TileLayout
    px: float = field(init=False)
    step_x: float = field(init=False)
    step_y: float = field(init=False)
    cu: float = field(init=False)
    cv: float = field(init=False)

    def __post_init__(self) -> None:
        self.px = self.layout.voxel_um[2]
        self.step_y, self.step_x = (s * self.px for s in self.layout.step_px)
        _, ny, nx = self.layout.shape
        self.cu, self.cv = nx * self.px / 2 / 1000, ny * self.px / 2 / 1000


@dataclass(frozen=True)
class SheetModel:
    """A fitted model.

    Attributes
    ----------
    layout : TileLayout
        The grid it was fit to.
    params : numpy.ndarray
        Values in :data:`PARAM_NAMES` order.
    """

    layout: TileLayout
    params: np.ndarray

    @property
    def named(self) -> dict[str, float]:
        """Parameters by name."""
        return {n: float(v) for n, v in zip(PARAM_NAMES, self.params, strict=True)}

    def log_illumination(self, tile: Tile) -> np.ndarray:
        """``log M_t`` on ``tile``'s (y, x) voxel grid.

        Parameters
        ----------
        tile : Tile
            Which tile.

        Returns
        -------
        numpy.ndarray
            (y, x) float64.
        """
        g = _Geometry(self.layout)
        _, ny, nx = self.layout.shape
        uu, vv = (a.ravel() for a in np.meshgrid((np.arange(nx) + 0.5) * g.px, (np.arange(ny) + 0.5) * g.px))
        tid = np.full(uu.size, self.layout.tiles.index(tile))
        return _log_m(self.params, g, _history(self.layout), tid, uu, vv).reshape(ny, nx)

    def save(self, path: str) -> None:
        """Write the parameters and every tile's ``log M`` to an ``.npz``."""
        maps = np.stack([self.log_illumination(t) for t in self.layout.tiles])
        np.savez(
            path,
            params=self.params,
            names=np.array(PARAM_NAMES),
            log_m=maps,
            tiles=np.array([t.name for t in self.layout.tiles]),
        )


def _history(layout: TileLayout) -> History:
    """Per tile: ((side, rows back, columns across), summed relative power) over every earlier stack."""
    out = []
    for i, t in enumerate(layout.tiles):
        h: dict[tuple[int, int, int], float] = {}
        for e in layout.tiles[:i]:
            key = (e.side, t.row - e.row, t.col - e.col)
            h[key] = h.get(key, 0.0) + layout.relative_power[(layout.channel, e.side)]
        out.append(sorted(h.items()))
    return out


def _log_sheet(q: dict[str, float], s: int, dv: np.ndarray, beam: np.ndarray, cu: float) -> np.ndarray:
    beam = np.clip(beam, -cu, cu)
    y = dv - (q[f"y0_{s}"] + q[f"dy0_{s}"] * beam)
    w = np.exp(q[f"logw_{s}"])
    sig = np.exp(q[f"logsig_{s}"]) * np.sqrt(2)
    h = q[f"H_{s}"]
    return np.asarray(-2 * y**2 / w**2 + np.log(np.maximum(0.5 * (erf((h - y) / sig) + erf((h + y) / sig)), 1e-12)))


def _log_m(
    params: np.ndarray, g: _Geometry, history: History, tid: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    q = dict(zip(PARAM_NAMES, params, strict=True))
    uc, vc = u / 1000 - g.cu, v / 1000 - g.cv
    side = np.array([t.side for t in g.layout.tiles])[tid]
    out = np.where(side == 1, q["gain1"], 0.0)
    for s in (0, 1):
        m = side == s
        if m.any():
            out[m] += q[f"a1_{s}"] * uc[m] + q[f"a2_{s}"] * uc[m] ** 2 + _log_sheet(q, s, vc[m], uc[m], g.cu)
    norm = {s: float(np.exp(_log_sheet(q, s, np.zeros(1), np.zeros(1), g.cu))[0]) for s in (0, 1)}
    dose = np.zeros_like(u)
    for t in np.unique(tid):
        m = tid == t
        for (s_e, drow, dcol), power in history[t]:
            dv = vc[m] + drow * g.step_y / 1000
            dose[m] += power * np.exp(_log_sheet(q, s_e, dv, uc[m] + dcol * g.step_x / 1000, g.cu)) / norm[s_e]
    return np.asarray(out - np.exp(q["logk"]) * dose)


@dataclass
class _Pairs:
    """Log-ratio samples ``y = log I_b - log I_a`` at tile positions (u, v) in µm."""

    a: np.ndarray
    ua: np.ndarray
    va: np.ndarray
    b: np.ndarray
    ub: np.ndarray
    vb: np.ndarray
    y: np.ndarray
    label: np.ndarray

    @classmethod
    def concat(cls, parts: list[dict[str, np.ndarray]]) -> _Pairs:
        keys = ("a", "ua", "va", "b", "ub", "vb", "y", "label")
        return cls(*(np.concatenate([p[k] for p in parts]) for k in keys))


def _overlap_pairs(
    layout: TileLayout, stacks: dict[str, np.ndarray], offsets: Offsets, g: _Geometry, rng: np.random.Generator
) -> _Pairs:
    _, ny, nx = layout.shape
    ov_y, ov_x = layout.overlap_px
    index = {t.name: i for i, t in enumerate(layout.tiles)}
    parts = []
    for kind, ta, tb in layout.neighbor_pairs():
        a = stacks[ta.name]
        # a voxel k of a shows what b shows at k + d, d = offset_a - offset_b
        d_ab = np.subtract(offsets[ta.name], offsets[tb.name])
        b = nd_shift(stacks[tb.name], -d_ab, order=1, mode="nearest")
        va, vb = (a[:, :, nx - ov_x :], b[:, :, :ov_x]) if kind == "LR" else (a[:, ny - ov_y :, :], b[:, :ov_y, :])
        zz, yy, xx = np.nonzero((va > _TISSUE_FLOOR) & (vb > _TISSUE_FLOOR))
        if len(zz) < 2000:
            continue
        pick = rng.choice(len(zz), min(3000, len(zz)), replace=False)
        zz, yy, xx = zz[pick], yy[pick], xx[pick]
        if kind == "LR":
            ua, va_, ub, vb_ = (xx + nx - ov_x + 0.5) * g.px, (yy + 0.5) * g.px, (xx + 0.5) * g.px, (yy + 0.5) * g.px
        else:
            ua, va_, ub, vb_ = (xx + 0.5) * g.px, (yy + ny - ov_y + 0.5) * g.px, (xx + 0.5) * g.px, (yy + 0.5) * g.px
        n = len(zz)
        label = f"{kind} r{ta.row + 1}c{ta.col + 1}->r{tb.row + 1}c{tb.col + 1}"
        parts.append(
            {
                "a": np.full(n, index[ta.name]),
                "ua": ua,
                "va": va_,
                "b": np.full(n, index[tb.name]),
                "ub": ub,
                "vb": vb_,
                "y": np.log(vb[zz, yy, xx]) - np.log(va[zz, yy, xx]),
                "label": np.array([label] * n),
            }
        )
    return _Pairs.concat(parts)


def _mirror_pairs(
    layout: TileLayout,
    fused: np.ndarray,
    mask: np.ndarray,
    mirror: np.ndarray,
    placement: tuple[int, int, int],
    spacing_um: float,
    g: _Geometry,
    rng: np.random.Generator,
) -> _Pairs:
    eroded = sitk.GetArrayFromImage(sitk.BinaryErode(sitk.GetImageFromArray(mask.astype(np.uint8)), [8, 8, 8])) > 0
    _, ny, nx = layout.shape
    index = {(t.row, t.col): i for i, t in enumerate(layout.tiles)}

    def owner(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = (idx[:, 2] - placement[2] + 0.5) * spacing_um
        y = (idx[:, 1] - placement[1] + 0.5) * spacing_um
        col = np.clip(np.floor(x / g.step_x).astype(int), 0, layout.n_cols - 1)
        row = np.clip(np.floor(y / g.step_y).astype(int), 0, layout.n_rows - 1)
        u, v = x - col * g.step_x, y - row * g.step_y
        ok = ((u > nx * g.px - g.step_x + 50) | (col == 0)) & ((u < g.step_x - 50) | (col == layout.n_cols - 1))
        ok &= ((v > ny * g.px - g.step_y + 50) | (row == 0)) & ((v < g.step_y - 50) | (row == layout.n_rows - 1))
        ok &= (u >= 0) & (u < nx * g.px) & (v >= 0) & (v < ny * g.px)
        tid = np.array([index.get((r, c), -1) for r, c in zip(row, col, strict=True)])
        return np.where(ok, tid, -1), u, v

    p = np.argwhere(eroded)
    p = p[rng.choice(len(p), min(600000, len(p)), replace=False)]
    q = np.rint(np.stack([mirror[i][tuple(p.T)] for i in range(3)], axis=1)).astype(int)
    ok = np.all((q >= 0) & (q < np.array(fused.shape)), axis=1)
    p, q = p[ok], q[ok]
    ok = (
        eroded[tuple(q.T)]
        & (p[:, 2] < q[:, 2])
        & (fused[tuple(p.T)] > _TISSUE_FLOOR)
        & (fused[tuple(q.T)] > _TISSUE_FLOOR)
    )
    p, q = p[ok], q[ok]
    tp, up, vp = owner(p)
    tq, uq, vq = owner(q)
    keep = rng.permutation(np.nonzero((tp >= 0) & (tq >= 0))[0])[:60000]
    cols = [t.col for t in layout.tiles]
    return _Pairs(
        a=tp[keep],
        ua=up[keep],
        va=vp[keep],
        b=tq[keep],
        ub=uq[keep],
        vb=vq[keep],
        y=np.log(fused[tuple(q[keep].T)]) - np.log(fused[tuple(p[keep].T)]),
        label=np.array([f"mirror c{cols[i] + 1}<->c{cols[j] + 1}" for i, j in zip(tp[keep], tq[keep], strict=True)]),
    )


def _profile_bands(layout: TileLayout) -> tuple[int, slice, slice]:
    """Give the x margin, the rows that must hold tissue, and the rows a profile is normalized over."""
    _, ny, nx = layout.shape
    margin = round(0.1 * nx)
    return margin, slice(round(0.4 * ny), round(0.6 * ny)), slice(round(0.3 * ny), round(0.7 * ny))


def _ap_profiles(
    layout: TileLayout, stacks: dict[str, np.ndarray]
) -> tuple[dict[int, np.ndarray], dict[int, list[int]]]:
    """Per side, the median A→P log profile over interior-row tiles, and which tiles it used."""
    nz, _, nx = layout.shape
    margin, center, norm = _profile_bands(layout)
    min_center = 0.25 * (center.stop - center.start) * (nx - 2 * margin)
    min_row = 0.25 * (nx - 2 * margin)
    profiles, used_tiles = {}, {}
    for s in (0, 1):
        rows, tids = [], []
        for i, t in enumerate(layout.tiles):
            if t.side != s or t.row in (0, layout.n_rows - 1):
                continue
            a = stacks[t.name]
            used = False
            for z in range(int(0.2 * nz), int(0.8 * nz), 2):
                plane = a[z, :, margin:-margin]
                tissue = plane > _TISSUE_FLOOR
                if tissue[center].sum() < min_center:
                    continue
                lg = np.where(tissue, np.log(np.maximum(plane, 1)), np.nan)
                r = np.nanmedian(lg, axis=1)
                r[tissue.sum(1) < min_row] = np.nan
                rows.append(r - np.nanmedian(lg[norm]))
                used = True
            if used:
                tids.append(i)
        if rows:
            profiles[s] = np.nanmedian(np.stack(rows), axis=0)
            used_tiles[s] = tids
    return profiles, used_tiles


def fit_sheet_model(
    layout: TileLayout,
    load: Callable[[Tile], np.ndarray],
    offsets: Offsets,
    fused: sitk.Image,
    mask: sitk.Image,
    mirror: np.ndarray,
    *,
    placement: tuple[int, int, int] = (0, 0, 0),
    seed: int = 0,
    report: Callable[[str], None] = print,
) -> SheetModel:
    """Fit the model by robust (IRLS Huber) least squares.

    Parameters
    ----------
    layout : TileLayout
        The grid.
    load : callable
        Returns a tile's dark-subtracted, destriped stack.
    offsets : dict
        Tile offsets from :func:`~aind_lightsheet_registration.stitch.solve_offsets`, aligning overlap samples.
    fused : sitk.Image
        Uncorrected nominal fusion of the same tiles, placed on a reference grid.
    mask : sitk.Image
        Tissue mask on ``fused``'s grid.
    mirror : numpy.ndarray
        From :func:`~aind_lightsheet_registration.mirror.mirror_indices` on ``fused``.
    placement : tuple of int
        Voxel shift (z, y, x) of the mosaic within ``fused``'s grid.
    seed : int
        Seeds the sample draws.
    report : callable
        Receives progress and fit-summary lines.

    Returns
    -------
    SheetModel
        The fit.
    """
    rng = np.random.default_rng(seed)
    g = _Geometry(layout)
    history = _history(layout)
    stacks = {t.name: load(t) for t in layout.tiles}
    overlaps = _overlap_pairs(layout, stacks, offsets, g, rng)
    mirrors = _mirror_pairs(
        layout,
        sitk.GetArrayFromImage(fused),
        sitk.GetArrayFromImage(mask) > 0,
        mirror,
        placement,
        fused.GetSpacing()[0] * 1000,
        g,
        rng,
    )
    profiles, profile_tiles = _ap_profiles(layout, stacks)
    del stacks
    report(
        f"{len(overlaps.y)} overlap samples ({len(set(overlaps.label))} pairs), {len(mirrors.y)} mirror samples, "
        f"A->P profiles for sides {sorted(profiles)}"
    )

    _, ny, nx = layout.shape
    margin, _, norm = _profile_bands(layout)
    grid_u = (np.arange(margin, nx - margin, max(1, round(nx / 25))) + 0.5) * g.px
    grid_v = (np.arange(ny) + 0.5) * g.px
    gu, gv = (a.ravel() for a in np.meshgrid(grid_u, grid_v))

    def model_profile(params: np.ndarray, s: int) -> np.ndarray:
        per_tile = []
        for i in profile_tiles[s]:
            lm = _log_m(params, g, history, np.full(gu.size, i), gu, gv).reshape(ny, len(grid_u)).mean(axis=1)
            per_tile.append(lm - np.median(lm[norm]))
        return np.asarray(np.median(np.stack(per_tile), axis=0))

    def pair_pred(params: np.ndarray, d: _Pairs) -> np.ndarray:
        return np.asarray(_log_m(params, g, history, d.b, d.ub, d.vb) - _log_m(params, g, history, d.a, d.ua, d.va))

    def raw_residuals(params: np.ndarray) -> list[np.ndarray]:
        prof = [profiles[s] - model_profile(params, s) for s in sorted(profiles)]
        return [
            overlaps.y - pair_pred(params, overlaps),
            mirrors.y - pair_pred(params, mirrors),
            np.nan_to_num(np.concatenate(prof)) if prof else np.zeros(0),
        ]

    x0, lo, hi = _initial(g)
    sizes = [max(1, len(r)) for r in raw_residuals(x0)]
    sigmas = [_SIG_VOXEL, _SIG_VOXEL, _SIG_PROFILE]
    huber = [np.ones(len(r)) for r in raw_residuals(x0)]
    best = x0
    for it in range(4):

        def residuals(params: np.ndarray) -> np.ndarray:
            parts = zip(raw_residuals(params), sigmas, sizes, huber, strict=True)
            return np.concatenate([r / sg / np.sqrt(n) * np.sqrt(h) for r, sg, n, h in parts])

        fit = least_squares(residuals, best, bounds=(lo, hi), x_scale="jac", max_nfev=200)
        best = fit.x
        rr = raw_residuals(best)
        for i in (0, 1):
            c = rr[i] - np.median(rr[i])
            huber[i] = np.minimum(1.0, 1.345 * 1.4826 * np.median(np.abs(c)) / np.maximum(np.abs(c), 1e-12))
        report(
            f"IRLS {it + 1}: cost {fit.cost:.4f}, nfev {fit.nfev}; overlap MAD {np.median(np.abs(rr[0])):.4f}, "
            f"mirror MAD {np.median(np.abs(rr[1])):.4f}, profile RMS {np.sqrt(np.mean(rr[2] ** 2)):.4f}"
        )
    model = SheetModel(layout=layout, params=best)
    for line in _summary(model, overlaps, mirrors, pair_pred):
        report(line)
    return model


def _initial(g: _Geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = {n: i for i, n in enumerate(PARAM_NAMES)}
    x0, lo, hi = np.zeros(len(PARAM_NAMES)), np.full(len(PARAM_NAMES), -np.inf), np.full(len(PARAM_NAMES), np.inf)
    for s in (0, 1):
        x0[idx[f"logw_{s}"]], lo[idx[f"logw_{s}"]], hi[idx[f"logw_{s}"]] = np.log(5.0), np.log(0.5), np.log(50)
        # the aperture is a property of the illumination optics, so its prior does not scale with the field
        x0[idx[f"H_{s}"]], lo[idx[f"H_{s}"]], hi[idx[f"H_{s}"]] = 1.8, 1.0, max(4.0, 2 * g.cv)
        x0[idx[f"logsig_{s}"]], lo[idx[f"logsig_{s}"]], hi[idx[f"logsig_{s}"]] = np.log(0.2), np.log(0.02), np.log(1.5)
    x0[idx["logk"]] = np.log(0.2)
    return np.clip(x0, lo + 1e-6, hi - 1e-6), lo, hi


def _summary(
    model: SheetModel, overlaps: _Pairs, mirrors: _Pairs, pair_pred: Callable[[np.ndarray, _Pairs], np.ndarray]
) -> Iterator[str]:
    q = model.named
    for s in (0, 1):
        yield (
            f"side {s}: sheet {2 * q[f'H_{s}']:.2f} mm, center {q[f'y0_{s}'] * 1000:+.0f} um "
            f"({q[f'dy0_{s}'] * 1000:+.1f} um per mm along the beam), edge sigma {np.exp(q[f'logsig_{s}']) * 1000:.0f} "
            f"um, Gaussian w {np.exp(q[f'logw_{s}']):.2f} mm, amplitude log {q[f'a1_{s}']:+.3f} u "
            f"{q[f'a2_{s}']:+.3f} u^2"
        )
    yield f"bleaching k {np.exp(q['logk']):.3f} per pass; right/left gain {np.exp(q['gain1']):.3f}"
    for d in (overlaps, mirrors):
        pred = pair_pred(model.params, d)
        for label in sorted(set(d.label)):
            m = d.label == label
            if m.sum() >= 500:
                yield (
                    f"  {label:22s} n={m.sum():5d} observed {np.exp(np.median(d.y[m])):.3f} "
                    f"predicted {np.exp(np.median(pred[m])):.3f}"
                )
