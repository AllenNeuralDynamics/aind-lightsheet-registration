"""Tile offsets from overlap correlation, and the point field from a nominal-grid fusion to a stitched one.

Each overlap is searched over every integer shift within a fixed physical window for the highest Pearson correlation
of log intensity over the tissue both tiles hold there, by FFT, and refined to subvoxel by a parabola per axis; phase
correlation locks onto zero lag on these tiles. Offsets then solve ``offset[later] - offset[earlier] = measured`` by
correlation-weighted least squares, centered on their median, after dropping pairs the others contradict.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy.fft import irfftn, next_fast_len, rfftn
from scipy.ndimage import distance_transform_edt, gaussian_filter

from aind_lightsheet_registration.tiles import Tile, TileLayout

Offsets = dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class PairShift:
    """Measured offset of one overlap.

    Attributes
    ----------
    kind : str
        ``"AP"`` or ``"LR"``.
    earlier, later : Tile
        The anterior or left tile, and its neighbor.
    shift : tuple of float
        Position (z, y, x voxels) of ``later`` relative to nominal, minus that of ``earlier``.
    correlation : float
        Peak correlation.
    at_edge : bool
        The peak sat on the search boundary, so the shift is unreliable.
    """

    kind: str
    earlier: Tile
    later: Tile
    shift: tuple[float, float, float]
    correlation: float
    at_edge: bool

    @property
    def label(self) -> str:
        """Human-readable pair name, with 1-based row and column positions."""
        a, b = self.earlier, self.later
        return f"{self.kind} r{a.row + 1}c{a.col + 1}->r{b.row + 1}c{b.col + 1}"


def _parabolic_peak(score: np.ndarray, lags: list[np.ndarray], best: np.ndarray) -> np.ndarray:
    d = np.array([lag[i] for lag, i in zip(lags, best, strict=True)], float)
    for axis in range(score.ndim):
        i = best[axis]
        if 0 < i < len(lags[axis]) - 1:
            y0, y1, y2 = (score[(*best[:axis], k, *best[axis + 1 :])] for k in (i - 1, i, i + 1))
            den = y0 - 2 * y1 + y2
            if den < 0:
                d[axis] += 0.5 * (y0 - y2) / den
    return d


_NEIGHBORS = np.array(list(np.ndindex(3, 3, 3))) - 1
_QUADRATIC = np.column_stack(
    [
        np.ones(27),
        _NEIGHBORS,
        _NEIGHBORS**2 / 2,
        *(_NEIGHBORS[:, i] * _NEIGHBORS[:, j] for i, j in ((0, 1), (0, 2), (1, 2))),
    ]
)


def _subvoxel_peak(score: np.ndarray, lags: list[np.ndarray], best: np.ndarray) -> np.ndarray:
    """Lag of the peak of a 3D quadratic fitted to the 27 scores around ``best``.

    The joint fit halves the error of a parabola per axis wherever the peak is oblique to the axes; the per-axis
    parabola stands in where the neighborhood is incomplete or the fit has no maximum within it.
    """
    lo, hi = best - 1, best + 2
    if np.all(lo >= 0) and np.all(hi <= np.array(score.shape)):
        local = score[tuple(slice(i, j) for i, j in zip(lo, hi, strict=True))].reshape(-1)
        if np.all(np.isfinite(local)):
            c = np.linalg.lstsq(_QUADRATIC, local, rcond=None)[0]
            hess = np.diag(c[4:7])
            for (i, j), v in zip(((0, 1), (0, 2), (1, 2)), c[7:10], strict=True):
                hess[i, j] = hess[j, i] = v
            if np.all(np.linalg.eigvalsh(hess) < 0):
                step = -np.linalg.solve(hess, c[1:4])
                if np.all(np.abs(step) <= 1):
                    return np.array([lag[i] for lag, i in zip(lags, best, strict=True)], float) + step
    return _parabolic_peak(np.nan_to_num(score, nan=-1.0), lags, best)


def _overlap_strips(
    a: np.ndarray, b: np.ndarray, kind: str, layout: TileLayout, normal_margin: int
) -> tuple[np.ndarray, np.ndarray, tuple[slice, ...]]:
    """Log-intensity strips of ``a`` and ``b`` in one frame, ``b`` at its nominal place; NaN where a tile is absent.

    Along the overlap's normal each strip runs ``normal_margin`` voxels past the nominal overlap, so a shift of up to
    that much still finds the overlapping tissue. Also returns the frame region holding ``a``'s nominal overlap.
    """
    axis = 1 if kind == "AP" else 2
    ov = layout.overlap_px[axis - 1]
    n = layout.shape[axis]
    width = ov + normal_margin
    frame_shape = list(layout.shape)
    frame_shape[axis] = width + normal_margin
    fa = np.full(frame_shape, np.nan, np.float32)
    fb = np.full(frame_shape, np.nan, np.float32)

    def along(sl: slice) -> tuple[slice, ...]:
        return tuple(sl if i == axis else slice(None) for i in range(3))

    fa[along(slice(0, width))] = np.log(np.clip(a[along(slice(n - width, n))], 1, None))
    fb[along(slice(normal_margin, None))] = np.log(np.clip(b[along(slice(0, width))], 1, None))
    return fa, fb, along(slice(normal_margin, width))


def _masked_ncc(
    fa: np.ndarray, ma: np.ndarray, fb: np.ndarray, mb: np.ndarray, margin: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation of ``fa`` and ``fb`` over voxels inside both masks, at every lag within ``margin``.

    Lag ``d`` pairs ``fa[x]`` with ``fb[x + d]``. Six FFT cross-correlations give, per lag, the overlap count and the
    sums Pearson needs (Padfield 2012); zero padding by ``margin`` keeps the circular wrap out of the searched lags.

    Returns
    -------
    tuple of numpy.ndarray
        Correlation and overlap voxel count, each of shape ``2 * margin + 1``.
    """
    shape = [next_fast_len(s + m, real=True) for s, m in zip(fa.shape, margin, strict=True)]
    # centering is exact for Pearson and keeps the float32 sums from cancelling
    a = np.where(ma, fa - fa[ma].mean(), 0).astype(np.float32)
    b = np.where(mb, fb - fb[mb].mean(), 0).astype(np.float32)
    ma32, mb32 = ma.astype(np.float32), mb.astype(np.float32)
    idx = np.ix_(*[np.arange(-m, m + 1) % s for m, s in zip(margin, shape, strict=True)])

    def spectrum(x: np.ndarray) -> np.ndarray:
        return rfftn(x, shape, workers=-1)

    def corr(p_conj: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.asarray(irfftn(p_conj * q, shape, workers=-1)[idx], np.float64)

    a_m, a_1, a_2 = (np.conj(spectrum(x)) for x in (ma32, a, a * a))
    b_m, b_1, b_2 = (spectrum(x) for x in (mb32, b, b * b))
    count = np.rint(corr(a_m, b_m))
    with np.errstate(divide="ignore", invalid="ignore"):
        sa, sb = corr(a_1, b_m), corr(a_m, b_1)
        var_a = corr(a_2, b_m) - sa * sa / count
        var_b = corr(a_m, b_2) - sb * sb / count
        r = (corr(a_1, b_1) - sa * sb / count) / np.sqrt(var_a * var_b)
    return np.where(np.isfinite(r) & (var_a > 0) & (var_b > 0), r, np.nan), count


def _highpass(f: np.ndarray, mask: np.ndarray, sigma: tuple[float, ...]) -> np.ndarray:
    """``f`` minus its Gaussian mean over ``mask`` (normalized convolution), so background does not leak in."""
    m = mask.astype(np.float32)
    f0 = np.where(mask, f, 0).astype(np.float32)
    mean = gaussian_filter(f0, sigma) / np.maximum(gaussian_filter(m, sigma), 1e-3)
    return np.where(mask, f0 - mean, 0)


def measure_pair(
    a: np.ndarray,
    b: np.ndarray,
    kind: str,
    layout: TileLayout,
    *,
    margin_um: tuple[float, float, float] = (192.0, 172.8, 172.8),
    tissue_floor: float = 40.0,
    min_voxels: int = 5000,
    min_overlap_fraction: float = 0.3,
    highpass_um: float = 60.0,
    refine_voxels: int = 2,
) -> tuple[np.ndarray, float, bool] | None:
    """Offset of ``b`` relative to ``a`` across their overlap.

    Every integer shift within the window is scored by FFT masked correlation of high-passed log intensity, since
    large-scale brightness trends make plain log intensity score highest at the window's edge once the window is
    wide. The plain-log peak within ``refine_voxels`` of that one, refined to subvoxel by a parabola per axis, gives
    the shift and correlation.

    Parameters
    ----------
    a, b : numpy.ndarray
        Dark-subtracted (z, y, x) stacks; ``b`` is below (AP) or right of (LR) ``a``.
    kind : str
        ``"AP"`` or ``"LR"``.
    layout : TileLayout
        Grid giving the overlap width.
    margin_um : tuple of float
        Search half-width (z, y, x) in µm, rounded to whole voxels.
    tissue_floor : float
        Intensity below which voxels are background.
    min_voxels : int
        Fewest tissue voxels a shift needs to be scored.
    min_overlap_fraction : float
        Fewest tissue voxels a shift needs to be scored, as a fraction of the most any shift has; shifts that leave a
        sliver of the overlap score high by chance.
    highpass_um : float
        Gaussian sigma (µm) of the trend removed for the wide search.
    refine_voxels : int
        Half-width of the plain-log search around the high-passed peak.

    Returns
    -------
    tuple or None
        ``(shift, correlation, at_edge)``, shift being ``b``'s position minus ``a``'s in voxels (z, y, x); None when
        the overlap holds too little tissue. ``at_edge`` marks a peak on either search's boundary, so the two disagree
        or the true shift lies outside the window.
    """
    margin = np.array([max(1, round(m / v)) for m, v in zip(margin_um, layout.voxel_um, strict=True)])
    normal = 1 if kind == "AP" else 2
    fa, fb, nominal = _overlap_strips(a, b, kind, layout, int(margin[normal]))
    floor = np.log(tissue_floor)
    ma, mb = fa > floor, fb > floor  # NaN compares False
    if ma[nominal].mean() < 0.05:
        return None
    # voxels outside both masks contribute nothing, so search only their bounding box
    box = tuple(slice(i.min(), i.max() + 1) for i in np.nonzero(ma | mb))
    fa, fb, ma, mb = fa[box], fb[box], ma[box], mb[box]
    sigma = tuple(highpass_um / v for v in layout.voxel_um)
    coarse, count = _masked_ncc(_highpass(fa, ma, sigma), ma, _highpass(fb, mb, sigma), mb, tuple(margin))
    enough = count >= max(min_voxels, min_overlap_fraction * count.max())
    if not np.any(enough & np.isfinite(coarse)):
        return None
    peak = np.array(np.unravel_index(np.nanargmax(np.where(enough, coarse, np.nan)), coarse.shape))
    plain, _ = _masked_ncc(fa, ma, fb, mb, tuple(margin))
    plain = np.where(enough, plain, np.nan)
    lo = np.maximum(peak - refine_voxels, 0)
    near = plain[tuple(slice(i, j) for i, j in zip(lo, peak + refine_voxels + 1, strict=True))]
    if np.isnan(near).all():
        return None
    best = lo + np.array(np.unravel_index(np.nanargmax(near), near.shape))
    at_edge = bool(np.any((peak == 0) | (peak == np.array(coarse.shape) - 1) | (np.abs(best - peak) == refine_voxels)))
    lags = [np.arange(-m, m + 1) for m in margin]
    d = _subvoxel_peak(plain, lags, best)
    # a voxel k of a's overlap matches b at k + d, so b sits -d from its nominal place relative to a
    return -d, float(plain[tuple(best)]), at_edge


def tile_row_profile(
    layout: TileLayout,
    load: Callable[[Tile], np.ndarray],
    *,
    tissue_floor: float = 40.0,
    min_tissue_fraction: float = 0.2,
) -> np.ndarray:
    """Log brightness of each tile row (y) relative to the tile's middle, pooled over tiles.

    Tiles darken toward their A-P edges, so in an overlap each tile's falloff runs the opposite way to its neighbor's;
    left in, that shifts the correlation peak along y by several µm. The falloff is the same in every tile, so its
    median over tiles, where anatomy differs, estimates it without knowing where the tiles sit.

    Parameters
    ----------
    layout : TileLayout
        The grid.
    load : callable
        Returns a tile's dark-subtracted stack.
    tissue_floor : float
        Intensity below which voxels are background.
    min_tissue_fraction : float
        Fewest tissue voxels a tile, or one of its rows, needs to count.

    Returns
    -------
    numpy.ndarray
        Log gain per row, length ``layout.shape[1]``; 0 where no tile has enough tissue.
    """
    profiles = []
    for t in layout.tiles:
        log = np.log(np.clip(load(t), 1, None))
        tissue = log > np.log(tissue_floor)
        if tissue.mean() < min_tissue_fraction:
            continue
        rows = np.nanmedian(np.where(tissue, log, np.nan), axis=(0, 2))
        rows[tissue.mean(axis=(0, 2)) < min_tissue_fraction / 10] = np.nan
        profiles.append(rows - np.nanmedian(rows))
    if not profiles:
        return np.zeros(layout.shape[1])
    with warnings.catch_warnings():  # rows no tile covers
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nan_to_num(np.nanmedian(profiles, axis=0)), float)


def measure_pairs(
    layout: TileLayout,
    load: Callable[[Tile], np.ndarray],
    *,
    margin_um: tuple[float, float, float] = (192.0, 172.8, 172.8),
    tissue_floor: float = 40.0,
    row_profile: np.ndarray | None = None,
) -> list[PairShift]:
    """Measure every overlapping pair, each tile first divided by the tiles' common row profile.

    Parameters
    ----------
    layout : TileLayout
        The grid.
    load : callable
        Returns a tile's dark-subtracted stack.
    margin_um : tuple of float
        Search half-width (z, y, x) in µm.
    tissue_floor : float
        Intensity below which voxels are background.
    row_profile : numpy.ndarray, optional
        Log gain per tile row to divide out; :func:`tile_row_profile` when omitted, zeros to divide out nothing.

    Returns
    -------
    list of PairShift
        One per overlap with enough tissue.
    """
    if row_profile is None:
        row_profile = tile_row_profile(layout, load, tissue_floor=tissue_floor)
    flat = np.exp(-row_profile).astype(np.float32)[None, :, None]
    neighbors = list(layout.neighbor_pairs())
    uses = Counter(t.name for _, a, b in neighbors for t in (a, b))
    cache: dict[str, np.ndarray] = {}

    def get(t: Tile) -> np.ndarray:
        if t.name not in cache:
            cache[t.name] = load(t) * flat
        return cache[t.name]

    out = []
    for kind, a, b in neighbors:
        measured = measure_pair(get(a), get(b), kind, layout, margin_um=margin_um, tissue_floor=tissue_floor)
        for t in (a, b):  # hold a tile only until its last overlap
            uses[t.name] -= 1
            if uses[t.name] == 0:
                del cache[t.name]
        if measured is None:
            continue
        shift, corr, at_edge = measured
        out.append(PairShift(kind, a, b, (shift[0], shift[1], shift[2]), corr, at_edge))
    return out


def consistent_pairs(
    layout: TileLayout, pairs: list[PairShift], *, min_correlation: float = 0.3, max_residual: float = 1.0
) -> list[PairShift]:
    """Select the pairs a solve should use.

    Pairs at the search edge or below ``min_correlation`` are dropped. Then, while some pair disagrees with the
    solved offsets by more than ``max_residual`` voxels, the worst one is dropped and the offsets solved again; a pair
    whose loss would leave tiles unplaced is kept, since nothing else can check it. A pair in only one cycle, as on
    the grid's perimeter, shares its error with the cycle's other pairs in proportion to their weights, so it is
    told apart only by a lower correlation.

    Parameters
    ----------
    layout : TileLayout
        The grid.
    pairs : list of PairShift
        From :func:`measure_pairs`.
    min_correlation : float
        Lowest peak correlation kept.
    max_residual : float
        Largest disagreement (voxels) between a kept pair and the solve.

    Returns
    -------
    list of PairShift
        The kept pairs.
    """
    kept = [p for p in pairs if not p.at_edge and p.correlation >= min_correlation]
    names = _names_with_tissue(layout, pairs)
    _check_connected(names, kept)
    while True:
        pos = _solve(names, kept)
        resid = [float(np.linalg.norm(pos[p.later.name] - pos[p.earlier.name] - p.shift)) for p in kept]
        for i in np.argsort(resid)[::-1]:
            if resid[i] <= max_residual:
                return kept
            rest = kept[:i] + kept[i + 1 :]
            if _connected(names, rest):
                kept = rest
                break
        else:
            return kept


def solve_offsets(
    layout: TileLayout, pairs: list[PairShift], *, min_correlation: float = 0.3, max_residual: float = 1.0
) -> Offsets:
    """Per-tile offsets from the :func:`consistent_pairs`, by correlation-weighted least squares.

    A tile with no measured pair has no tissue in its overlaps and keeps its nominal position. Every other tile must
    be reached by kept pairs, or it is an error, since the solve would give it an arbitrary offset.

    Parameters
    ----------
    layout : TileLayout
        The grid.
    pairs : list of PairShift
        From :func:`measure_pairs`.
    min_correlation : float
        Lowest peak correlation kept.
    max_residual : float
        Largest disagreement (voxels) between a kept pair and the solve.

    Returns
    -------
    dict
        Tile name → offset (z, y, x voxels at the layout's level) to add to its nominal position; median zero over
        the placed tiles, zero for tiles without tissue.
    """
    kept = consistent_pairs(layout, pairs, min_correlation=min_correlation, max_residual=max_residual)
    placed = _solve(_names_with_tissue(layout, pairs), kept)
    return {
        t.name: tuple(float(v) for v in placed[t.name]) if t.name in placed else (0.0, 0.0, 0.0)  # type: ignore[misc]
        for t in layout.tiles
    }


def _names_with_tissue(layout: TileLayout, pairs: list[PairShift]) -> list[str]:
    with_tissue = {t.name for p in pairs for t in (p.earlier, p.later)}
    return [t.name for t in layout.tiles if t.name in with_tissue]


def _solve(names: list[str], pairs: list[PairShift]) -> dict[str, np.ndarray]:
    n = len(names)
    rows, rhs, weights = [], [], []
    for p in pairs:
        r = np.zeros(n)
        r[names.index(p.later.name)], r[names.index(p.earlier.name)] = 1, -1
        rows.append(r)
        rhs.append(p.shift)
        weights.append(p.correlation)
    anchor = np.zeros(n)
    anchor[0] = 1
    w = np.sqrt(np.r_[weights, 10.0])[:, None]
    pos, *_ = np.linalg.lstsq(np.vstack([*rows, anchor]) * w, np.vstack([*rhs, np.zeros(3)]) * w, rcond=None)
    pos -= np.median(pos, axis=0)
    return dict(zip(names, pos, strict=True))


def _reached(names: list[str], pairs: list[PairShift]) -> set[str]:
    reached = {names[0]}
    grew = True
    while grew:
        grew = False
        for p in pairs:
            ends = {p.earlier.name, p.later.name}
            if len(ends & reached) == 1:
                reached |= ends
                grew = True
    return reached


def _connected(names: list[str], pairs: list[PairShift]) -> bool:
    return _reached(names, pairs) >= set(names)


def _check_connected(names: list[str], pairs: list[PairShift]) -> None:
    if not names:
        raise ValueError("no tile pair has tissue in its overlap")
    missing = sorted(set(names) - _reached(names, pairs))
    if missing:
        raise ValueError(f"no reliable overlap places tiles {missing}")


def solve_residuals(offsets: Offsets, pairs: list[PairShift], *, min_correlation: float = 0.3) -> list[float]:
    """Distance (voxels) between each kept pair's measured shift and the solved offsets' difference."""
    return [
        float(np.linalg.norm(np.subtract(offsets[p.later.name], offsets[p.earlier.name]) - p.shift))
        for p in pairs
        if not p.at_edge and p.correlation >= min_correlation
    ]


def blend_weights(
    layout: TileLayout, tile: Tile, y: np.ndarray, x: np.ndarray, *, ap_weights: tuple[float, float] | None
) -> np.ndarray:
    """Fusion weight of ``tile`` at continuous mosaic positions ``y`` and ``x`` (level voxels), as a (y, x) grid.

    L→R overlaps ramp linearly. A→P overlaps ramp linearly when ``ap_weights`` is None; otherwise the earlier row's
    weight falls linearly from ``ap_weights[0]`` at the overlap's start to ``ap_weights[1]`` at its end.

    Parameters
    ----------
    layout : TileLayout
        The grid.
    tile : Tile
        Which tile.
    y, x : numpy.ndarray
        1D mosaic positions in level voxels; voxel k's center is at k.
    ap_weights : tuple of float or None
        Earlier row's weight at the start and end of an A→P overlap.

    Returns
    -------
    numpy.ndarray
        (len(y), len(x)) weights, 0 outside the tile.
    """
    ov_y, ov_x = layout.overlap_px
    _, ny, nx = layout.shape
    u, v = y - tile.row * layout.step_px[0], x - tile.col * layout.step_px[1]

    def ramp_up(t: np.ndarray, n: int) -> np.ndarray:
        return np.clip((t + 0.5) / n, 0, 1)

    wx = ((v >= -0.5) & (v < nx - 0.5)).astype(float)
    if tile.col > 0:
        wx = np.where(v < ov_x, wx * ramp_up(v, ov_x), wx)
    if tile.col < layout.n_cols - 1:
        wx = np.where(v >= nx - ov_x, wx * ramp_up(nx - 1 - v, ov_x), wx)
    wy = ((u >= -0.5) & (u < ny - 0.5)).astype(float)
    if ap_weights is None:
        if tile.row > 0:
            wy = np.where(u < ov_y, wy * ramp_up(u, ov_y), wy)
        if tile.row < layout.n_rows - 1:
            wy = np.where(u >= ny - ov_y, wy * ramp_up(ny - 1 - u, ov_y), wy)
    else:
        start, end = ap_weights
        if tile.row > 0:  # the later row of the overlap above
            frac = ramp_up(u, ov_y)
            wy = np.where(u < ov_y, wy * (1 - (start + (end - start) * frac)), wy)
        if tile.row < layout.n_rows - 1:  # the earlier row of the overlap below
            frac = ramp_up(u - (ny - ov_y), ov_y)
            wy = np.where(u >= ny - ov_y, wy * (start + (end - start) * frac), wy)
    return np.asarray(wy[:, None] * wx[None, :])


def nominal_to_stitched_field(
    layout: TileLayout,
    offsets: Offsets,
    reference: sitk.Image,
    *,
    ap_weights: tuple[float, float] | None,
    placement: tuple[int, int, int] = (0, 0, 0),
) -> sitk.Image:
    """Displacement field taking a point in a nominal-grid fusion to the same tissue in the stitched fusion.

    A point shows tissue from the tiles covering it, and the stitched fusion moves each tile's content by its offset.
    Where tiles overlap the offset is averaged with the nominal fusion's blending weights, so a point on a blended
    feature moves to the feature's blended position. Beyond the mosaic the nearest covered value is carried, so the
    field has no step at the footprint's edge. The field is constant along z.

    Parameters
    ----------
    layout : TileLayout
        The grid.
    offsets : dict
        From :func:`solve_offsets`.
    reference : sitk.Image
        Grid of the nominal fusion: the mosaic taken to its spacing by
        :func:`~aind_lightsheet_registration.reader.finish_downsampling`, then shifted by ``placement``.
    ap_weights : tuple of float or None
        A→P blending of the nominal fusion; see :func:`blend_weights`.
    placement : tuple of int
        Voxel shift (z, y, x) of the mosaic within ``reference``, from
        :func:`~aind_lightsheet_registration.fuse.place_on_reference`.

    Returns
    -------
    sitk.Image
        Float64 vector image on ``reference``'s grid, ITK point convention ``p -> p + d(p)``.
    """
    size_x, size_y, size_z = reference.GetSize()
    spacing_um = reference.GetSpacing()[0] * 1000
    voxel_um = np.asarray(layout.voxel_um)
    # finish_downsampling puts output voxel j at level continuous index -0.5 + (j + 0.5) * target / level spacing
    y = -0.5 + (np.arange(size_y) - placement[1] + 0.5) * spacing_um / voxel_um[1]
    x = -0.5 + (np.arange(size_x) - placement[2] + 0.5) * spacing_um / voxel_um[2]
    num = np.zeros((size_y, size_x, 3))
    den = np.zeros((size_y, size_x))
    for tile in layout.tiles:
        w = blend_weights(layout, tile, y, x, ap_weights=ap_weights)
        num += w[..., None] * (np.asarray(offsets[tile.name]) * voxel_um / 1000)
        den += w
    covered = den > 0
    d_zyx = np.where(covered[..., None], num / np.maximum(den, 1e-12)[..., None], 0.0)
    if not covered.all():
        nearest = np.asarray(distance_transform_edt(~covered, return_distances=False, return_indices=True))
        d_zyx = d_zyx[nearest[0], nearest[1]]
    # array axes (z, y, x) are ITK index axes (k, j, i); a physical displacement is direction @ index displacement
    direction = np.array(reference.GetDirection()).reshape(3, 3)
    d_phys = d_zyx[..., ::-1] @ direction.T
    field = np.broadcast_to(d_phys[None], (size_z, size_y, size_x, 3)).astype(np.float64)
    image = sitk.GetImageFromArray(np.ascontiguousarray(field), isVector=True)
    image.CopyInformation(reference)
    return image
