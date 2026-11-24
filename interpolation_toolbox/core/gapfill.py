"""Filling no-data gaps in a raster by kriging or conditional simulation.

Kriging gives the best (smoothest) estimate for each missing cell. Conditional
simulation gives realisations that honour the observed cells, share the
data's spatial texture (variogram) and, taken together, express the
uncertainty in the filled values.

Conditional simulation uses the classic "kriging the residual" construction:

    Z_cs = Z_krige(data) + [S - S_krige(S at data locations)]

where S is an unconditional Gaussian random field with the fitted covariance,
generated with FFT circulant embedding. Because the kriging weights depend
only on locations, they are computed once and reused for the data and for
every realisation.

Cells are addressed by (row, col) and converted to map distances with the
pixel sizes ``dx`` / ``dy``, so this module needs no geotransform.
"""

import math

import numpy as np

from . import variogram as vg
from .methods import kriging_chunks
from .neighbors import _report

#: Largest FFT embedding (cells) used for unconditional simulation.
MAX_EMBEDDING_CELLS = 24_000_000


class GapFillError(ValueError):
    pass


def box_dilate(mask, w):
    """True wherever a True cell lies within the (2w+1)^2 window around it."""
    mask = np.asarray(mask, bool)
    if w <= 0 or not mask.any():
        return mask.copy()
    R, C = mask.shape
    integral = np.zeros((R + 1, C + 1), dtype=np.int64)
    integral[1:, 1:] = mask.cumsum(0).cumsum(1)
    r0 = np.clip(np.arange(R) - w, 0, R)
    r1 = np.clip(np.arange(R) + w + 1, 0, R)
    c0 = np.clip(np.arange(C) - w, 0, C)
    c1 = np.clip(np.arange(C) + w + 1, 0, C)
    total = (integral[r1][:, c1] - integral[r0][:, c1]
             - integral[r1][:, c0] + integral[r0][:, c0])
    return total > 0


def _fft_size(n):
    """Smallest 2^a 3^b 5^c >= n (fast FFT length)."""
    n = max(int(n), 1)
    best = 1 << (n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


def embedding_shape(shape, dx, dy, variogram):
    """Torus size used to simulate a field of ``shape`` (rows, cols)."""
    R, C = shape
    rng = variogram.range * (1.0 if variogram.model != "gaussian" else 1.5)
    pad_r = min(max(R, int(math.ceil(rng / dy))), 4 * max(R, 16))
    pad_c = min(max(C, int(math.ceil(rng / dx))), 4 * max(C, 16))
    return _fft_size(R + pad_r), _fft_size(C + pad_c)


def unconditional_fields(shape, dx, dy, variogram, n, rng):
    """Generate ``n`` zero-mean Gaussian fields with covariance sill - gamma(h).

    Returns ``(fields, negative_fraction)`` where ``fields`` is a float32 array
    (n, rows, cols) and ``negative_fraction`` is the share of spectral mass
    clipped because the embedding was not positive definite (a measure of
    approximation error; normally ~0).
    """
    R, C = shape
    M1, M2 = embedding_shape(shape, dx, dy, variogram)
    if M1 * M2 > MAX_EMBEDDING_CELLS:
        raise GapFillError(
            "The simulation grid would need {:,} cells ({} x {}). Fill a smaller area, "
            "use a coarser raster or use kriging instead.".format(M1 * M2, M1, M2))
    i = np.arange(M1)
    j = np.arange(M2)
    hy = np.minimum(i, M1 - i) * dy
    hx = np.minimum(j, M2 - j) * dx
    h = np.hypot(hy[:, None], hx[None, :])
    cov = variogram.sill - variogram(h)
    lam = np.fft.fft2(cov).real
    total = np.abs(lam).sum()
    neg = float(-lam[lam < 0].sum() / total) if total > 0 else 0.0
    root = np.sqrt(np.maximum(lam, 0.0) / (M1 * M2))
    del cov, lam, h

    out = np.empty((n, R, C), dtype=np.float32)
    made = 0
    while made < n:
        w = rng.standard_normal((M1, M2)) + 1j * rng.standard_normal((M1, M2))
        y = np.fft.fft2(root * w)
        out[made] = y.real[:R, :C]
        made += 1
        if made < n:  # the imaginary part is an independent second field
            out[made] = y.imag[:R, :C]
            made += 1
    return out, neg


def fit_raster_variogram(values, observed, dx, dy, model="spherical", n_lags=15,
                         max_lag=None, nugget=None, psill=None, rng=None, seed=0):
    rows, cols = np.nonzero(observed)
    xy = np.column_stack([cols * dx, rows * dy]).astype(float)
    z = values[observed].astype(float)
    lags, gamma, counts, ml = vg.experimental(xy, z, n_lags=n_lags, max_lag=max_lag,
                                              seed=seed)
    return vg.fit(lags, gamma, counts, model, nugget, psill, rng, max_lag=ml)


def fill_gaps(values, dx, dy, variogram, gaps=None, k=16, max_distance=None,
              min_points=3, n_sims=0, seed=0, want_variance=False, progress=None,
              log=None):
    """Fill missing cells of a 2-D array.

    Parameters
    ----------
    values : 2-D array, ``nan`` marks missing cells.
    dx, dy : pixel width and height (positive, map units).
    variogram : fitted :class:`~.variogram.Variogram`.
    gaps : optional bool array of the cells to fill (default: every ``nan``).
    k, max_distance, min_points : kriging neighbourhood. Cells with fewer than
        ``min_points`` observed cells within ``max_distance`` stay missing.
    n_sims : number of conditional simulations (0 = kriging only).

    Returns a dict with ``kriged`` (filled array), ``variance`` (kriging
    variance, 0 on observed cells, or None), ``simulations`` (n_sims, R, C)
    float32 array or None, ``filled_cells`` and ``unfilled_cells``.
    """
    values = np.asarray(values, float)
    log = log or (lambda msg: None)
    observed = np.isfinite(values)
    target = ~observed if gaps is None else (np.asarray(gaps, bool) & ~observed)
    if not observed.any():
        raise GapFillError("The raster has no valid cells to fill from.")

    result = dict(kriged=values.copy(), variance=None, simulations=None,
                  filled_cells=0, unfilled_cells=int(target.sum()), clipped=0.0)
    if want_variance:
        result["variance"] = np.where(observed, 0.0, np.nan)
    if not target.any():
        if n_sims:
            result["simulations"] = np.repeat(values[None].astype(np.float32), n_sims, 0)
        return result

    # Only observed cells close to a gap can be kriging neighbours.
    ring = int(math.ceil(2.0 * math.sqrt(max(k, 1)))) + 2
    if max_distance and max_distance > 0:
        ring = min(ring, int(math.ceil(max_distance / min(dx, dy))) + 1)
    data = box_dilate(target, ring) & observed
    if not data.any():
        return result

    # Work inside the bounding box of the cells involved.
    rr, cc = np.nonzero(target | data)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    sub_t = target[r0:r1, c0:c1]
    sub_d = data[r0:r1, c0:c1]
    sub_v = values[r0:r1, c0:c1]
    dr, dc = np.nonzero(sub_d)
    tr, tc = np.nonzero(sub_t)
    xy_d = np.column_stack([dc * dx, dr * dy]).astype(float)
    z_d = sub_v[dr, dc]
    q = np.column_stack([tc * dx, tr * dy]).astype(float)
    log("Kriging {:,} gap cells from {:,} nearby observed cells.".format(len(q), len(z_d)))

    sims = None
    if n_sims > 0:
        sims, clipped = unconditional_fields(sub_v.shape, dx, dy, variogram, n_sims,
                                             np.random.default_rng(seed))
        result["clipped"] = clipped
        if clipped > 0.01:
            log("Note: {:.1%} of the simulation spectrum was clipped; the simulated "
                "texture is approximate.".format(clipped))
        s_data = sims[:, dr, dc].astype(float)          # (n, nd)
        s_target = sims[:, tr, tc].astype(float)        # (n, nt)
        sim_vals = np.full((n_sims, len(q)), np.nan)

    kriged = np.full(len(q), np.nan)
    var = np.full(len(q), np.nan)
    for sl, ii, lam, v in kriging_chunks(xy_d, q, variogram, k, max_distance,
                                         min_points, progress):
        ok = np.isfinite(v)
        est = np.sum(lam * z_d[ii], axis=1)
        kriged[sl] = np.where(ok, est, np.nan)
        var[sl] = v
        if sims is not None:
            for s in range(n_sims):
                resid = s_target[s, sl] - np.sum(lam * s_data[s][ii], axis=1)
                sim_vals[s, sl] = np.where(ok, est + resid, np.nan)

    out = result["kriged"]
    out[r0 + tr, c0 + tc] = kriged
    if want_variance:
        result["variance"][r0 + tr, c0 + tc] = var
    if sims is not None:
        full = np.repeat(values[None].astype(np.float32), n_sims, 0)
        full[:, r0 + tr, c0 + tc] = sim_vals
        result["simulations"] = full
    filled = int(np.isfinite(kriged).sum())
    result["filled_cells"] = filled
    result["unfilled_cells"] = len(q) - filled
    return result
