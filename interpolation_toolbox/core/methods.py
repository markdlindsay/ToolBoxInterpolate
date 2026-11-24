"""Point interpolation methods, written against plain numpy.

Every predictor has the signature ``f(xy, z, q, ..., progress=None)`` where
``xy`` is an (n, 2) array of sample coordinates, ``z`` the (n,) sample values
and ``q`` an (m, 2) array of locations to predict. They return an (m,) float
array with ``nan`` wherever no prediction could be made.

``progress`` is an optional callable taking a fraction in [0, 1]; it may raise
:class:`~.neighbors.Cancelled` to abort.
"""

import numpy as np

from .neighbors import NeighborSearch, chunks, pairwise_distances, _report

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def deduplicate(xy, z):
    """Average values that share identical coordinates.

    Coincident samples make kriging and RBF systems singular, so all methods
    run on de-duplicated data. Returns ``(xy, z, n_removed)``.
    """
    xy = np.asarray(xy, float)
    z = np.asarray(z, float)
    uniq, inv = np.unique(xy, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    if len(uniq) == len(xy):
        return xy, z, 0
    sums = np.bincount(inv, weights=z, minlength=len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq))
    return uniq, sums / cnt, len(xy) - len(uniq)


def _normaliser(xy):
    """Centre and scale factor that map coordinates to roughly unit size."""
    centre = xy.mean(axis=0)
    span = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    return centre, (span if span > 0 else 1.0)


def _k_or_all(k, n):
    return n if (k is None or k <= 0 or k > n) else int(k)


# ----------------------------------------------------------------------------
# Inverse distance weighting
# ----------------------------------------------------------------------------


def idw(xy, z, q, power=2.0, k=12, radius=None, min_points=1, smoothing=0.0,
        progress=None, search=None):
    """Inverse distance weighting.

    ``k <= 0`` uses every sample (global IDW). ``smoothing`` is added to the
    distances (``sqrt(d^2 + s^2)``) so the surface no longer passes exactly
    through the samples.
    """
    xy, z, q = np.asarray(xy, float), np.asarray(z, float), np.asarray(q, float)
    search = search or NeighborSearch(xy)
    k = _k_or_all(k, len(z))
    out = np.full(len(q), np.nan)
    for sl in chunks(len(q), k):
        d, i = search.query(q[sl], k, radius)
        valid = i >= 0
        zz = np.where(valid, z[np.maximum(i, 0)], 0.0)
        dd = np.where(valid, np.sqrt(d ** 2 + smoothing ** 2) if smoothing else d, np.inf)
        exact = valid & (dd <= 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(valid & ~exact, 1.0 / dd ** power, 0.0)
            val = np.sum(w * zz, axis=1) / np.sum(w, axis=1)
        hit = exact.any(axis=1)
        if hit.any():
            val[hit] = zz[hit, np.argmax(exact[hit], axis=1)]
        val[valid.sum(axis=1) < max(1, min_points)] = np.nan
        out[sl] = val
        _report(progress, sl.stop / len(q))
    return out


# ----------------------------------------------------------------------------
# Nearest neighbour
# ----------------------------------------------------------------------------


def nearest(xy, z, q, radius=None, progress=None, search=None):
    """Value of the closest sample (Voronoi / Thiessen surface)."""
    xy, z, q = np.asarray(xy, float), np.asarray(z, float), np.asarray(q, float)
    search = search or NeighborSearch(xy)
    out = np.full(len(q), np.nan)
    for sl in chunks(len(q), 64):
        _, i = search.query(q[sl], 1, radius)
        i = i[:, 0]
        out[sl] = np.where(i >= 0, z[np.maximum(i, 0)], np.nan)
        _report(progress, sl.stop / len(q))
    return out


# ----------------------------------------------------------------------------
# Moving window statistics
# ----------------------------------------------------------------------------

STATISTICS = ["mean", "median", "minimum", "maximum", "range", "stdev", "count"]


def moving_window(xy, z, q, radius, statistic="mean", max_points=0, min_points=1,
                  progress=None, search=None):
    """Statistic of all samples within ``radius`` of each location."""
    xy, z, q = np.asarray(xy, float), np.asarray(z, float), np.asarray(q, float)
    search = search or NeighborSearch(xy)
    k = _k_or_all(max_points, len(z))
    out = np.full(len(q), np.nan)
    for sl in chunks(len(q), k):
        _, i = search.query(q[sl], k, radius)
        valid = i >= 0
        cnt = valid.sum(axis=1)
        zz = np.where(valid, z[np.maximum(i, 0)], np.nan)
        with np.errstate(all="ignore"), _quiet():
            if statistic == "mean":
                val = np.nanmean(zz, axis=1)
            elif statistic == "median":
                val = np.nanmedian(zz, axis=1)
            elif statistic == "minimum":
                val = np.nanmin(zz, axis=1)
            elif statistic == "maximum":
                val = np.nanmax(zz, axis=1)
            elif statistic == "range":
                val = np.nanmax(zz, axis=1) - np.nanmin(zz, axis=1)
            elif statistic == "stdev":
                val = np.nanstd(zz, axis=1)
            elif statistic == "count":
                val = cnt.astype(float)
            else:
                raise ValueError("Unknown statistic: %s" % statistic)
        if statistic != "count":
            val[cnt < max(1, min_points)] = np.nan
        out[sl] = val
        _report(progress, sl.stop / len(q))
    return out


class _quiet:
    """Silence numpy's 'All-NaN slice' RuntimeWarnings inside a block."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)


# ----------------------------------------------------------------------------
# Ordinary kriging
# ----------------------------------------------------------------------------


def _batched_solve(A, b):
    try:
        return np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.einsum("cij,cj->ci", np.linalg.pinv(A), b)


def kriging_chunks(xy, q, variogram, k=16, radius=None, min_points=3, progress=None,
                   search=None):
    """Yield ordinary-kriging weights chunk by chunk.

    Each item is ``(slice, idx, lam, var)``: ``idx`` (c, k) sample indices
    (0 where unused), ``lam`` (c, k) weights (0 where unused) and ``var`` (c,)
    kriging variance, ``nan`` where fewer than ``min_points`` samples were
    found. The prediction for any value vector ``z`` is ``sum(lam * z[idx])``,
    so the same weights can krige many data sets (used by conditional
    simulation).
    """
    xy, q = np.asarray(xy, float), np.asarray(q, float)
    search = search or NeighborSearch(xy)
    k = _k_or_all(k, len(xy))
    m = len(q)
    eye = np.eye(k + 1)

    for sl in chunks(m, 3 * (k + 1) ** 2):
        d, i = search.query(q[sl], k, radius)
        valid = i >= 0
        enough = valid.sum(axis=1) >= max(1, min_points)
        ii = np.maximum(i, 0)
        P = xy[ii]
        D = np.sqrt(np.sum((P[:, :, None, :] - P[:, None, :, :]) ** 2, axis=-1))
        vv = valid[:, :, None] & valid[:, None, :]

        c = len(d)
        A = np.zeros((c, k + 1, k + 1))
        A[:, :k, :k] = np.where(vv, variogram(D), 0.0)
        A[:, :k, k] = valid
        A[:, k, :k] = valid
        # neutralise missing neighbours (weight forced to zero)
        diag = np.arange(k)
        A[:, diag, diag] += ~valid
        b = np.zeros((c, k + 1))
        b[:, :k] = np.where(valid, variogram(np.where(valid, d, 0.0)), 0.0)
        b[:, k] = 1.0
        # rows without enough data get a dummy identity system
        A[~enough] = eye
        b[~enough] = 0.0

        sol = _batched_solve(A, b)
        lam = sol[:, :k] * valid
        lam[~enough] = 0.0
        var = np.maximum(np.sum(lam * b[:, :k], axis=1) + sol[:, k], 0.0)
        var[~enough] = np.nan
        yield sl, ii, lam, var
        _report(progress, sl.stop / m)


def ordinary_kriging(xy, z, q, variogram, k=16, radius=None, min_points=3,
                     return_variance=False, progress=None, search=None):
    """Ordinary kriging with a local search neighbourhood.

    ``variogram`` is a callable ``gamma(h)`` (see :mod:`.variogram`).
    ``k <= 0`` uses every sample (only sensible for small datasets).
    Returns predictions, or ``(predictions, kriging_variance)``.
    """
    z = np.asarray(z, float)
    m = len(q)
    pred = np.full(m, np.nan)
    var = np.full(m, np.nan) if return_variance else None
    for sl, ii, lam, v in kriging_chunks(xy, q, variogram, k, radius, min_points,
                                         progress, search):
        p = np.sum(lam * z[ii], axis=1)
        p[np.isnan(v)] = np.nan
        pred[sl] = p
        if return_variance:
            var[sl] = v
    return (pred, var) if return_variance else pred


# ----------------------------------------------------------------------------
# Radial basis functions
# ----------------------------------------------------------------------------

RBF_KERNELS = ["thin_plate", "multiquadric", "inverse_multiquadric", "gaussian",
               "linear", "cubic"]


def _phi(kernel, r, eps):
    with np.errstate(divide="ignore", invalid="ignore"):
        if kernel == "thin_plate":
            return np.where(r > 0, r * r * np.log(np.where(r > 0, r, 1.0)), 0.0)
        if kernel == "multiquadric":
            return np.sqrt(r * r + eps * eps)
        if kernel == "inverse_multiquadric":
            return 1.0 / np.sqrt(r * r + eps * eps)
        if kernel == "gaussian":
            return np.exp(-(r / eps) ** 2)
        if kernel == "linear":
            return r
        if kernel == "cubic":
            return r ** 3
    raise ValueError("Unknown RBF kernel: %s" % kernel)


def mean_spacing(xy):
    """Mean nearest-neighbour distance between samples."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return 1.0
    d, _ = NeighborSearch(xy).query(xy, 2)
    s = float(np.mean(d[:, 1][np.isfinite(d[:, 1])]))
    return s if s > 0 else 1.0


def rbf(xy, z, q, kernel="thin_plate", epsilon=0.0, smoothing=0.0, k=0,
        min_points=3, progress=None, search=None):
    """Radial basis function interpolation with a linear polynomial trend.

    ``k <= 0`` solves one global system (best quality, O(n^3), fine up to a
    few thousand points); ``k > 0`` solves a small system per output cell from
    its ``k`` nearest samples. ``epsilon`` is the shape parameter in map units
    (0 = mean sample spacing). ``smoothing`` > 0 turns exact interpolation into
    a smoothing fit.
    """
    xy, z, q = np.asarray(xy, float), np.asarray(z, float), np.asarray(q, float)
    centre, scale = _normaliser(xy)
    xs = (xy - centre) / scale
    qs = (q - centre) / scale
    eps = (epsilon / scale) if epsilon and epsilon > 0 else mean_spacing(xs)
    n, m = len(z), len(q)
    out = np.full(m, np.nan)

    if k is None or k <= 0 or k >= n:
        A = np.zeros((n + 3, n + 3))
        A[:n, :n] = _phi(kernel, pairwise_distances(xs, xs), eps) + smoothing * np.eye(n)
        A[:n, n] = 1.0
        A[:n, n + 1:] = xs
        A[n, :n] = 1.0
        A[n + 1:, :n] = xs.T
        rhs = np.concatenate([z, np.zeros(3)])
        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A, rhs, rcond=None)[0]
        w, poly = coef[:n], coef[n:]
        for sl in chunks(m, n):
            phi = _phi(kernel, pairwise_distances(qs[sl], xs), eps)
            out[sl] = phi @ w + poly[0] + qs[sl] @ poly[1:]
            _report(progress, sl.stop / m)
        return out

    search = search or NeighborSearch(xs)
    K = k + 3
    eye = np.eye(K)
    for sl in chunks(m, 3 * K * K):
        d, i = search.query(qs[sl], k, None)
        valid = i >= 0
        enough = valid.sum(axis=1) >= max(3, min_points)
        ii = np.maximum(i, 0)
        P = xs[ii] - qs[sl][:, None, :]  # local coordinates for conditioning
        D = np.sqrt(np.sum((P[:, :, None, :] - P[:, None, :, :]) ** 2, axis=-1))
        vv = valid[:, :, None] & valid[:, None, :]
        c = len(d)
        A = np.zeros((c, K, K))
        A[:, :k, :k] = np.where(vv, _phi(kernel, D, eps), 0.0)
        diag = np.arange(k)
        A[:, diag, diag] += np.where(valid, smoothing, 1.0)
        A[:, :k, k] = valid
        A[:, :k, k + 1:] = P * valid[:, :, None]
        A[:, k, :k] = valid
        A[:, k + 1:, :k] = np.swapaxes(P * valid[:, :, None], 1, 2)
        b = np.zeros((c, K))
        b[:, :k] = np.where(valid, z[ii], 0.0)
        A[~enough] = eye
        b[~enough] = 0.0
        coef = _batched_solve(A, b)
        # evaluate at the query point, which is the local origin
        phi0 = np.where(valid, _phi(kernel, np.where(valid, d, 0.0), eps), 0.0)
        val = np.sum(coef[:, :k] * phi0, axis=1) + coef[:, k]
        val[~enough] = np.nan
        out[sl] = val
        _report(progress, sl.stop / m)
    return out


# ----------------------------------------------------------------------------
# Polynomial trend surface
# ----------------------------------------------------------------------------


def _poly_terms(xs, order):
    cols = []
    for total in range(order + 1):
        for j in range(total + 1):
            cols.append(xs[:, 0] ** (total - j) * xs[:, 1] ** j)
    return np.column_stack(cols)


class TrendSurface:
    """Least-squares polynomial surface z = sum a_ij x^i y^j, i + j <= order."""

    def __init__(self, xy, z, order=1):
        xy, z = np.asarray(xy, float), np.asarray(z, float)
        self.order = int(order)
        self.centre, self.scale = _normaliser(xy)
        X = _poly_terms((xy - self.centre) / self.scale, self.order)
        self.coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        fitted = X @ self.coef
        sse = float(np.sum((z - fitted) ** 2))
        sst = float(np.sum((z - z.mean()) ** 2))
        self.r2 = 1.0 - sse / sst if sst > 0 else 1.0
        self.rmse = float(np.sqrt(sse / len(z)))
        self.n_terms = X.shape[1]

    def __call__(self, q):
        q = np.asarray(q, float)
        return _poly_terms((q - self.centre) / self.scale, self.order) @ self.coef


def trend_surface(xy, z, q, order=1, progress=None):
    model = TrendSurface(xy, z, order)
    q = np.asarray(q, float)
    out = np.empty(len(q))
    for sl in chunks(len(q), 64):
        out[sl] = model(q[sl])
        _report(progress, sl.stop / len(q))
    return out


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


def cross_validate(xy, z, predictor, folds=10, seed=0, progress=None):
    """Out-of-fold predictions via k-fold cross-validation.

    ``predictor(xy_train, z_train, xy_test)`` must return predictions for the
    test points. ``folds >= n`` gives leave-one-out.
    """
    xy, z = np.asarray(xy, float), np.asarray(z, float)
    n = len(z)
    folds = int(max(2, min(folds, n)))
    order = np.random.default_rng(seed).permutation(n)
    groups = np.array_split(order, folds)
    pred = np.full(n, np.nan)
    for f, test in enumerate(groups):
        train = np.setdiff1d(order, test, assume_unique=True)
        pred[test] = predictor(xy[train], z[train], xy[test])
        _report(progress, (f + 1) / folds)
    return pred


def error_metrics(observed, predicted):
    """Mean error, MAE, RMSE, R^2 (1 - SSE/SST) and Pearson r, ignoring nan."""
    o = np.asarray(observed, float)
    p = np.asarray(predicted, float)
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    n = len(o)
    if n == 0:
        return dict(n=0, me=np.nan, mae=np.nan, rmse=np.nan, r2=np.nan, r=np.nan)
    e = p - o
    sst = float(np.sum((o - o.mean()) ** 2))
    r = float(np.corrcoef(o, p)[0, 1]) if n > 1 and np.std(p) > 0 and np.std(o) > 0 else np.nan
    return dict(
        n=n,
        me=float(e.mean()),
        mae=float(np.abs(e).mean()),
        rmse=float(np.sqrt(np.mean(e ** 2))),
        r2=(1.0 - float(np.sum(e ** 2)) / sst) if sst > 0 else np.nan,
        r=r,
    )
