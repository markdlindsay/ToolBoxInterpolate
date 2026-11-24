"""Semivariogram estimation and model fitting (pure numpy)."""

import numpy as np

MODELS = ["spherical", "exponential", "gaussian", "linear"]


def model_shape(model, h, rng):
    """Unit-sill shape function f(h) for a variogram model (practical range)."""
    h = np.asarray(h, dtype=float)
    a = max(float(rng), 1e-12)
    r = h / a
    if model == "spherical":
        return np.where(r < 1.0, 1.5 * r - 0.5 * r ** 3, 1.0)
    if model == "exponential":
        return 1.0 - np.exp(-3.0 * r)
    if model == "gaussian":
        return 1.0 - np.exp(-3.0 * r ** 2)
    if model == "linear":
        return np.minimum(r, 1.0)
    raise ValueError("Unknown variogram model: %s" % model)


class Variogram:
    """A fitted variogram model ``gamma(h) = nugget + psill * f(h / range)``."""

    def __init__(self, model, nugget, psill, rng):
        self.model = model
        self.nugget = float(nugget)
        self.psill = float(psill)
        self.range = float(rng)

    @property
    def sill(self):
        return self.nugget + self.psill

    def __call__(self, h):
        h = np.asarray(h, dtype=float)
        g = self.nugget + self.psill * model_shape(self.model, h, self.range)
        return np.where(h > 0, g, 0.0)

    def describe(self):
        return "%s model: nugget=%.6g, partial sill=%.6g, range=%.6g" % (
            self.model, self.nugget, self.psill, self.range)


def experimental(xy, z, n_lags=15, max_lag=None, max_points=2500, seed=0):
    """Empirical semivariogram.

    Returns ``(lag_mean_distance, semivariance, pair_count, max_lag)``; empty
    bins are dropped. Large datasets are randomly subsampled to ``max_points``
    so that the pair count stays manageable.
    """
    xy = np.asarray(xy, float)
    z = np.asarray(z, float)
    n = len(z)
    if n > max_points:
        sel = np.random.default_rng(seed).choice(n, max_points, replace=False)
        xy, z = xy[sel], z[sel]
        n = max_points
    iu, ju = np.triu_indices(n, k=1)
    d = np.hypot(xy[iu, 0] - xy[ju, 0], xy[iu, 1] - xy[ju, 1])
    sv = 0.5 * (z[iu] - z[ju]) ** 2
    if max_lag is None or max_lag <= 0:
        max_lag = 0.5 * d.max() if len(d) else 1.0
    keep = d <= max_lag
    d, sv = d[keep], sv[keep]
    edges = np.linspace(0.0, max_lag, int(n_lags) + 1)
    b = np.clip(np.digitize(d, edges) - 1, 0, int(n_lags) - 1)
    counts = np.bincount(b, minlength=n_lags).astype(float)
    sum_d = np.bincount(b, weights=d, minlength=n_lags)
    sum_g = np.bincount(b, weights=sv, minlength=n_lags)
    ok = counts > 0
    return sum_d[ok] / counts[ok], sum_g[ok] / counts[ok], counts[ok], float(max_lag)


def fit(lags, gamma, counts, model="spherical", nugget=None, psill=None, rng=None,
        max_lag=None):
    """Weighted least-squares fit of a variogram model.

    Any of ``nugget``, ``psill`` or ``rng`` may be fixed by passing a value;
    ``None`` means "estimate it". The range is found with a 1-D search, and for
    each candidate range the nugget / partial sill are solved linearly (and
    kept non-negative), which is robust without needing scipy.optimize.
    """
    lags = np.asarray(lags, float)
    gamma = np.asarray(gamma, float)
    w = np.asarray(counts, float)
    if max_lag is None:
        max_lag = float(lags.max()) if len(lags) else 1.0

    if len(lags) == 0:  # not enough data: fall back to something usable
        return Variogram(model, nugget or 0.0, psill or 1.0, rng or max_lag)

    candidates = [float(rng)] if rng else np.geomspace(0.02 * max_lag, 2.0 * max_lag, 120)
    best = None
    for a in candidates:
        f = model_shape(model, lags, a)
        n0, p0 = _solve_np(f, gamma, w, nugget, psill)
        sse = float(np.sum(w * (n0 + p0 * f - gamma) ** 2))
        if best is None or sse < best[0]:
            best = (sse, n0, p0, a)
    _, n0, p0, a = best
    if n0 + p0 <= 0:  # constant data: any positive sill gives the mean
        p0 = 1.0
    return Variogram(model, n0, p0, a)


def _solve_np(f, g, w, nugget, psill):
    """Solve for (nugget, psill) >= 0 given shape values f and targets g."""
    sw = np.sum(w)
    if nugget is not None and psill is not None:
        return float(nugget), float(psill)
    if nugget is not None:
        den = np.sum(w * f * f)
        p = np.sum(w * f * (g - nugget)) / den if den > 0 else 0.0
        return float(nugget), max(float(p), 0.0)
    if psill is not None:
        n = np.sum(w * (g - psill * f)) / sw
        return max(float(n), 0.0), float(psill)
    # both free: weighted linear regression g ~ n + p f
    A = np.column_stack([np.ones_like(f), f]) * np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq(A, g * np.sqrt(w), rcond=None)
    n, p = float(sol[0]), float(sol[1])
    if n < 0:
        den = np.sum(w * f * f)
        n, p = 0.0, (np.sum(w * f * g) / den if den > 0 else 0.0)
    if p < 0:
        n, p = float(np.sum(w * g) / sw), 0.0
    return max(n, 0.0), max(p, 0.0)
