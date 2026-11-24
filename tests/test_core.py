"""Stand-alone tests for the numpy interpolation engine.

Run with:  python -m pytest tests   (or simply: python tests/test_core.py)
They need only numpy; QGIS is not required.
"""

import os
import sys
import importlib.util

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interpolation_toolbox")


def _load_core():
    """Import interpolation_toolbox.core without running the plugin __init__ (needs QGIS)."""
    pkg = "itb_core"
    spec = importlib.util.spec_from_file_location(
        pkg, os.path.join(ROOT, "core", "__init__.py"),
        submodule_search_locations=[os.path.join(ROOT, "core")])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    spec.loader.exec_module(mod)
    import importlib as il
    return (il.import_module(pkg + ".methods"), il.import_module(pkg + ".variogram"),
            il.import_module(pkg + ".neighbors"), il.import_module(pkg + ".gapfill"))


M, V, N, G = _load_core()


def surface(xy):
    x, y = xy[:, 0], xy[:, 1]
    return np.sin(x / 15.0) * np.cos(y / 20.0) * 10 + 0.05 * x


def data(n=300, seed=1):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (n, 2))
    return xy, surface(xy)


def grid(step=2.0):
    g = np.arange(1, 100, step)
    X, Y = np.meshgrid(g, g)
    return np.column_stack([X.ravel(), Y.ravel()])


def rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def test_neighbors_brute_matches_scipy_or_self():
    xy, _ = data(200)
    q = grid(7)
    d1, i1 = N.NeighborSearch(xy, use_scipy=False).query(q, 5)
    full = N.pairwise_distances(q, xy)
    assert np.allclose(np.sort(full, axis=1)[:, :5], d1)
    d2, i2 = N.NeighborSearch(xy, use_scipy=False).query(q, 5, radius=8)
    assert np.all((d2 <= 8) | (i2 == -1))
    assert np.all(np.isinf(d2[i2 == -1]))


def test_exact_methods_honour_samples():
    xy, z = data(120)
    for name, f in [
        ("idw", lambda: M.idw(xy, z, xy)),
        ("nearest", lambda: M.nearest(xy, z, xy)),
        ("rbf_global", lambda: M.rbf(xy, z, xy)),
        ("rbf_local", lambda: M.rbf(xy, z, xy, k=20)),
        ("kriging", lambda: M.ordinary_kriging(xy, z, xy, V.Variogram("spherical", 0, 20, 40))),
    ]:
        assert np.allclose(f(), z, atol=1e-6), name


def test_accuracy_on_smooth_surface():
    xy, z = data(400)
    q = grid()
    truth = surface(q)
    lags, g, c, ml = V.experimental(xy, z)
    vg = V.fit(lags, g, c, "spherical", max_lag=ml)
    res = {
        "idw": rmse(M.idw(xy, z, q), truth),
        "nearest": rmse(M.nearest(xy, z, q), truth),
        "mean": rmse(M.moving_window(xy, z, q, radius=8), truth),
        "kriging": rmse(M.ordinary_kriging(xy, z, q, vg), truth),
        "rbf_tps": rmse(M.rbf(xy, z, q), truth),
        "rbf_local": rmse(M.rbf(xy, z, q, k=30), truth),
        "trend3": rmse(M.trend_surface(xy, z, q, 3), truth),
    }
    print(res, vg.describe())
    assert res["kriging"] < res["idw"] < res["nearest"]
    assert res["rbf_tps"] < 0.6 and res["kriging"] < 0.6


def test_kriging_variance_and_radius():
    xy, z = data(150)
    q = np.array([[50.0, 50.0], [500.0, 500.0]])
    vg = V.Variogram("exponential", 0.5, 20, 40)
    p, v = M.ordinary_kriging(xy, z, q, vg, radius=30, return_variance=True)
    assert np.isfinite(p[0]) and np.isnan(p[1]) and np.isnan(v[1])
    _, v0 = M.ordinary_kriging(xy, z, xy[:3], vg, return_variance=True)
    assert np.all(v0 < 1e-6)


def test_variogram_fit_recovers_parameters():
    rng = np.random.default_rng(3)
    h = np.linspace(1, 60, 25)
    true = V.Variogram("spherical", 2.0, 8.0, 35.0)
    g = true(h) + rng.normal(0, 0.05, h.size)
    fit = V.fit(h, g, np.ones_like(h), "spherical")
    assert abs(fit.nugget - 2) < 0.3 and abs(fit.psill - 8) < 0.4 and abs(fit.range - 35) < 3


def test_trend_surface_exact_for_plane():
    xy, _ = data(50)
    z = 3 + 2 * xy[:, 0] - 0.5 * xy[:, 1]
    m = M.TrendSurface(xy, z, 1)
    assert m.r2 > 0.999999
    assert np.allclose(m(xy), z)


def test_duplicates_and_cv():
    xy = np.array([[0, 0], [0, 0], [1, 1], [2, 0.5]], float)
    z = np.array([1, 3, 5, 7], float)
    xy2, z2, rem = M.deduplicate(xy, z)
    assert rem == 1 and len(z2) == 3 and 2.0 in z2
    xy, z = data(200)
    pred = M.cross_validate(xy, z, lambda a, b, c: M.idw(a, b, c), folds=10)
    met = M.error_metrics(z, pred)
    assert met["n"] == 200 and met["r2"] > 0.8


def test_moving_window_stats():
    xy = np.array([[0, 0], [1, 0], [0, 1], [10, 10]], float)
    z = np.array([1, 2, 6, 100], float)
    q = np.array([[0.2, 0.2]])
    assert M.moving_window(xy, z, q, 2, "mean")[0] == 3
    assert M.moving_window(xy, z, q, 2, "median")[0] == 2
    assert M.moving_window(xy, z, q, 2, "range")[0] == 5
    assert M.moving_window(xy, z, q, 2, "count")[0] == 3
    assert np.isnan(M.moving_window(xy, z, np.array([[50.0, 50]]), 2, "mean")[0])


def test_constant_data_is_stable():
    xy, _ = data(60)
    z = np.full(60, 4.2)
    lags, g, c, ml = V.experimental(xy, z)
    vg = V.fit(lags, g, c, "gaussian", max_lag=ml)
    q = grid(10)
    assert np.allclose(M.ordinary_kriging(xy, z, q, vg), 4.2)
    assert np.allclose(M.rbf(xy, z, q), 4.2, atol=1e-6)


# ---------------------------------------------------------------- gap filling


def test_box_dilate_matches_naive():
    rng = np.random.default_rng(0)
    m = rng.random((30, 41)) > 0.97
    for w in (0, 1, 3):
        naive = np.zeros_like(m)
        for r, c in zip(*np.nonzero(m)):
            naive[max(r - w, 0):r + w + 1, max(c - w, 0):c + w + 1] = True
        assert np.array_equal(G.box_dilate(m, w), naive)


def test_unconditional_simulation_reproduces_variogram():
    v = V.Variogram("exponential", 0.0, 4.0, 30.0)
    fields, clipped = G.unconditional_fields((64, 64), 1.0, 1.0, v, 200,
                                             np.random.default_rng(1))
    assert clipped < 0.01
    assert abs(fields.var() - 4.0) < 0.4
    for lag in (2, 8, 20):
        g = 0.5 * np.mean((fields[:, :, lag:] - fields[:, :, :-lag]) ** 2)
        assert abs(g - v(lag)) < 0.15 * v(lag) + 0.05, (lag, g, v(lag))


def _gappy(seed=0):
    rng = np.random.default_rng(seed)
    r, c = np.mgrid[0:80, 0:100]
    truth = np.sin(c / 9.0) * np.cos(r / 11.0) * 5 + 0.05 * c
    vals = truth.copy()
    vals[30:45, 40:60] = np.nan              # one big hole
    vals[rng.random(vals.shape) < 0.2] = np.nan  # scattered drop-outs
    return truth, vals


def test_fill_gaps_kriging_accuracy():
    truth, vals = _gappy()
    obs = np.isfinite(vals)
    vgm = G.fit_raster_variogram(vals, obs, 1.0, 1.0, "gaussian")
    res = G.fill_gaps(vals, 1.0, 1.0, vgm, want_variance=True)
    out = res["kriged"]
    assert np.isfinite(out).all() and res["unfilled_cells"] == 0
    assert np.array_equal(out[obs], vals[obs])
    err = np.abs(out - truth)[~obs]
    assert err.mean() < 0.15, err.mean()
    var = res["variance"]
    assert np.all(var[obs] == 0)
    assert var[37, 50] > var[~obs].mean()  # centre of the big hole is least certain


def test_conditional_simulation_statistics():
    truth, vals = _gappy(1)
    obs = np.isfinite(vals)
    vgm = V.Variogram("spherical", 0.0, 12.0, 40.0)
    res = G.fill_gaps(vals, 1.0, 1.0, vgm, n_sims=300, want_variance=True, seed=3)
    sims = res["simulations"].astype(float)
    assert sims.shape == (300,) + vals.shape
    assert np.all(sims[:, obs] == vals[obs].astype(np.float32))
    hole = np.zeros_like(obs)
    hole[30:45, 40:60] = True
    mean_err = np.abs(sims.mean(0) - res["kriged"])[hole].mean()
    ratio = sims.var(0)[hole].mean() / res["variance"][hole].mean()
    assert mean_err < 0.15 * np.sqrt(res["variance"][hole].mean()) + 0.05, mean_err
    assert 0.85 < ratio < 1.15, ratio


def test_fill_gaps_max_distance_and_mask():
    _, vals = _gappy()
    vgm = V.Variogram("spherical", 0.0, 10.0, 40.0)
    res = G.fill_gaps(vals, 1.0, 1.0, vgm, max_distance=3.0)
    assert np.isnan(res["kriged"][37, 50]) and res["unfilled_cells"] > 0
    allowed = np.zeros(vals.shape, bool)
    allowed[:, :50] = True
    res = G.fill_gaps(vals, 1.0, 1.0, vgm, gaps=allowed)
    assert np.isfinite(res["kriged"][:, :50]).all()
    assert np.isnan(res["kriged"][:, 50:]).sum() == np.isnan(vals[:, 50:]).sum()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
