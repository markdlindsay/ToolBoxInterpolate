"""Integration test: runs every algorithm through QGIS Processing, headless.

Run with the Python that ships with QGIS, e.g. on macOS see README.md
("Running the tests"). Exits non-zero on failure.
"""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qgis.core import (  # noqa: E402
    QgsApplication, QgsFeature, QgsField, QgsGeometry, QgsPoint, QgsPointXY,
    QgsProcessingFeedback, QgsRasterLayer, QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant  # noqa: E402

app = QgsApplication([], False)
app.initQgis()

from processing.core.Processing import Processing  # noqa: E402
import processing  # noqa: E402

Processing.initialize()

from interpolation_toolbox.provider import PointInterpolationProvider  # noqa: E402

provider = PointInterpolationProvider()
QgsApplication.processingRegistry().addProvider(provider)

OUT = tempfile.mkdtemp(prefix="interp_test_")
rng = np.random.default_rng(7)
XY = rng.uniform(0, 1000, (250, 2))
Z = np.sin(XY[:, 0] / 150) * np.cos(XY[:, 1] / 200) * 10 + 0.01 * XY[:, 0]


def point_layer(with_z=False):
    uri = ("PointZ" if with_z else "Point") + "?crs=EPSG:32633&field=value:double&field=label:string"
    lyr = QgsVectorLayer(uri, "pts", "memory")
    feats = []
    for (x, y), z in zip(XY, Z):
        f = QgsFeature(lyr.fields())
        f.setGeometry(QgsGeometry(QgsPoint(x, y, z)) if with_z
                      else QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        f.setAttributes([float(z), "a"])
        feats.append(f)
    # one null value and one duplicate location, which must be handled
    f = QgsFeature(lyr.fields())
    f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(500, 500)))
    f.setAttributes([None, "null"])
    feats.append(f)
    f = QgsFeature(lyr.fields())
    f.setGeometry(QgsGeometry(QgsPoint(XY[0, 0], XY[0, 1], Z[0])) if with_z
                  else QgsGeometry.fromPointXY(QgsPointXY(*XY[0])))
    f.setAttributes([float(Z[0]), "dup"])
    feats.append(f)
    lyr.dataProvider().addFeatures(feats)
    return lyr


def mask_layer():
    lyr = QgsVectorLayer("Polygon?crs=EPSG:32633", "mask", "memory")
    f = QgsFeature()
    f.setGeometry(QgsGeometry.fromWkt("POLYGON((0 0, 600 0, 600 600, 0 600, 0 0))"))
    lyr.dataProvider().addFeatures([f])
    return lyr


def truth(x, y):
    return np.sin(x / 150) * np.cos(y / 200) * 10 + 0.01 * x


def check_raster(path, max_rmse=None, expect_nodata=False):
    r = QgsRasterLayer(path, "r")
    assert r.isValid(), path
    assert r.crs().authid() == "EPSG:32633", r.crs().authid()
    from osgeo import gdal
    ds = gdal.Open(path)
    a = ds.GetRasterBand(1).ReadAsArray().astype(float)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    gt = ds.GetGeoTransform()
    a[a == nd] = np.nan
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    X = gt[0] + (xs + 0.5) * gt[1]
    Y = gt[3] + (ys + 0.5) * gt[5]
    ok = np.isfinite(a)
    assert ok.any(), "all nodata: " + path
    if expect_nodata:
        assert (~ok).any(), "expected some nodata: " + path
    err = float(np.sqrt(np.mean((a[ok] - truth(X[ok], Y[ok])) ** 2)))
    if max_rmse is not None:
        assert err < max_rmse, "%s rmse %.3f" % (path, err)
    return a.shape, err


def run(alg, params):
    fb = QgsProcessingFeedback()
    return processing.run("pointinterpolation:" + alg, params, feedback=fb)


pts = point_layer()
ptsz = point_layer(with_z=True)
base = dict(INPUT=pts, FIELD="value", PIXEL_SIZE=10)
report = []

for alg, extra, tol in [
    ("idw", {}, 2.0),
    ("idw", {"NEIGHBORS": 0, "POWER": 3}, 2.0),
    ("nearestneighbour", {}, 2.5),
    ("movingwindow", {"STATISTIC": 1}, 2.5),
    ("tin", {"METHOD": 0}, 1.5),
    ("tin", {"METHOD": 1}, 1.5),
    ("rbf", {}, 1.0),
    ("rbf", {"KERNEL": 1, "NEIGHBORS": 30}, 1.0),
    ("trendsurface", {"ORDER": 3}, 6.0),
    ("kriging", {}, 1.2),
    ("kriging", {"MODEL": 1, "DRIFT": 1, "RADIUS": 200}, 1.2),
]:
    out = os.path.join(OUT, "%s_%d.tif" % (alg, len(report)))
    res = run(alg, dict(base, OUTPUT=out, **extra))
    shape, err = check_raster(res["OUTPUT"], tol)
    report.append((alg, extra, shape, round(err, 3)))

# Z values instead of a field, custom extent, mask, ASCII grid output
out = os.path.join(OUT, "idw_z.asc")
res = run("idw", dict(INPUT=ptsz, FIELD=None, EXTENT="0,800,0,800 [EPSG:32633]",
                      PIXEL_SIZE=20, MASK=mask_layer(), OUTPUT=out))
shape, err = check_raster(res["OUTPUT"], 2.0, expect_nodata=True)
assert shape == (40, 40), shape
report.append(("idw Z+extent+mask .asc", {}, shape, round(err, 3)))

# kriging extras
res = run("kriging", dict(base, OUTPUT=os.path.join(OUT, "k.tif"),
                          VARIANCE=os.path.join(OUT, "kvar.tif"), VARIOGRAM="memory:"))
assert os.path.exists(res["VARIANCE"])
vt = context_layer = res["VARIOGRAM"]
report.append(("kriging variance + variogram", {}, None, None))

# variogram analysis
res = run("variogram", dict(INPUT=pts, FIELD="value", OUTPUT="memory:", MODELS="memory:"))
assert res["OUTPUT"].featureCount() > 5 and res["MODELS"].featureCount() == 4
best = [f["model"] for f in res["MODELS"].getFeatures() if f["best"] == 1]
report.append(("variogram", {"best": best}, None, None))

# cross-validation
res = run("crossvalidation", dict(INPUT=pts, FIELD="value", FOLDS=5,
                                  OUTPUT="memory:", POINTS="memory:"))
rows = sorted(res["OUTPUT"].getFeatures(), key=lambda f: f["rank"])
assert len(rows) == 9 and res["POINTS"].featureCount() == 250
for f in rows:
    print("  CV %d %-45s RMSE %.3f R2 %.3f" % (f["rank"], f["method"], f["rmse"], f["r2"]))

# trend residuals
res = run("trendsurface", dict(base, ORDER=2, OUTPUT=os.path.join(OUT, "t.tif"),
                               RESIDUALS="memory:"))
assert res["RESIDUALS"].featureCount() == 250

# ------------------------------------------------------------ raster gap filling
from osgeo import gdal, osr  # noqa: E402

rows, cols, px = 60, 80, 10.0
rr, cc = np.mgrid[0:rows, 0:cols]
gx, gy = 500000 + (cc + 0.5) * px, 6000000 - (rr + 0.5) * px
full = truth(gx - 500000, gy - 6000000 + rows * px)
gappy = full.copy()
gappy[20:32, 30:50] = -9999.0                       # nodata hole
gappy[np.random.default_rng(2).random(gappy.shape) < 0.15] = -9999.0
gappy[5:8, 5:8] = 123.0                             # "extra" gap value
src_path = os.path.join(OUT, "gappy.tif")
ds = gdal.GetDriverByName("GTiff").Create(src_path, cols, rows, 1, gdal.GDT_Float32)
ds.SetGeoTransform((500000, px, 0, 6000000, 0, -px))
srs = osr.SpatialReference()
srs.ImportFromEPSG(32633)
ds.SetProjection(srs.ExportToWkt())
ds.GetRasterBand(1).SetNoDataValue(-9999.0)
ds.GetRasterBand(1).WriteArray(gappy)
ds = None
gap_cells = (gappy == -9999.0) | (gappy == 123.0)


def read(path, band=None):
    d = gdal.Open(path)
    bands = range(1, d.RasterCount + 1) if band is None else [band]
    out = []
    for b in bands:
        a = d.GetRasterBand(b).ReadAsArray().astype(float)
        a[a == d.GetRasterBand(b).GetNoDataValue()] = np.nan
        out.append(a)
    assert d.GetGeoTransform() == (500000, px, 0, 6000000, 0, -px)
    return np.array(out)


res = run("fillrastergaps", dict(INPUT=src_path, BAND=1, METHOD=0, MODEL=2, EXTRA_NODATA=123,
                                 OUTPUT=os.path.join(OUT, "filled_k.tif"),
                                 UNCERTAINTY=os.path.join(OUT, "filled_k_sd.tif")))
k = read(res["OUTPUT"])[0]
sd = read(res["UNCERTAINTY"])[0]
assert np.isfinite(k).all()
assert np.allclose(k[~gap_cells], gappy[~gap_cells], atol=1e-4)
k_err = float(np.abs(k - full)[gap_cells].mean())
assert k_err < 0.3, k_err
assert np.all(sd[~gap_cells] == 0) and sd[26, 40] > sd[gap_cells].mean()
report.append(("gap fill kriging", {"mean abs err in gaps": round(k_err, 3)}, k.shape, None))

res = run("fillrastergaps", dict(INPUT=src_path, METHOD=1, N_SIMS=5, SEED=4,
                                 OUTPUT=os.path.join(OUT, "filled_s.tif"),
                                 UNCERTAINTY=os.path.join(OUT, "filled_s_sd.tif")))
sims = read(res["OUTPUT"])
assert sims.shape == (5, rows, cols)
nd_cells = gappy == -9999.0
assert np.isfinite(sims).all()
assert np.allclose(sims[:, ~nd_cells], gappy[~nd_cells], atol=1e-4)
assert not np.allclose(sims[0][nd_cells], sims[1][nd_cells])  # realisations differ
ssd = read(res["UNCERTAINTY"])[0]
assert np.all(ssd[~nd_cells] == 0) and np.all(ssd[nd_cells] > 0)
report.append(("gap fill simulation x5", {}, sims.shape, None))

mask = QgsVectorLayer("Polygon?crs=EPSG:32633", "m", "memory")
f = QgsFeature()
f.setGeometry(QgsGeometry.fromWkt(
    "POLYGON((500000 6000000, 500400 6000000, 500400 5999400, 500000 5999400, 500000 6000000))"))
mask.dataProvider().addFeatures([f])
res = run("fillrastergaps", dict(INPUT=src_path, FILL_MASK=mask, MAX_DISTANCE=25,
                                 OUTPUT=os.path.join(OUT, "filled_m.tif")))
m = read(res["OUTPUT"])[0]
assert np.isnan(m[:, 40:][nd_cells[:, 40:]]).all()          # outside polygon: untouched
scattered = nd_cells.copy()
scattered[20:32, 30:50] = False                             # drop-outs outside the hole
assert np.isfinite(m[:, :40][scattered[:, :40]]).mean() > 0.99
assert np.isnan(m[26, 38])                                  # >25 m into the hole
report.append(("gap fill mask + max distance", {}, m.shape, None))

for r in report:
    print(r)
print("ALL QGIS TESTS PASSED, outputs in", OUT)
app.exitQgis()
