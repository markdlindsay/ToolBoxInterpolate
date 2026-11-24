# Point Interpolation Toolbox for QGIS

A QGIS Processing plugin that adds a **Point Interpolation** provider with many ways to turn point samples into a raster surface, plus tools for choosing between them.

| Group | Tool | Notes |
|---|---|---|
| Deterministic | Inverse distance weighting (IDW) | power, k neighbours or all, search radius, smoothing |
| Deterministic | Nearest neighbour (Thiessen / Voronoi) | for categorical or coded values |
| Deterministic | Moving window statistics | mean, median, min, max, range, std. dev., count |
| Deterministic | Triangulated irregular network (TIN) | linear or Clough-Tocher, uses QGIS's TIN engine |
| Splines and functions | Radial basis functions / splines | thin plate, multiquadric, inverse multiquadric, Gaussian, linear, cubic; exact or smoothing; global or local |
| Splines and functions | Polynomial trend surface | order 1–6, R² in log, optional residual points |
| Geostatistics | Kriging (ordinary / with trend) | spherical, exponential, Gaussian, linear models; auto-fitted variogram; optional variance raster and variogram table |
| Geostatistics | Variogram analysis | experimental variogram plus a fit of every model, with the best one flagged |
| Geostatistics | Fill raster gaps | fills a raster's no-data cells by kriging (best estimate) or conditional simulation (realistic texture, one band per realisation); optional uncertainty raster, fill polygons and maximum fill distance |
| Validation | Compare methods (cross-validation) | k-fold or leave-one-out; MAE, RMSE, R², bias; optional per-point errors |

Every grid tool takes the same inputs:

- a point layer (multipoints are fine)
- a numeric field, or the geometry's Z value
- an optional extent and a pixel size (0 picks about 400 cells on the long side)
- optional mask polygons

Output is any GDAL raster format (GeoTIFF by default). Null values are skipped and points at the same location are averaged.

## Requirements

- QGIS 3.22 or later (tested on 3.44).
- Only **numpy** is needed, and QGIS always ships it. If **scipy** is installed it is used for faster neighbour searches. Without it, a built-in numpy spatial index is used: 20,000 points onto a 500 × 500 grid takes about 1 s for IDW and about 4 s for kriging.

## Install

Copy or symlink the `interpolation_toolbox` folder into your QGIS plugins directory, then enable **Point Interpolation Toolbox** in *Plugins → Manage and Install Plugins*. On macOS:

```bash
ln -s "$PWD/interpolation_toolbox" ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/interpolation_toolbox
```

Alternatively, run `scripts/build_zip.sh` to build a clean `dist/interpolation_toolbox-<version>.zip` and use *Install from ZIP*.

If the **qpip** plugin is installed, it re-checks the `requirements.txt` of every active plugin whenever any plugin loads. qpip 1.5.1 can crash with `IndexError` in `check_already_installed` because of another plugin's requirements. This plugin has none. If that happens, restart QGIS and enable the plugin in the Plugin Manager. The tools then appear under **Processing Toolbox → Point Interpolation**.

## Choosing a method

1. Run **Compare methods (cross-validation)** on your layer. The method with the lowest RMSE is usually the best choice.
2. For kriging, run **Variogram analysis** first to pick a model, and use the trend option if the data has a regional gradient.
3. To repair a raster with holes (clouds, sensor drop-outs, DEM voids), use **Fill raster gaps**. Choose kriging for a single best-guess surface. Choose conditional simulation when the filled area should look like real data, or when you need several plausible versions for uncertainty analysis. Simulation fits a variogram to the valid cells and simulates a Gaussian field with FFT circulant embedding. It then conditions that field on the observed cells by kriging the residuals. On a 2000 × 2000 raster with about 257,000 missing cells, kriging takes about 13 s and four realisations about 15 s.
4. Use **IDW** or **moving window** for quick, predictable surfaces. Use **TIN** for dense, exact data such as survey points. Use **thin plate spline** or **kriging** for smooth natural phenomena. Use **nearest neighbour** for categories.

## Layout

```
interpolation_toolbox/
  core/          pure-numpy engine (no QGIS imports): methods, variogram, neighbour search, gap filling
  algorithms/    one QGIS Processing algorithm per tool; base.py holds shared I/O
  provider.py    registers the algorithms
tests/
  test_core.py   engine tests, need only numpy:  python3 tests/test_core.py
  test_qgis.py   runs every tool through Processing, headless (needs QGIS's Python)
```

## Running the tests

`tests/test_core.py` runs with any Python that has numpy.

`tests/test_qgis.py` needs the Python bundled with QGIS. On macOS (adjust the app name to your version):

```bash
Q=/Applications/QGIS-final-3_44_5.app/Contents; R=$Q/Resources/python3.11
PYTHONHOME=$R PYTHONPATH=$R:$R/lib-dynload:$R/site-packages:$Q/Resources/qgis/python/plugins \
QGIS_PREFIX_PATH=$Q/MacOS QT_QPA_PLATFORM=offscreen PROJ_DATA=$Q/Resources/qgis/proj \
GDAL_DATA=$Q/Resources/qgis/gdal $Q/MacOS/python3.12 tests/test_qgis.py
```
