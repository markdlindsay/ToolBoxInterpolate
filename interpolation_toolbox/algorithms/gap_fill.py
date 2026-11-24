import numpy as np
from osgeo import gdal

from qgis.core import (QgsProcessing, QgsProcessingException, QgsProcessingParameterBand,
                       QgsProcessingParameterEnum, QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterRasterLayer)

from ..core import gapfill
from ..core.neighbors import Cancelled
from .base import PointAlgorithm, advanced, number, rasterize_polygons, write_array
from .kriging import add_variogram_parameters, read_variogram_settings
from ..core import variogram as vg

MAX_RASTER_CELLS = 100_000_000


class GapFillAlgorithm(PointAlgorithm):
    INPUT = "INPUT"
    BAND = "BAND"
    EXTRA_NODATA = "EXTRA_NODATA"
    METHOD = "METHOD"
    NEIGHBORS = "NEIGHBORS"
    MAX_DISTANCE = "MAX_DISTANCE"
    MIN_POINTS = "MIN_POINTS"
    FILL_MASK = "FILL_MASK"
    N_SIMS = "N_SIMS"
    SEED = "SEED"
    OUTPUT = "OUTPUT"
    UNCERTAINTY = "UNCERTAINTY"

    _group, _group_id = "Geostatistics", "geostatistics"

    def name(self):
        return "fillrastergaps"

    def displayName(self):
        return self.tr("Fill raster gaps (kriging / conditional simulation)")

    def shortHelpString(self):
        return self.tr(
            "Fills no-data cells of a raster band from the surrounding valid cells, for "
            "example cloud gaps in satellite products, sensor drop-outs or holes in a DEM. "
            "A variogram is fitted to the valid cells and observed cells are never "
            "changed.<br><br>"
            "<b>Kriging</b>: the best estimate for every missing cell. It is smooth inside "
            "large gaps.<br>"
            "<b>Conditional simulation</b>: realisations that honour the observed cells "
            "and reproduce the data's spatial texture (variogram) inside the gaps. Each "
            "realisation is one output band; together they describe the uncertainty of "
            "the fill. Their per-cell mean converges to the kriged value.<br><br>"
            "<b>Maximum fill distance</b>: gaps further than this from valid data stay "
            "no-data (0 = fill everything).<br>"
            "<b>Fill only inside polygons</b>: e.g. a land mask, so the sea is not filled."
            "<br>"
            "<b>Uncertainty</b> (optional): the standard deviation of the fill; 0 on "
            "observed cells. For simulation with 2 or more realisations it is the spread "
            "of the realisations, otherwise the kriging standard deviation.<br><br>"
            "Simulation assumes roughly Gaussian values; strongly skewed data (e.g. "
            "concentrations) are best log-transformed first.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, self.tr("Raster with gaps")))
        self.addParameter(QgsProcessingParameterBand(
            self.BAND, self.tr("Band"), 1, self.INPUT))
        self.addParameter(QgsProcessingParameterEnum(
            self.METHOD, self.tr("Method"),
            [self.tr("Kriging (best estimate)"), self.tr("Conditional simulation")],
            defaultValue=0))
        add_variogram_parameters(self)
        self.addParameter(number(self.NEIGHBORS, self.tr("Neighbours"), "int", 16, 1, 128))
        self.addParameter(number(self.MAX_DISTANCE,
                                 self.tr("Maximum fill distance (0 = fill all gaps)"),
                                 default=0.0, minimum=0.0))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.FILL_MASK, self.tr("Fill only inside these polygons"),
            [QgsProcessing.TypeVectorPolygon], optional=True))
        self.addParameter(number(self.N_SIMS, self.tr("Number of realisations (simulation)"),
                                 "int", 1, 1, 1000))
        self.addParameter(advanced(number(
            self.EXTRA_NODATA, self.tr("Also treat this value as a gap"),
            default=None, optional=True)))
        self.addParameter(advanced(number(self.MIN_POINTS, self.tr("Minimum points"),
                                          "int", 3, 1)))
        self.addParameter(advanced(number(self.SEED, self.tr("Random seed"), "int", 0, 0)))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, self.tr("Filled raster")))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.UNCERTAINTY, self.tr("Uncertainty (standard deviation)"), optional=True,
            createByDefault=False))

    def _read_band(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException(self.invalidRasterError(parameters, self.INPUT))
        ds = gdal.Open(layer.source())
        if ds is None:
            raise QgsProcessingException(self.tr(
                "Could not open the raster with GDAL. Save it to a file (e.g. GeoTIFF) "
                "first."))
        gt = ds.GetGeoTransform()
        if gt[2] != 0 or gt[4] != 0:
            raise QgsProcessingException(self.tr("Rotated rasters are not supported."))
        if ds.RasterXSize * ds.RasterYSize > MAX_RASTER_CELLS:
            raise QgsProcessingException(self.tr(
                "The raster has {:,} cells; clip or resample it first.")
                .format(ds.RasterXSize * ds.RasterYSize))
        band_no = self.parameterAsInt(parameters, self.BAND, context) or 1
        band = ds.GetRasterBand(band_no)
        values = band.ReadAsArray().astype(float)
        nd = band.GetNoDataValue()
        if nd is not None:
            values[values == nd] = np.nan
        if band.GetMaskFlags() & gdal.GMF_ALL_VALID == 0:
            values[band.GetMaskBand().ReadAsArray() == 0] = np.nan
        extra = parameters.get(self.EXTRA_NODATA)
        if extra is not None and extra != "":
            values[values == self.parameterAsDouble(parameters, self.EXTRA_NODATA,
                                                    context)] = np.nan
        return layer, values, gt

    def processAlgorithm(self, parameters, context, feedback):
        layer, values, gt = self._read_band(parameters, context, feedback)
        rows, cols = values.shape
        dx, dy = abs(gt[1]), abs(gt[5])
        observed = np.isfinite(values)
        gaps = ~observed
        feedback.pushInfo(self.tr("{:,} x {:,} raster: {:,} valid cells, {:,} gaps.")
                          .format(cols, rows, int(observed.sum()), int(gaps.sum())))
        if observed.sum() < 3:
            raise QgsProcessingException(self.tr("Need at least 3 valid cells."))

        if parameters.get(self.FILL_MASK):
            src = self.parameterAsSource(parameters, self.FILL_MASK, context)
            if src is not None:
                gaps &= rasterize_polygons(src, layer.crs(), context, cols, rows, gt)
                feedback.pushInfo(self.tr("{:,} gap cells lie inside the fill polygons.")
                                  .format(int(gaps.sum())))

        model = vg.MODELS[self.parameterAsEnum(parameters, "MODEL", context)]
        s = read_variogram_settings(self, parameters, context)
        vgm = gapfill.fit_raster_variogram(values, observed, dx, dy, model,
                                           n_lags=s["n_lags"], max_lag=s["max_lag"],
                                           nugget=s["nugget"], psill=s["psill"],
                                           rng=s["rng"])
        feedback.pushInfo(self.tr("Variogram: ") + vgm.describe())
        feedback.setProgress(5)

        simulate = self.parameterAsEnum(parameters, self.METHOD, context) == 1
        n_sims = self.parameterAsInt(parameters, self.N_SIMS, context) if simulate else 0
        unc_path = self.parameterAsOutputLayer(parameters, self.UNCERTAINTY, context)
        max_dist = self.parameterAsDouble(parameters, self.MAX_DISTANCE, context)
        try:
            res = gapfill.fill_gaps(
                values, dx, dy, vgm, gaps=gaps,
                k=self.parameterAsInt(parameters, self.NEIGHBORS, context),
                max_distance=max_dist if max_dist > 0 else None,
                min_points=self.parameterAsInt(parameters, self.MIN_POINTS, context),
                n_sims=n_sims, seed=self.parameterAsInt(parameters, self.SEED, context),
                want_variance=bool(unc_path),
                progress=self.progress_callback(feedback, 5, 90),
                log=feedback.pushInfo)
        except Cancelled:
            raise QgsProcessingException(self.tr("Cancelled."))
        except gapfill.GapFillError as e:
            raise QgsProcessingException(str(e))
        except MemoryError:
            raise QgsProcessingException(self.tr(
                "Out of memory. Use fewer realisations, a smaller area or kriging."))

        feedback.pushInfo(self.tr("Filled {:,} cells; {:,} gap cells left empty.")
                          .format(res["filled_cells"], res["unfilled_cells"]))

        out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        if simulate:
            sims = res["simulations"]
            write_array(out_path, list(sims), gt, layer.crs(),
                        ["realisation {}".format(i + 1) for i in range(len(sims))])
        else:
            write_array(out_path, [res["kriged"]], gt, layer.crs(), ["kriged"])
        result = {self.OUTPUT: out_path}

        if unc_path:
            if simulate and n_sims >= 2:
                sd = res["simulations"].astype(float).std(axis=0, ddof=1)
                sd[observed] = 0.0
                label = "simulation standard deviation"
            else:
                sd = np.sqrt(res["variance"])
                label = "kriging standard deviation"
            write_array(unc_path, [sd], gt, layer.crs(), [label])
            feedback.pushInfo(self.tr("Uncertainty raster: {}.").format(label))
            result[self.UNCERTAINTY] = unc_path
        feedback.setProgress(100)
        return result
