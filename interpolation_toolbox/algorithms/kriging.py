import numpy as np

from qgis.core import (QgsProcessing, QgsProcessingParameterEnum,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterRasterDestination, QgsWkbTypes,
                       QgsCoordinateReferenceSystem)

from ..core import methods, variogram as vg
from .base import GridInterpolationAlgorithm, add_row, advanced, number, table_fields

MODEL_LABELS = ["Spherical", "Exponential", "Gaussian", "Linear (with sill)"]
DRIFT_LABELS = ["None (ordinary kriging)", "Linear trend", "Quadratic trend"]


def add_variogram_parameters(alg, with_model=True):
    if with_model:
        alg.addParameter(QgsProcessingParameterEnum(
            "MODEL", alg.tr("Variogram model"), [alg.tr(s) for s in MODEL_LABELS],
            defaultValue=0))
    alg.addParameter(advanced(number("NUGGET", alg.tr("Nugget (-1 = fit automatically)"),
                                     default=-1.0, minimum=-1.0)))
    alg.addParameter(advanced(number("PSILL", alg.tr("Partial sill (0 = fit automatically)"),
                                     default=0.0, minimum=0.0)))
    alg.addParameter(advanced(number("RANGE", alg.tr("Range (0 = fit automatically)"),
                                     default=0.0, minimum=0.0)))
    alg.addParameter(advanced(number("N_LAGS", alg.tr("Number of lag bins"), "int", 15, 3, 200)))
    alg.addParameter(advanced(number("MAX_LAG", alg.tr("Maximum lag distance (0 = half the "
                                                       "largest point separation)"),
                                     default=0.0, minimum=0.0)))


def read_variogram_settings(alg, parameters, context):
    nug = alg.parameterAsDouble(parameters, "NUGGET", context)
    ps = alg.parameterAsDouble(parameters, "PSILL", context)
    rg = alg.parameterAsDouble(parameters, "RANGE", context)
    return dict(
        nugget=nug if nug >= 0 else None,
        psill=ps if ps > 0 else None,
        rng=rg if rg > 0 else None,
        n_lags=alg.parameterAsInt(parameters, "N_LAGS", context),
        max_lag=alg.parameterAsDouble(parameters, "MAX_LAG", context) or None,
    )


def fit_variogram(xy, z, model, settings):
    lags, gamma, counts, max_lag = vg.experimental(
        xy, z, n_lags=settings["n_lags"], max_lag=settings["max_lag"])
    fitted = vg.fit(lags, gamma, counts, model, settings["nugget"], settings["psill"],
                    settings["rng"], max_lag=max_lag)
    return fitted, (lags, gamma, counts)


class OrdinaryKrigingAlgorithm(GridInterpolationAlgorithm):
    DRIFT = "DRIFT"
    NEIGHBORS = "NEIGHBORS"
    RADIUS = "RADIUS"
    MIN_POINTS = "MIN_POINTS"
    VARIANCE = "VARIANCE"
    VARIOGRAM = "VARIOGRAM"

    _group, _group_id = "Geostatistics", "geostatistics"
    min_points = 3

    def name(self):
        return "kriging"

    def displayName(self):
        return self.tr("Kriging (ordinary / with trend)")

    def shortHelpString(self):
        return self.tr(
            "Geostatistical interpolation. A variogram model describing how values "
            "decorrelate with distance is fitted to the samples, and each cell is the "
            "best linear unbiased estimate from its neighbours.<br><br>"
            "<b>Variogram model</b>: spherical suits most data; Gaussian suits very smooth "
            "phenomena; exponential suits rougher ones. Nugget, partial sill and range are "
            "fitted automatically unless you set them (advanced). Run <i>Variogram "
            "analysis</i> first to compare models.<br>"
            "<b>Trend</b>: removes a linear or quadratic trend before kriging and adds it "
            "back afterwards (regression / universal-style kriging) - use when the data "
            "has a clear regional gradient.<br>"
            "<b>Neighbours / search radius</b>: local kriging neighbourhood. 12-32 "
            "neighbours is typical.<br><br>"
            "The optional <b>kriging variance</b> raster maps prediction uncertainty: it "
            "is low near samples and grows with distance from them.")

    def add_method_parameters(self):
        add_variogram_parameters(self)
        self.addParameter(QgsProcessingParameterEnum(
            self.DRIFT, self.tr("Trend (drift)"), [self.tr(s) for s in DRIFT_LABELS],
            defaultValue=0))
        self.addParameter(number(self.NEIGHBORS, self.tr("Neighbours (0 = all points)"),
                                 "int", 16, 0))
        self.addParameter(number(self.RADIUS, self.tr("Search radius (0 = unlimited)"),
                                 default=0.0, minimum=0.0))
        self.addParameter(advanced(number(self.MIN_POINTS, self.tr("Minimum points"),
                                          "int", 3, 1)))

    def add_extra_outputs(self):
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.VARIANCE, self.tr("Kriging variance"), optional=True,
            createByDefault=False))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.VARIOGRAM, self.tr("Experimental and fitted variogram"),
            QgsProcessing.TypeVector, optional=True, createByDefault=False))

    def processAlgorithm(self, parameters, context, feedback):
        self._variogram_id = None
        result = super().processAlgorithm(parameters, context, feedback)
        if self._variogram_id:
            result[self.VARIOGRAM] = self._variogram_id
        return result

    def interpolate(self, pts, grid, parameters, context, feedback):
        model = vg.MODELS[self.parameterAsEnum(parameters, "MODEL", context)]
        drift = self.parameterAsEnum(parameters, self.DRIFT, context)
        xy, z = pts.xy, pts.z
        trend = None
        if drift > 0:
            trend = methods.TrendSurface(xy, z, drift)
            feedback.pushInfo(self.tr("Removed order-{} trend (R² = {:.4f}); kriging the "
                                      "residuals.").format(drift, trend.r2))
            z = z - trend(xy)

        model_fit, empirical = fit_variogram(xy, z, model,
                                             read_variogram_settings(self, parameters, context))
        feedback.pushInfo(self.tr("Variogram: ") + model_fit.describe())
        self._write_variogram_table(model_fit, empirical, parameters, context)

        q = grid.cell_centres()
        want_var = bool(self.parameterAsOutputLayer(parameters, self.VARIANCE, context))
        res = methods.ordinary_kriging(
            xy, z, q, model_fit,
            k=self.parameterAsInt(parameters, self.NEIGHBORS, context),
            radius=self.parameterAsDouble(parameters, self.RADIUS, context),
            min_points=self.parameterAsInt(parameters, self.MIN_POINTS, context),
            return_variance=want_var,
            progress=self.progress_callback(feedback))
        pred, var = res if want_var else (res, None)
        if trend is not None:
            pred = pred + trend(q)
        out = {self.OUTPUT: pred}
        if var is not None:
            out[self.VARIANCE] = var
        return out

    def _write_variogram_table(self, model_fit, empirical, parameters, context):
        if not parameters.get(self.VARIOGRAM):
            return
        fields = table_fields([("lag", "double"), ("semivariance", "double"),
                               ("pairs", "int"), ("model", "double")])
        sink, dest = self.parameterAsSink(parameters, self.VARIOGRAM, context, fields,
                                          QgsWkbTypes.NoGeometry,
                                          QgsCoordinateReferenceSystem())
        if sink is None:
            return
        lags, gamma, counts = empirical
        for h, g, c, m in zip(lags, gamma, counts, model_fit(lags)):
            add_row(sink, fields, [float(h), float(g), int(c), float(m)])
        self._variogram_id = dest
