import numpy as np

from qgis.core import (QgsCoordinateReferenceSystem, QgsProcessing,
                       QgsProcessingParameterFeatureSink, QgsWkbTypes)

from ..core import variogram as vg
from .base import PointAlgorithm, add_row, table_fields
from .kriging import add_variogram_parameters, read_variogram_settings, fit_variogram


class VariogramAlgorithm(PointAlgorithm):
    OUTPUT = "OUTPUT"
    MODELS_OUTPUT = "MODELS"

    _group, _group_id = "Geostatistics", "geostatistics"

    def name(self):
        return "variogram"

    def displayName(self):
        return self.tr("Variogram analysis")

    def shortHelpString(self):
        return self.tr(
            "Computes the experimental semivariogram of the samples and fits every "
            "variogram model to it, to help choose kriging settings.<br><br>"
            "<b>Variogram table</b>: one row per lag bin with the mean distance, "
            "semivariance, pair count and each fitted model's value - plot lag against "
            "the other columns (e.g. with the Data Plotly plugin).<br>"
            "<b>Model summary</b>: nugget, partial sill, range and weighted fit error of "
            "each model; the lowest error is usually the best starting point.<br><br>"
            "Nugget / sill / range set here are held fixed for every model. Datasets "
            "larger than 2500 points are randomly subsampled for speed.")

    def initAlgorithm(self, config=None):
        self.add_point_parameters()
        add_variogram_parameters(self, with_model=False)
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, self.tr("Variogram table"), QgsProcessing.TypeVector))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.MODELS_OUTPUT, self.tr("Model summary"), QgsProcessing.TypeVector))

    def processAlgorithm(self, parameters, context, feedback):
        pts = self.read_points(parameters, context, feedback, 3)
        settings = read_variogram_settings(self, parameters, context)

        fits = {}
        empirical = None
        for i, model in enumerate(vg.MODELS):
            fits[model], empirical = fit_variogram(pts.xy, pts.z, model, settings)
            feedback.setProgress(10 + 80 * (i + 1) / len(vg.MODELS))
        lags, gamma, counts = empirical

        fields = table_fields([("lag", "double"), ("semivariance", "double"),
                               ("pairs", "int")] + [(m, "double") for m in vg.MODELS])
        sink, dest = self.parameterAsSink(parameters, self.OUTPUT, context, fields,
                                          QgsWkbTypes.NoGeometry,
                                          QgsCoordinateReferenceSystem())
        model_vals = {m: f(lags) for m, f in fits.items()}
        for j in range(len(lags)):
            add_row(sink, fields, [float(lags[j]), float(gamma[j]), int(counts[j])]
                    + [float(model_vals[m][j]) for m in vg.MODELS])

        sfields = table_fields([("model", "string"), ("nugget", "double"),
                                ("partial_sill", "double"), ("sill", "double"),
                                ("range", "double"), ("wsse", "double"),
                                ("best", "int")])
        ssink, sdest = self.parameterAsSink(parameters, self.MODELS_OUTPUT, context, sfields,
                                            QgsWkbTypes.NoGeometry,
                                            QgsCoordinateReferenceSystem())
        errors = {m: float(np.sum(counts * (model_vals[m] - gamma) ** 2)) for m in vg.MODELS}
        best = min(errors, key=errors.get)
        for m in vg.MODELS:
            f = fits[m]
            add_row(ssink, sfields, [m, f.nugget, f.psill, f.sill, f.range, errors[m],
                                     int(m == best)])
            feedback.pushInfo("{}  (weighted SSE {:.6g}){}".format(
                f.describe(), errors[m], "  <- best fit" if m == best else ""))
        feedback.setProgress(100)
        return {self.OUTPUT: dest, self.MODELS_OUTPUT: sdest}
