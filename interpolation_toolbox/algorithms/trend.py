from qgis.core import (QgsFeature, QgsFeatureSink, QgsGeometry, QgsPointXY,
                       QgsProcessing, QgsProcessingParameterFeatureSink, QgsWkbTypes)

from ..core import methods
from .base import GridInterpolationAlgorithm, number, table_fields


class TrendSurfaceAlgorithm(GridInterpolationAlgorithm):
    ORDER = "ORDER"
    RESIDUALS = "RESIDUALS"

    _group, _group_id = "Splines and functions", "splines"

    def name(self):
        return "trendsurface"

    def displayName(self):
        return self.tr("Polynomial trend surface")

    def shortHelpString(self):
        return self.tr(
            "Fits a single least-squares polynomial z = f(x, y) of the chosen order "
            "(1 = plane, 2 = quadratic, 3 = cubic ...). It does not pass through the "
            "samples: it shows the broad regional trend.<br><br>"
            "R<sup>2</sup> and RMSE of the fit are written to the log. The optional "
            "<b>Residuals</b> output holds each sample's observed value, fitted value and "
            "residual, handy for spotting local anomalies or for kriging the residuals.")

    def add_method_parameters(self):
        self.addParameter(number(self.ORDER, self.tr("Polynomial order"), "int", 1, 1, 6))

    def add_extra_outputs(self):
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.RESIDUALS, self.tr("Residuals"), QgsProcessing.TypeVectorPoint,
            optional=True, createByDefault=False))

    def interpolate(self, pts, grid, parameters, context, feedback):
        order = self.parameterAsInt(parameters, self.ORDER, context)
        model = methods.TrendSurface(pts.xy, pts.z, order)
        if model.n_terms > pts.n:
            feedback.reportError(self.tr(
                "Order {} needs {} coefficients but there are only {} points; the fit is "
                "under-determined.").format(order, model.n_terms, pts.n))
        feedback.pushInfo(self.tr("Trend surface order {}: R² = {:.4f}, RMSE = {:.6g}")
                          .format(order, model.r2, model.rmse))
        self._write_residuals(model, pts, parameters, context)
        out = methods.trend_surface(pts.xy, pts.z, grid.cell_centres(), order,
                                    progress=self.progress_callback(feedback))
        return {self.OUTPUT: out}

    def processAlgorithm(self, parameters, context, feedback):
        self._residual_id = None
        result = super().processAlgorithm(parameters, context, feedback)
        if self._residual_id:
            result[self.RESIDUALS] = self._residual_id
        return result

    def _write_residuals(self, model, pts, parameters, context):
        if not parameters.get(self.RESIDUALS):
            return
        fields = table_fields([("observed", "double"), ("fitted", "double"),
                               ("residual", "double")])
        sink, dest = self.parameterAsSink(parameters, self.RESIDUALS, context, fields,
                                          QgsWkbTypes.Point, pts.crs)
        if sink is None:
            return
        fitted = model(pts.xy)
        for (x, y), o, f in zip(pts.xy, pts.z, fitted):
            feat = QgsFeature(fields)
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y))))
            feat.setAttributes([float(o), float(f), float(o - f)])
            sink.addFeature(feat, QgsFeatureSink.FastInsert)
        self._residual_id = dest
