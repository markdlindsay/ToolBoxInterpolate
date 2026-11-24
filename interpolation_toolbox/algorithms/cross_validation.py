import numpy as np

from qgis.core import (QgsCoordinateReferenceSystem, QgsFeature, QgsFeatureSink,
                       QgsGeometry, QgsPointXY, QgsProcessing, QgsProcessingException,
                       QgsProcessingParameterEnum, QgsProcessingParameterFeatureSink,
                       QgsWkbTypes)

from ..core import methods, variogram as vg
from ..core.neighbors import Cancelled
from .base import PointAlgorithm, add_row, number, table_fields
from .kriging import MODEL_LABELS
from .rbf import AUTO_LOCAL_K, GLOBAL_LIMIT


class CrossValidationAlgorithm(PointAlgorithm):
    METHODS = "METHODS"
    FOLDS = "FOLDS"
    POWER = "POWER"
    NEIGHBORS = "NEIGHBORS"
    MODEL = "MODEL"
    SEED = "SEED"
    OUTPUT = "OUTPUT"
    POINTS = "POINTS"

    _group, _group_id = "Validation", "validation"

    # (key, label) - key is used for field names in the point output
    CANDIDATES = [
        ("idw", "Inverse distance weighting"),
        ("nearest", "Nearest neighbour"),
        ("moving_mean", "Moving average (radius = 3 x mean spacing)"),
        ("tps", "Thin plate spline"),
        ("multiquadric", "Multiquadric RBF"),
        ("kriging", "Ordinary kriging"),
        ("kriging_trend", "Kriging with linear trend"),
        ("trend1", "Trend surface, order 1"),
        ("trend2", "Trend surface, order 2"),
    ]

    def name(self):
        return "crossvalidation"

    def displayName(self):
        return self.tr("Compare methods (cross-validation)")

    def shortHelpString(self):
        return self.tr(
            "Scores several interpolation methods on your data with k-fold "
            "cross-validation: the points are split into <i>folds</i> groups; each group "
            "is predicted from the others and the prediction errors are summarised.<br><br>"
            "<b>Results table</b>: per method - mean error (bias), mean absolute error, "
            "RMSE, R² (1 - SSE/SST) and Pearson r, ranked by RMSE (rank 1 = best).<br>"
            "<b>Point predictions</b> (optional): each sample with its observed value and "
            "the out-of-fold prediction and error of every method, to map where each "
            "method struggles.<br><br>"
            "Folds = number of points gives leave-one-out validation. IDW and kriging use "
            "the neighbour count and power set here; the kriging variogram is fitted once "
            "on all data.")

    def initAlgorithm(self, config=None):
        self.add_point_parameters()
        self.addParameter(QgsProcessingParameterEnum(
            self.METHODS, self.tr("Methods to compare"),
            [self.tr(label) for _, label in self.CANDIDATES], allowMultiple=True,
            defaultValue=list(range(len(self.CANDIDATES)))))
        self.addParameter(number(self.FOLDS, self.tr("Number of folds"), "int", 10, 2))
        self.addParameter(number(self.NEIGHBORS, self.tr("Neighbours for IDW and kriging"),
                                 "int", 12, 1))
        self.addParameter(number(self.POWER, self.tr("IDW power"), default=2.0, minimum=0.0))
        self.addParameter(QgsProcessingParameterEnum(
            self.MODEL, self.tr("Kriging variogram model"),
            [self.tr(s) for s in MODEL_LABELS], defaultValue=0))
        self.addParameter(number(self.SEED, self.tr("Random seed for fold assignment"),
                                 "int", 0, 0))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, self.tr("Cross-validation results"), QgsProcessing.TypeVector))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.POINTS, self.tr("Point predictions"), QgsProcessing.TypeVectorPoint,
            optional=True, createByDefault=False))

    def _predictors(self, pts, parameters, context, feedback):
        k = self.parameterAsInt(parameters, self.NEIGHBORS, context)
        power = self.parameterAsDouble(parameters, self.POWER, context)
        model = vg.MODELS[self.parameterAsEnum(parameters, self.MODEL, context)]
        radius = 3.0 * methods.mean_spacing(pts.xy)
        rbf_k = 0 if pts.n <= GLOBAL_LIMIT else AUTO_LOCAL_K
        cache = {}

        def variogram_for(xy, z, key):
            if key not in cache:
                lags, g, c, ml = vg.experimental(xy, z)
                cache[key] = vg.fit(lags, g, c, model, max_lag=ml)
                feedback.pushInfo(self.tr("{} variogram: {}").format(key, cache[key].describe()))
            return cache[key]

        def kriging(a, b, q):
            return methods.ordinary_kriging(a, b, q, variogram_for(pts.xy, pts.z, "Kriging"), k=k)

        trend_all = methods.TrendSurface(pts.xy, pts.z, 1)
        resid_all = pts.z - trend_all(pts.xy)

        def kriging_trend(a, b, q):
            t = methods.TrendSurface(a, b, 1)
            v = variogram_for(pts.xy, resid_all, "Residual")
            return methods.ordinary_kriging(a, b - t(a), q, v, k=k) + t(q)

        return {
            "idw": lambda a, b, q: methods.idw(a, b, q, power=power, k=k),
            "nearest": lambda a, b, q: methods.nearest(a, b, q),
            "moving_mean": lambda a, b, q: methods.moving_window(a, b, q, radius, "mean"),
            "tps": lambda a, b, q: methods.rbf(a, b, q, "thin_plate", k=rbf_k),
            "multiquadric": lambda a, b, q: methods.rbf(a, b, q, "multiquadric", k=rbf_k),
            "kriging": kriging,
            "kriging_trend": kriging_trend,
            "trend1": lambda a, b, q: methods.trend_surface(a, b, q, 1),
            "trend2": lambda a, b, q: methods.trend_surface(a, b, q, 2),
        }

    def processAlgorithm(self, parameters, context, feedback):
        pts = self.read_points(parameters, context, feedback, 5)
        chosen = self.parameterAsEnums(parameters, self.METHODS, context)
        if not chosen:
            raise QgsProcessingException(self.tr("Select at least one method."))
        folds = self.parameterAsInt(parameters, self.FOLDS, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        predictors = self._predictors(pts, parameters, context, feedback)

        results = []
        for n, idx in enumerate(chosen):
            key, label = self.CANDIDATES[idx]
            feedback.pushInfo(self.tr("Validating: {}").format(label))
            cb = self.progress_callback(feedback, 5 + 90 * n / len(chosen),
                                        5 + 90 * (n + 1) / len(chosen))
            try:
                pred = methods.cross_validate(pts.xy, pts.z, predictors[key], folds, seed, cb)
            except Cancelled:
                raise QgsProcessingException(self.tr("Cancelled."))
            except Exception as e:  # keep going: one failing method shouldn't sink the rest
                feedback.reportError(self.tr("{} failed: {}").format(label, e))
                pred = np.full(pts.n, np.nan)
            results.append((key, label, pred, methods.error_metrics(pts.z, pred)))

        order = sorted(range(len(results)),
                       key=lambda i: (not np.isfinite(results[i][3]["rmse"]),
                                      results[i][3]["rmse"]))
        rank = {i: r + 1 for r, i in enumerate(order)}

        fields = table_fields([("rank", "int"), ("method", "string"), ("n", "int"),
                               ("mean_error", "double"), ("mae", "double"),
                               ("rmse", "double"), ("r2", "double"), ("pearson_r", "double")])
        sink, dest = self.parameterAsSink(parameters, self.OUTPUT, context, fields,
                                          QgsWkbTypes.NoGeometry,
                                          QgsCoordinateReferenceSystem())
        feedback.pushInfo("")
        feedback.pushInfo("{:<4} {:<45} {:>12} {:>12} {:>8}".format(
            "rank", "method", "MAE", "RMSE", "R²"))
        for i in order:
            key, label, _, m = results[i]
            add_row(sink, fields, [rank[i], label, m["n"], m["me"], m["mae"], m["rmse"],
                                   m["r2"], m["r"]])
            feedback.pushInfo("{:<4} {:<45} {:>12.5g} {:>12.5g} {:>8.4f}".format(
                rank[i], label, m["mae"], m["rmse"], m["r2"]))

        out = {self.OUTPUT: dest}
        if parameters.get(self.POINTS):
            out[self.POINTS] = self._write_points(pts, results, folds, seed,
                                                  parameters, context)
        feedback.setProgress(100)
        return out

    def _write_points(self, pts, results, folds, seed, parameters, context):
        spec = [("observed", "double")]
        for key, *_ in results:
            spec += [("pred_" + key, "double"), ("err_" + key, "double")]
        fields = table_fields(spec)
        sink, dest = self.parameterAsSink(parameters, self.POINTS, context, fields,
                                          QgsWkbTypes.Point, pts.crs)
        if sink is None:
            return None
        for j in range(pts.n):
            vals = [float(pts.z[j])]
            for _, _, pred, _ in results:
                p = float(pred[j])
                ok = np.isfinite(p)
                vals += [p if ok else None, (p - pts.z[j]) if ok else None]
            f = QgsFeature(fields)
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*map(float, pts.xy[j]))))
            f.setAttributes(vals)
            sink.addFeature(f, QgsFeatureSink.FastInsert)
        return dest
