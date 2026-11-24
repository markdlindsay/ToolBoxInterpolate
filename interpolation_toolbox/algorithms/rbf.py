from qgis.core import QgsProcessingParameterEnum

from ..core import methods
from .base import GridInterpolationAlgorithm, advanced, number

LABELS = ["Thin plate spline", "Multiquadric", "Inverse multiquadric", "Gaussian",
          "Linear", "Cubic"]
GLOBAL_LIMIT = 3000
AUTO_LOCAL_K = 50


class RbfAlgorithm(GridInterpolationAlgorithm):
    KERNEL = "KERNEL"
    EPSILON = "EPSILON"
    SMOOTHING = "SMOOTHING"
    NEIGHBORS = "NEIGHBORS"

    _group, _group_id = "Splines and functions", "splines"
    min_points = 3

    def name(self):
        return "rbf"

    def displayName(self):
        return self.tr("Radial basis functions / splines")

    def shortHelpString(self):
        return self.tr(
            "Fits a smooth surface made of radial basis functions plus a linear trend. "
            "The <b>thin plate spline</b> kernel gives the classic minimum-curvature "
            "spline; multiquadric and Gaussian kernels are tuned by the shape parameter."
            "<br><br>"
            "<b>Shape parameter</b>: width of the multiquadric / inverse multiquadric / "
            "Gaussian kernels, in map units (0 = mean sample spacing). Ignored by the "
            "other kernels.<br>"
            "<b>Smoothing</b>: 0 passes exactly through every sample; larger values give a "
            "smoother, approximating surface (useful for noisy data).<br>"
            "<b>Neighbours</b>: 0 = automatic (one global fit up to {} points, otherwise "
            "local fits from the {} nearest samples). Local fits scale to big datasets "
            "but may show faint seams.<br><br>"
            "Unlike IDW, splines can overshoot beyond the data range near steep gradients."
        ).format(GLOBAL_LIMIT, AUTO_LOCAL_K)

    def add_method_parameters(self):
        self.addParameter(QgsProcessingParameterEnum(
            self.KERNEL, self.tr("Kernel"), [self.tr(s) for s in LABELS], defaultValue=0))
        self.addParameter(number(self.SMOOTHING, self.tr("Smoothing (0 = exact)"),
                                 default=0.0, minimum=0.0))
        self.addParameter(number(self.NEIGHBORS, self.tr("Neighbours (0 = automatic)"),
                                 "int", 0, 0))
        self.addParameter(advanced(number(
            self.EPSILON, self.tr("Shape parameter in map units (0 = automatic)"),
            default=0.0, minimum=0.0)))

    def interpolate(self, pts, grid, parameters, context, feedback):
        k = self.parameterAsInt(parameters, self.NEIGHBORS, context)
        if k <= 0:
            k = 0 if pts.n <= GLOBAL_LIMIT else AUTO_LOCAL_K
        if k == 0 or k >= pts.n:
            feedback.pushInfo(self.tr("Solving one global system for {} points.").format(pts.n))
        else:
            feedback.pushInfo(self.tr("Local fits using the {} nearest points.").format(k))
        kernel = methods.RBF_KERNELS[self.parameterAsEnum(parameters, self.KERNEL, context)]
        out = methods.rbf(
            pts.xy, pts.z, grid.cell_centres(), kernel=kernel,
            epsilon=self.parameterAsDouble(parameters, self.EPSILON, context),
            smoothing=self.parameterAsDouble(parameters, self.SMOOTHING, context),
            k=k, progress=self.progress_callback(feedback))
        return {self.OUTPUT: out}
