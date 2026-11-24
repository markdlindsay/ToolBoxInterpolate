from qgis.core import QgsProcessingParameterEnum

from ..core import methods
from .base import GridInterpolationAlgorithm, advanced, number

LABELS = ["Mean", "Median", "Minimum", "Maximum", "Range", "Standard deviation", "Count"]


class MovingWindowAlgorithm(GridInterpolationAlgorithm):
    STATISTIC = "STATISTIC"
    RADIUS = "RADIUS"
    MAX_POINTS = "MAX_POINTS"
    MIN_POINTS = "MIN_POINTS"

    _group, _group_id = "Deterministic", "deterministic"

    def name(self):
        return "movingwindow"

    def displayName(self):
        return self.tr("Moving window statistics (moving average)")

    def shortHelpString(self):
        return self.tr(
            "Computes a statistic of all samples within a circular search radius "
            "around each cell: mean (moving average), median, minimum, maximum, "
            "range, standard deviation or count (point density).<br><br>"
            "<b>Search radius</b>: 0 picks 3 x the mean spacing between samples.<br>"
            "<b>Maximum points</b>: only use the N closest samples in the window (0 = all).<br>"
            "Cells with fewer than <i>Minimum points</i> samples become no-data.")

    def add_method_parameters(self):
        self.addParameter(QgsProcessingParameterEnum(
            self.STATISTIC, self.tr("Statistic"), [self.tr(s) for s in LABELS], defaultValue=0))
        self.addParameter(number(self.RADIUS, self.tr("Search radius (0 = automatic)"),
                                 default=0.0, minimum=0.0))
        self.addParameter(advanced(number(self.MAX_POINTS, self.tr("Maximum points (0 = all)"),
                                          "int", 0, 0)))
        self.addParameter(advanced(number(self.MIN_POINTS, self.tr("Minimum points"),
                                          "int", 1, 1)))

    def interpolate(self, pts, grid, parameters, context, feedback):
        radius = self.parameterAsDouble(parameters, self.RADIUS, context)
        if radius <= 0:
            radius = 3.0 * methods.mean_spacing(pts.xy)
            feedback.pushInfo(self.tr("Automatic search radius: {:.6g}").format(radius))
        stat = methods.STATISTICS[self.parameterAsEnum(parameters, self.STATISTIC, context)]
        out = methods.moving_window(
            pts.xy, pts.z, grid.cell_centres(), radius, stat,
            max_points=self.parameterAsInt(parameters, self.MAX_POINTS, context),
            min_points=self.parameterAsInt(parameters, self.MIN_POINTS, context),
            progress=self.progress_callback(feedback))
        return {self.OUTPUT: out}
