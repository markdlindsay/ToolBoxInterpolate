from ..core import methods
from .base import GridInterpolationAlgorithm, advanced, number


class IdwAlgorithm(GridInterpolationAlgorithm):
    POWER = "POWER"
    NEIGHBORS = "NEIGHBORS"
    RADIUS = "RADIUS"
    MIN_POINTS = "MIN_POINTS"
    SMOOTHING = "SMOOTHING"

    _group, _group_id = "Deterministic", "deterministic"

    def name(self):
        return "idw"

    def displayName(self):
        return self.tr("Inverse distance weighting (IDW)")

    def shortHelpString(self):
        return self.tr(
            "Each cell is a weighted average of nearby samples, with weight "
            "1 / distance<sup>power</sup>.<br><br>"
            "<b>Power</b>: higher values give more local, 'bullseye' surfaces; 2 is typical.<br>"
            "<b>Neighbours</b>: number of closest samples used (0 = all samples).<br>"
            "<b>Search radius</b>: ignore samples further than this (0 = unlimited). "
            "Cells with fewer than <i>Minimum points</i> samples in range become no-data.<br>"
            "<b>Smoothing</b>: added to distances so the surface no longer passes "
            "exactly through the samples.<br><br>"
            "Fast and predictable, but cannot predict values beyond the data range "
            "and tends to produce plateaus around samples.")

    def add_method_parameters(self):
        self.addParameter(number(self.POWER, self.tr("Power"), default=2.0, minimum=0.0))
        self.addParameter(number(self.NEIGHBORS, self.tr("Neighbours (0 = all points)"),
                                 "int", 12, 0))
        self.addParameter(number(self.RADIUS, self.tr("Search radius (0 = unlimited)"),
                                 default=0.0, minimum=0.0))
        self.addParameter(advanced(number(self.MIN_POINTS, self.tr("Minimum points"),
                                          "int", 1, 1)))
        self.addParameter(advanced(number(self.SMOOTHING, self.tr("Smoothing distance"),
                                          default=0.0, minimum=0.0)))

    def interpolate(self, pts, grid, parameters, context, feedback):
        p = lambda n: self.parameterAsDouble(parameters, n, context)  # noqa: E731
        out = methods.idw(
            pts.xy, pts.z, grid.cell_centres(),
            power=p(self.POWER),
            k=self.parameterAsInt(parameters, self.NEIGHBORS, context),
            radius=p(self.RADIUS),
            min_points=self.parameterAsInt(parameters, self.MIN_POINTS, context),
            smoothing=p(self.SMOOTHING),
            progress=self.progress_callback(feedback))
        return {self.OUTPUT: out}
