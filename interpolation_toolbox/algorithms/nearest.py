from ..core import methods
from .base import GridInterpolationAlgorithm, number


class NearestNeighbourAlgorithm(GridInterpolationAlgorithm):
    RADIUS = "RADIUS"

    _group, _group_id = "Deterministic", "deterministic"

    def name(self):
        return "nearestneighbour"

    def displayName(self):
        return self.tr("Nearest neighbour (Thiessen / Voronoi)")

    def shortHelpString(self):
        return self.tr(
            "Each cell takes the value of the closest sample, producing a "
            "Thiessen/Voronoi-style surface. Best for categorical or coded values "
            "where averaging makes no sense.<br><br>"
            "<b>Search radius</b>: cells further than this from any sample become "
            "no-data (0 = unlimited).")

    def add_method_parameters(self):
        self.addParameter(number(self.RADIUS, self.tr("Search radius (0 = unlimited)"),
                                 default=0.0, minimum=0.0))

    def interpolate(self, pts, grid, parameters, context, feedback):
        out = methods.nearest(
            pts.xy, pts.z, grid.cell_centres(),
            radius=self.parameterAsDouble(parameters, self.RADIUS, context),
            progress=self.progress_callback(feedback))
        return {self.OUTPUT: out}
