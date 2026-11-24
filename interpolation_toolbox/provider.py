import os

from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingProvider

from .algorithms.cross_validation import CrossValidationAlgorithm
from .algorithms.gap_fill import GapFillAlgorithm
from .algorithms.idw import IdwAlgorithm
from .algorithms.kriging import OrdinaryKrigingAlgorithm
from .algorithms.moving_window import MovingWindowAlgorithm
from .algorithms.nearest import NearestNeighbourAlgorithm
from .algorithms.rbf import RbfAlgorithm
from .algorithms.tin import TinAlgorithm
from .algorithms.trend import TrendSurfaceAlgorithm
from .algorithms.variogram import VariogramAlgorithm

ALGORITHMS = [
    IdwAlgorithm,
    NearestNeighbourAlgorithm,
    MovingWindowAlgorithm,
    TinAlgorithm,
    RbfAlgorithm,
    TrendSurfaceAlgorithm,
    OrdinaryKrigingAlgorithm,
    VariogramAlgorithm,
    GapFillAlgorithm,
    CrossValidationAlgorithm,
]


class PointInterpolationProvider(QgsProcessingProvider):
    def id(self):
        return "pointinterpolation"

    def name(self):
        return "Point Interpolation"

    def longName(self):
        return "Point Interpolation Toolbox"

    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), "icon.svg"))

    def loadAlgorithms(self):  # noqa: N802
        for alg in ALGORITHMS:
            self.addAlgorithm(alg())
