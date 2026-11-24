from qgis.core import QgsApplication

from .provider import PointInterpolationProvider


class PointInterpolationPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):  # noqa: N802
        self.provider = PointInterpolationProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):  # noqa: N802
        self.initProcessing()

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
