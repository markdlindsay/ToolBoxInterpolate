import numpy as np
from osgeo import gdal

from qgis.analysis import QgsGridFileWriter, QgsInterpolator, QgsTinInterpolator
from qgis.core import (QgsProcessingException, QgsProcessingParameterEnum,
                       QgsProcessingUtils, QgsRectangle)

from .base import GridInterpolationAlgorithm


def _enum(cls, enum_name, member, legacy=None):
    """Enum lookup that works with both scoped (3.3x+) and legacy PyQGIS names."""
    scoped = getattr(cls, enum_name, None)
    if scoped is not None and hasattr(scoped, member):
        return getattr(scoped, member)
    return getattr(cls, legacy or member)


class TinAlgorithm(GridInterpolationAlgorithm):
    METHOD = "METHOD"

    _group, _group_id = "Deterministic", "deterministic"
    min_points = 3

    def name(self):
        return "tin"

    def displayName(self):
        return self.tr("Triangulated irregular network (TIN)")

    def shortHelpString(self):
        return self.tr(
            "Builds a Delaunay triangulation of the samples and interpolates inside "
            "each triangle using QGIS's own TIN engine.<br><br>"
            "<b>Linear</b>: flat triangle facets; exact and robust.<br>"
            "<b>Clough-Tocher</b>: smooth cubic surface over the same triangles.<br><br>"
            "Cells outside the convex hull of the points are no-data. Unlike the other "
            "tools, the TIN uses every input feature as-is (duplicates are not averaged).")

    def add_method_parameters(self):
        self.addParameter(QgsProcessingParameterEnum(
            self.METHOD, self.tr("Interpolation method"),
            [self.tr("Linear"), self.tr("Clough-Tocher (cubic)")], defaultValue=0))

    def interpolate(self, pts, grid, parameters, context, feedback):
        data = QgsInterpolator.LayerData()
        data.source = pts.source
        data.transformContext = context.transformContext()
        if pts.field_index >= 0:
            data.valueSource = _enum(QgsInterpolator, "ValueSource", "Attribute", "ValueAttribute")
            data.interpolationAttribute = pts.field_index
        else:
            data.valueSource = _enum(QgsInterpolator, "ValueSource", "Z", "ValueZ")
        data.sourceType = _enum(QgsInterpolator, "SourceType", "Points", "SourcePoints")

        method = self.parameterAsEnum(parameters, self.METHOD, context)
        kind = _enum(QgsTinInterpolator, "TinInterpolation",
                     "Linear" if method == 0 else "CloughTocher")
        interpolator = QgsTinInterpolator([data], kind, feedback)

        tmp = QgsProcessingUtils.generateTempFilename("tin_surface.tif", context)
        bbox = QgsRectangle(grid.xmin, grid.ymin, grid.xmax, grid.ymax)
        writer = QgsGridFileWriter(interpolator, tmp, bbox, grid.cols, grid.rows)
        if writer.writeFile(feedback) != 0 or feedback.isCanceled():
            raise QgsProcessingException(self.tr("TIN interpolation failed or was cancelled."))

        ds = gdal.Open(tmp)
        if ds is None:
            raise QgsProcessingException(self.tr("Could not read the TIN grid."))
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray().astype(float)
        nd = band.GetNoDataValue()
        if nd is not None:
            arr[arr == nd] = np.nan
        ds = None
        return {self.OUTPUT: arr}
