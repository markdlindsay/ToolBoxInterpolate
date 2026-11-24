"""Shared plumbing for the interpolation algorithms.

``PointAlgorithm`` reads the input points into numpy arrays.
``GridInterpolationAlgorithm`` adds the output grid (extent, pixel size,
optional mask polygon) and writes the resulting arrays to GDAL rasters, so a
concrete method only has to implement :meth:`interpolate`.
"""

import math
import os

import numpy as np
from osgeo import gdal, ogr, osr

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsGeometry,
    QgsRasterFileWriter,
    QgsWkbTypes,
)

from ..core.methods import deduplicate, mean_spacing
from ..core.neighbors import Cancelled, has_scipy

NODATA = -9999.0
MAX_CELLS = 60_000_000
ICON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icon.svg")


def tr(text):
    return QCoreApplication.translate("PointInterpolation", text)


def advanced(param):
    flags = param.flags() | QgsProcessingParameterDefinition.FlagAdvanced
    param.setFlags(flags)
    return param


def number(name, label, kind="double", default=0.0, minimum=None, maximum=None,
           optional=False):
    t = (QgsProcessingParameterNumber.Integer if kind == "int"
         else QgsProcessingParameterNumber.Double)
    p = QgsProcessingParameterNumber(name, label, t, default, optional)
    if minimum is not None:
        p.setMinimum(minimum)
    if maximum is not None:
        p.setMaximum(maximum)
    return p


def make_field(name, kind="double"):
    """QgsField that works on both pre- and post-3.38 QGIS."""
    if Qgis.QGIS_VERSION_INT >= 33800:
        from qgis.PyQt.QtCore import QMetaType
        t = {"double": QMetaType.Double, "int": QMetaType.Int,
             "string": QMetaType.QString}[kind]
    else:
        t = {"double": QVariant.Double, "int": QVariant.Int,
             "string": QVariant.String}[kind]
    return QgsField(name, t)


def table_fields(spec):
    fields = QgsFields()
    for name, kind in spec:
        fields.append(make_field(name, kind))
    return fields


def add_row(sink, fields, values):
    f = QgsFeature(fields)
    f.setAttributes([None if (isinstance(v, float) and not math.isfinite(v)) else v
                     for v in values])
    sink.addFeature(f, QgsFeatureSink.FastInsert)


class PointData:
    """Sample points pulled out of a QGIS layer."""

    def __init__(self, xy, z, crs, source, field_index, n_skipped, n_duplicates):
        self.xy = xy
        self.z = z
        self.crs = crs
        self.source = source
        self.field_index = field_index  # -1 means geometry Z
        self.n_skipped = n_skipped
        self.n_duplicates = n_duplicates

    @property
    def n(self):
        return len(self.z)


class PointAlgorithm(QgsProcessingAlgorithm):
    """Base for every algorithm in the toolbox: input points + value field."""

    INPUT = "INPUT"
    FIELD = "FIELD"

    _group = "Interpolation"
    _group_id = "interpolation"

    # --- QgsProcessingAlgorithm boilerplate ---------------------------------
    def tr(self, text):
        return tr(text)

    def createInstance(self):
        return type(self)()

    def group(self):
        return self.tr(self._group)

    def groupId(self):
        return self._group_id

    def icon(self):
        return QIcon(ICON)

    def helpUrl(self):
        return ""

    # --- parameters ---------------------------------------------------------
    def add_point_parameters(self):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT, self.tr("Input point layer"), [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterField(
            self.FIELD, self.tr("Value field (leave empty to use the geometry Z value)"),
            parentLayerParameterName=self.INPUT,
            type=QgsProcessingParameterField.Numeric, optional=True))

    # --- helpers ------------------------------------------------------------
    def read_points(self, parameters, context, feedback, min_points=1):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        if hasattr(self, "parameterAsStrings"):  # QGIS >= 3.40
            fields = self.parameterAsStrings(parameters, self.FIELD, context)
        else:
            fields = self.parameterAsFields(parameters, self.FIELD, context)
        field_name = fields[0] if fields else None
        field_index = source.fields().lookupField(field_name) if field_name else -1
        if field_name and field_index < 0:
            raise QgsProcessingException(self.tr("Field not found: {}").format(field_name))

        request = QgsFeatureRequest()
        if field_index >= 0:
            request.setSubsetOfAttributes([field_index])
        else:
            request.setNoAttributes()
            if not QgsWkbTypes.hasZ(source.wkbType()):
                raise QgsProcessingException(self.tr(
                    "No value field was chosen and the layer has no Z values. "
                    "Pick a numeric field to interpolate."))

        xs, ys, zs = [], [], []
        skipped = 0
        total = source.featureCount() or 1
        for n, feat in enumerate(source.getFeatures(request)):
            if feedback.isCanceled():
                raise QgsProcessingException(self.tr("Cancelled."))
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                skipped += 1
                continue
            if field_index >= 0:
                try:
                    val = float(feat.attribute(field_index))
                except (TypeError, ValueError):
                    skipped += 1
                    continue
            for v in geom.vertices():
                value = val if field_index >= 0 else v.z()
                if value is None or not math.isfinite(value):
                    skipped += 1
                    continue
                xs.append(v.x())
                ys.append(v.y())
                zs.append(value)
            if n % 1000 == 0:
                feedback.setProgress(5.0 * n / total)

        xy = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)]) \
            if xs else np.zeros((0, 2))
        z = np.asarray(zs, float)
        xy, z, dup = deduplicate(xy, z) if len(z) else (xy, z, 0)
        if len(z) < min_points:
            raise QgsProcessingException(self.tr(
                "At least {} valid points are needed, found {}.").format(min_points, len(z)))

        feedback.pushInfo(self.tr("Using {} points ({} values skipped as null/empty, "
                                  "{} duplicate locations averaged).")
                          .format(len(z), skipped, dup))
        if not has_scipy():
            feedback.pushInfo(self.tr("scipy not found: using numpy neighbour search "
                                      "(slower on large datasets)."))
        return PointData(xy, z, source.sourceCrs(), source, field_index, skipped, dup)

    def progress_callback(self, feedback, start=5.0, end=95.0):
        def cb(fraction):
            if feedback.isCanceled():
                raise Cancelled()
            feedback.setProgress(start + (end - start) * fraction)
        return cb


class Grid:
    """Regular output grid anchored at the top-left of the extent."""

    def __init__(self, xmin, ymax, pixel, cols, rows):
        self.xmin, self.ymax, self.pixel = xmin, ymax, pixel
        self.cols, self.rows = cols, rows

    @property
    def xmax(self):
        return self.xmin + self.cols * self.pixel

    @property
    def ymin(self):
        return self.ymax - self.rows * self.pixel

    @property
    def geotransform(self):
        return (self.xmin, self.pixel, 0.0, self.ymax, 0.0, -self.pixel)

    def cell_centres(self):
        x = self.xmin + (np.arange(self.cols) + 0.5) * self.pixel
        y = self.ymax - (np.arange(self.rows) + 0.5) * self.pixel
        X, Y = np.meshgrid(x, y)
        return np.column_stack([X.ravel(), Y.ravel()])


class GridInterpolationAlgorithm(PointAlgorithm):
    """Interpolate points onto a raster grid."""

    EXTENT = "EXTENT"
    PIXEL_SIZE = "PIXEL_SIZE"
    MASK = "MASK"
    OUTPUT = "OUTPUT"

    #: minimum number of points the method needs
    min_points = 1

    def initAlgorithm(self, config=None):
        self.add_point_parameters()
        self.add_method_parameters()
        self.addParameter(QgsProcessingParameterExtent(
            self.EXTENT, self.tr("Output extent (default: extent of the points)"),
            optional=True))
        self.addParameter(number(
            self.PIXEL_SIZE, self.tr("Pixel size (0 = automatic, ~400 cells on the long side)"),
            default=0.0, minimum=0.0))
        self.addParameter(advanced(QgsProcessingParameterFeatureSource(
            self.MASK, self.tr("Mask polygons (cells outside are set to no-data)"),
            [QgsProcessing.TypeVectorPolygon], optional=True)))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, self.tr("Interpolated surface")))
        self.add_extra_outputs()

    # --- hooks for subclasses -------------------------------------------------
    def add_method_parameters(self):
        pass

    def add_extra_outputs(self):
        pass

    def interpolate(self, pts, grid, parameters, context, feedback):
        """Return ``{output_parameter_name: 2-D array (rows, cols)}``."""
        raise NotImplementedError

    # --- main -------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        pts = self.read_points(parameters, context, feedback, self.min_points)
        grid = self.build_grid(pts, parameters, context, feedback)
        try:
            arrays = self.interpolate(pts, grid, parameters, context, feedback)
        except Cancelled:
            raise QgsProcessingException(self.tr("Cancelled."))
        except MemoryError:
            raise QgsProcessingException(self.tr(
                "Out of memory. Try a larger pixel size or fewer neighbours."))

        mask = self.mask_array(pts, grid, parameters, context, feedback)
        results = {}
        for name, arr in arrays.items():
            path = self.parameterAsOutputLayer(parameters, name, context)
            if not path:
                continue
            arr = np.asarray(arr, dtype=float).reshape(grid.rows, grid.cols)
            if mask is not None:
                arr = np.where(mask, arr, np.nan)
            self.write_raster(path, arr, grid, pts.crs)
            results[name] = path
            finite = arr[np.isfinite(arr)]
            if finite.size:
                feedback.pushInfo(self.tr("{}: min {:.6g}, max {:.6g}, mean {:.6g}, "
                                          "{} no-data cells").format(
                    name, finite.min(), finite.max(), finite.mean(), arr.size - finite.size))
            else:
                feedback.reportError(self.tr("{}: every cell is no-data. Check the search "
                                             "radius / minimum points settings.").format(name))
        feedback.setProgress(100)
        return results

    def build_grid(self, pts, parameters, context, feedback):
        extent = None
        if parameters.get(self.EXTENT):
            extent = self.parameterAsExtent(parameters, self.EXTENT, context, pts.crs)
        if extent is not None and not extent.isNull() and extent.width() > 0:
            xmin, xmax = extent.xMinimum(), extent.xMaximum()
            ymin, ymax = extent.yMinimum(), extent.yMaximum()
        else:
            xmin, ymin = pts.xy.min(axis=0)
            xmax, ymax = pts.xy.max(axis=0)

        px = self.parameterAsDouble(parameters, self.PIXEL_SIZE, context)
        span = max(xmax - xmin, ymax - ymin)
        if span <= 0:
            span = max(mean_spacing(pts.xy), 1.0)
        if px <= 0:
            px = span / 400.0
            feedback.pushInfo(self.tr("Automatic pixel size: {:.6g}").format(px))
        # pad degenerate (line-like) extents by one pixel each side
        if xmax - xmin <= 0:
            xmin, xmax = xmin - px, xmax + px
        if ymax - ymin <= 0:
            ymin, ymax = ymin - px, ymax + px

        cols = max(1, int(math.ceil((xmax - xmin) / px - 1e-9)))
        rows = max(1, int(math.ceil((ymax - ymin) / px - 1e-9)))
        if cols * rows > MAX_CELLS:
            raise QgsProcessingException(self.tr(
                "The output grid would have {:,} cells ({} x {}). Increase the pixel size."
            ).format(cols * rows, cols, rows))
        feedback.pushInfo(self.tr("Output grid: {} columns x {} rows, pixel size {:.6g}")
                          .format(cols, rows, px))
        return Grid(xmin, ymax, px, cols, rows)

    def mask_array(self, pts, grid, parameters, context, feedback):
        if not parameters.get(self.MASK):
            return None
        mask_src = self.parameterAsSource(parameters, self.MASK, context)
        if mask_src is None:
            return None
        arr = rasterize_polygons(mask_src, pts.crs, context, grid.cols, grid.rows,
                                 grid.geotransform)
        feedback.pushInfo(self.tr("Mask keeps {:,} of {:,} cells.").format(
            int(arr.sum()), arr.size))
        return arr

    @staticmethod
    def write_raster(path, arr, grid, crs):
        write_array(path, [arr], grid.geotransform, crs)


def rasterize_polygons(source, crs, context, cols, rows, geotransform):
    """Boolean (rows, cols) array: True where a cell centre is inside a polygon."""
    xform = QgsCoordinateTransform(source.sourceCrs(), crs, context.transformContext())
    # GDAL >= 3.11 serves in-memory vectors from "MEM"; older versions use "Memory"
    vdrv = ogr.GetDriverByName("MEM") if int(gdal.VersionInfo()) >= 3110000 else None
    ds = (vdrv or ogr.GetDriverByName("Memory")).CreateDataSource("mask")
    layer = ds.CreateLayer("mask", geom_type=ogr.wkbMultiPolygon)
    for feat in source.getFeatures(QgsFeatureRequest().setNoAttributes()):
        g = feat.geometry()
        if g is None or g.isEmpty():
            continue
        g = QgsGeometry(g)
        g.transform(xform)
        of = ogr.Feature(layer.GetLayerDefn())
        of.SetGeometry(ogr.CreateGeometryFromWkb(bytes(g.asWkb())))
        layer.CreateFeature(of)
    mem = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(geotransform)
    gdal.RasterizeLayer(mem, [1], layer, burn_values=[1])
    return mem.GetRasterBand(1).ReadAsArray().astype(bool)


def write_array(path, bands, geotransform, crs, descriptions=None):
    """Write float arrays (one per band) to ``path``; nan becomes no-data."""
    rows, cols = np.shape(bands[0])
    ext = os.path.splitext(path)[1].lstrip(".")
    driver_name = QgsRasterFileWriter.driverForExtension(ext) or "GTiff"
    driver = gdal.GetDriverByName(driver_name) or gdal.GetDriverByName("GTiff")
    options = ["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES"] \
        if driver.ShortName == "GTiff" else []
    if driver.ShortName == "GTiff" and len(bands) > 1:
        options.append("INTERLEAVE=BAND")
    create = "DCAP_CREATE" in (driver.GetMetadata() or {})
    target = driver if create else gdal.GetDriverByName("MEM")
    ds = target.Create(path if create else "", cols, rows, len(bands),
                       gdal.GDT_Float32, options)
    ds.SetGeoTransform(geotransform)
    if crs is not None and crs.isValid():
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs.toWkt())
        ds.SetProjection(srs.ExportToWkt())
    for b, arr in enumerate(bands, start=1):
        band = ds.GetRasterBand(b)
        band.SetNoDataValue(NODATA)
        band.WriteArray(np.where(np.isfinite(arr), arr, NODATA).astype(np.float32))
        if descriptions:
            band.SetDescription(descriptions[b - 1])
        band.FlushCache()
    if not create:
        driver.CreateCopy(path, ds)
    ds = None
