"""Point Interpolation Toolbox - a QGIS Processing provider."""


def classFactory(iface):  # noqa: N802 (QGIS API name)
    from .plugin import PointInterpolationPlugin
    return PointInterpolationPlugin(iface)
