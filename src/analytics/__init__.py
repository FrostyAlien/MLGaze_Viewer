"""Analytics plugins for MLGaze Viewer."""

from .base import AnalyticsPlugin
from .aoi_analyzer import AOIAnalyzer

__all__ = ["AnalyticsPlugin", "AOIAnalyzer"]


def load_plugins(plugin_names: list) -> list:
    """Load analytics plugins by name.
    
    Args:
        plugin_names: List of plugin names to load
        
    Returns:
        List of instantiated plugin objects
    """
    plugins = []
    
    for name in plugin_names:
        if name == "aoi_analyzer":
            plugins.append(AOIAnalyzer())
        # Add more plugins here as they are implemented
        # elif name == "object_detector":
        #     plugins.append(ObjectDetector())
        # elif name == "dwell_analyzer":
        #     plugins.append(DwellAnalyzer())
    
    return plugins