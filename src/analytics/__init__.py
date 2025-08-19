"""Analytics plugins for MLGaze Viewer."""

from src.plugin_sys.base import AnalyticsPlugin
from .aoi_analyzer import AOIAnalyzer
from .object_detector import ObjectDetector
from .gaze_object_interaction import GazeObjectInteraction

__all__ = ["AnalyticsPlugin", "AOIAnalyzer", "ObjectDetector", "GazeObjectInteraction"]


def load_plugins(plugin_names: list) -> list:
    """Load analytics plugins by name.
    
    Args:
        plugin_names: List of plugin class names to load
        
    Returns:
        List of instantiated plugin objects
    """
    plugins = []
    
    for name in plugin_names:
        # Updated to use class names consistently
        if name == "AOIAnalyzer":
            plugins.append(AOIAnalyzer())
        elif name == "ObjectDetector":
            plugins.append(ObjectDetector())
        elif name == "GazeObjectInteraction":
            plugins.append(GazeObjectInteraction())
        # Add more plugins here as they are implemented
    
    return plugins