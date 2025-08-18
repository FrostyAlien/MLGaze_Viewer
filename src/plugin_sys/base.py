"""Base plugin interfaces for MLGaze Viewer plugin system."""

from typing import Dict, Any, Optional, List
import rerun as rr
from src.core import SessionData
from src.utils.logger import MLGazeLogger


class AnalyticsPlugin:
    """Base class for analytics plugins with dependency management."""
    
    def __init__(self, name: str):
        """Initialize analytics plugin.
        
        Args:
            name: Human-readable name for the plugin
        """
        self.name = name
        self.enabled = True
        self.results = None
        self.entity_path = f"/analytics/{name.lower().replace(' ', '_')}"
        self._logger = None  # Lazy initialization
    
    @property
    def logger(self):
        """Get logger instance for this plugin."""
        if self._logger is None:
            self._logger = MLGazeLogger().get_logger(self.__class__.__name__)
        return self._logger
    
    def get_dependencies(self) -> List[str]:
        """Return list of required plugin class names.
        
        Returns:
            List of plugin class names that must execute before this plugin
        """
        return []
    
    def get_optional_dependencies(self) -> List[str]:
        """Return list of optional plugin class names.
        
        Returns:
            List of plugin class names that can enhance this plugin if available
        """
        return []
    
    def validate_dependencies(self, available_results: Dict[str, Any]) -> bool:
        """Check if all required dependencies have results.
        
        Args:
            available_results: Dictionary of plugin_name -> results
            
        Returns:
            True if all required dependencies are satisfied
        """
        missing_deps = []
        failed_deps = []
        
        for dep in self.get_dependencies():
            if dep not in available_results:
                missing_deps.append(dep)
            elif isinstance(available_results[dep], dict) and "error" in available_results[dep]:
                failed_deps.append(dep)
        
        if missing_deps:
            self.logger.error(f"Missing required dependencies: {missing_deps}")
            return False
        if failed_deps:
            self.logger.error(f"Required dependencies failed: {failed_deps}")
            return False
        
        if self.get_dependencies():
            self.logger.debug(f"All required dependencies satisfied: {self.get_dependencies()}")
        return True
    
    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict:
        """Process session data and return analytics results.
        
        Args:
            session: SessionData object containing all sensor data
            config: Optional configuration dictionary for the plugin.
                   May include "dependencies" key with results from required plugins.
            
        Returns:
            Dictionary containing analysis results
        """
        raise NotImplementedError(f"Plugin {self.name} must implement process method")
    
    def visualize(self, results: Dict, rr_stream=None) -> None:
        """Optional: Add custom visualizations to Rerun.
        
        Args:
            results: Analysis results from process method
            rr_stream: Optional Rerun recording stream
        """
        pass
    
    def get_summary(self, results: Dict) -> str:
        """Generate a text summary of the analysis results.
        
        Args:
            results: Analysis results from process method
            
        Returns:
            Human-readable summary string
        """
        return f"Analysis complete for {self.name}"
    
    def validate_data(self, session: SessionData) -> bool:
        """Check if the session data is valid for this plugin.
        
        Args:
            session: SessionData to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Basic validation that can be overridden by plugins
        validation_issues = []
        
        if session.gaze.empty:
            validation_issues.append("No gaze data available")
        
        if not session.frames:
            validation_issues.append("No camera frames available")
        
        if validation_issues:
            self.logger.warning(f"Data validation issues for {self.name}: {validation_issues}")
            # Return True by default - let plugins decide if these are fatal
        
        return True
    
    def get_required_columns(self) -> Dict[str, list]:
        """Get required columns for each dataframe.
        
        Returns:
            Dictionary mapping dataframe names to required column lists
        """
        return {}