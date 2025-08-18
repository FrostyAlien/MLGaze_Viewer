"""Base plugin interfaces for MLGaze Viewer plugin system."""

from typing import Dict, Any, Optional, List
import rerun as rr
from src.core import SessionData


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
        for dep in self.get_dependencies():
            if dep not in available_results:
                return False
            # Check if dependency result contains error
            if isinstance(available_results[dep], dict) and "error" in available_results[dep]:
                return False
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
        return True
    
    def get_required_columns(self) -> Dict[str, list]:
        """Get required columns for each dataframe.
        
        Returns:
            Dictionary mapping dataframe names to required column lists
        """
        return {}