"""Base analytics plugin interface for MLGaze Viewer."""

from typing import Dict, Any, Optional
import rerun as rr
from src.core import SessionData


class AnalyticsPlugin:
    """Simple plugin interface for analytics - lightweight and extensible."""
    
    def __init__(self, name: str):
        """Initialize analytics plugin.
        
        Args:
            name: Human-readable name for the plugin
        """
        self.name = name
        self.enabled = True
        self.results = None
        self.entity_path = f"/analytics/{name.lower().replace(' ', '_')}"
    
    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict:
        """Process session data and return analytics results.
        
        Args:
            session: SessionData object containing all sensor data
            config: Optional configuration dictionary for the plugin
            
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
        pass  # Default implementation does nothing
    
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
        return True  # Default accepts all data
    
    def get_required_columns(self) -> Dict[str, list]:
        """Get required columns for each dataframe.
        
        Returns:
            Dictionary mapping dataframe names to required column lists
        """
        return {}  # Default has no requirements