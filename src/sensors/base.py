"""Base sensor interface for MLGaze Viewer."""

from typing import Dict, Any, Optional
import rerun as rr
from src.core import SessionData


class BaseSensor:
    """Simple base interface for sensors - using duck typing approach."""
    
    def __init__(self, entity_path: str, name: str):
        """Initialize sensor.
        
        Args:
            entity_path: Rerun entity path for this sensor
            name: Human-readable name for the sensor
        """
        self.entity_path = entity_path
        self.name = name
        self.enabled = True
    
    def log_to_rerun(self, session: SessionData, config: Dict[str, Any]) -> None:
        """Log sensor data to Rerun.
        
        Args:
            session: SessionData containing all sensor data
            config: Configuration dictionary
        """
        raise NotImplementedError(f"Sensor {self.name} must implement log_to_rerun")
    
    def get_color_for_state(self, state: str) -> list:
        """Get RGB color for different gaze states.
        
        Args:
            state: State string (e.g., 'Fixation', 'Saccade')
            
        Returns:
            RGB color as [R, G, B] values (0-255)
        """
        colors = {
            'Fixation': [0, 255, 0],     # Green
            'Saccade': [255, 255, 0],    # Yellow  
            'Pursuit': [0, 255, 255],    # Cyan
            'WinkRight': [255, 0, 255],  # Magenta
            'WinkLeft': [255, 0, 255],   # Magenta
            'Blink': [128, 0, 128],      # Purple
        }
        return colors.get(state, [128, 128, 128])  # Gray for unknown