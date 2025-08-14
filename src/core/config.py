"""Configuration management for MLGaze Viewer."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import json


@dataclass
class VisualizationConfig:
    """Configuration settings for gaze visualization."""
    
    input_directory: str = "input"
    output_directory: str = "output"
    
    # Multi-camera settings
    primary_camera: str = ""  # Selected primary camera for 3D visualization
    
    enable_fade_trail: bool = True
    fade_duration: float = 5.0
    
    # Sliding window settings
    enable_sliding_window: bool = False
    sliding_window_duration: float = 10.0  # seconds
    sliding_window_update_rate: float = 0.5  # seconds between updates
    sliding_window_3d_gaze: bool = True  # Apply to 3D gaze points
    sliding_window_3d_trajectory: bool = True  # Apply to 3D trajectories
    sliding_window_camera: bool = True  # Apply to camera positions
    
    # Visualization toggles
    show_point_cloud: bool = True
    show_gaze_trajectory: bool = True
    show_camera_trajectory: bool = True
    color_by_gaze_state: bool = True
    test_y_flip: bool = False
    show_coordinate_indicators: bool = True
    
    show_imu_data: bool = True
    
    # Entity paths for Rerun
    entity_paths: Dict[str, str] = field(default_factory=lambda: {
        "world": "/world",
        "gaze": "/world/gaze",
        "camera": "/world/camera",
        "sensors": "/sensors",
        "analytics": "/analytics"
    })
    
    # Analytics settings
    enabled_plugins: List[str] = field(default_factory=lambda: [
        "fixation_detector"
    ])
    
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "fixation_detector": {
            "velocity_threshold": 30,  # degrees/sec
            "min_duration": 100,  # milliseconds
            "max_dispersion": 1.0  # degrees
        },
        "aoi_analyzer": {
            "regions": []  # Will be populated at runtime
        },
        "object_detector": {
            "model": "yolov8n",
            "confidence": 0.5,
            "device": "cpu"
        },
        "dwell_analyzer": {
            "min_dwell_time": 500,  # milliseconds
            "merge_threshold": 50  # milliseconds
        }
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self, file_path: Path) -> None:
        """Save configuration to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, file_path: Path) -> None:
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_yaml(cls, file_path: Path) -> 'VisualizationConfig':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, file_path: Path) -> 'VisualizationConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualizationConfig':
        """Create configuration from dictionary."""
        return cls(**data)
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin."""
        return self.plugin_configs.get(plugin_name, {})
    
    def update_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific plugin."""
        if plugin_name not in self.plugin_configs:
            self.plugin_configs[plugin_name] = {}
        self.plugin_configs[plugin_name].update(config)
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled."""
        return plugin_name in self.enabled_plugins