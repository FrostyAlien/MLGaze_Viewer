"""Main Rerun visualizer for MLGaze Viewer."""

import time
import rerun as rr
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.core import SessionData, VisualizationConfig
from src.sensors import GazeSensor, CameraSensor, IMUSensor
from src.analytics.base import AnalyticsPlugin


class RerunVisualizer:
    """Main visualization orchestrator using Rerun for multi-camera sessions.
    
    This class coordinates the visualization of multi-sensor, multi-camera data
    and analytics results in Rerun's 3D viewer. It shows:
    - Primary camera in 3D world space with 3D gaze rays
    - All cameras in separate 2D views with per-camera gaze overlays
    - IMU sensor data and analytics results
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualizer.
        
        Args:
            config: Visualization configuration object
        """
        self.config = config or VisualizationConfig()
        self.sensors = []
        self.plugins = []
        self.recording_stream = None
    
    def add_sensor(self, sensor) -> None:
        """Add a sensor to visualize.
        
        Args:
            sensor: Sensor object (e.g., GazeSensor, CameraSensor)
        """
        self.sensors.append(sensor)
    
    def add_plugin(self, plugin: AnalyticsPlugin) -> None:
        """Add an analytics plugin.
        
        Args:
            plugin: Analytics plugin to run
        """
        self.plugins.append(plugin)
    
    def initialize_rerun(self, session_name: str = "ml2_gaze_viewer") -> None:
        """Initialize Rerun viewer and recording stream.
        
        Args:
            session_name: Name for the Rerun recording
        """
        print("Initializing Rerun viewer...")
        self.recording_stream = rr.RecordingStream(session_name)
        self.recording_stream.spawn()
        
        # Configure gRPC connection with longer flush timeout to prevent data loss
        print("Configuring gRPC connection with 10-second flush timeout...")
        self.recording_stream.connect_grpc(flush_timeout_sec=10.0)
        
        # Set as global recording stream
        rr.set_global_data_recording(self.recording_stream)
        
        # Set coordinate system - RDF aligns with standard camera coordinates
        rr.log("/", rr.ViewCoordinates.RDF, static=True)
        
        # Log coordinate indicators if enabled
        if self.config.show_coordinate_indicators:
            self._log_coordinate_indicators()
        
        # Initial flush after setup
        print("Flushing initial setup...")
        flush_start = time.time()
        self.recording_stream.flush(blocking=True)
        flush_time = time.time() - flush_start
        print(f"  Initial flush completed in {flush_time:.3f}s")
    
    def _log_coordinate_indicators(self) -> None:
        """Log coordinate system indicators at world origin.
        
        Creates three colored arrows at the world origin to visualize the
        Rerun RDF coordinate system (X=right, Y=down, Z=forward).
        """
        print("Adding coordinate system indicators at world origin")
        
        # RDF coordinate system: X=right, Y=down, Z=forward
        rr.log(
            "world/coords/origin",
            rr.Arrows3D(
                origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # RGB = XYZ
                radii=0.002,
            ),
            static=True
        )
    
    def visualize(self, session: SessionData, analytics_results: Optional[Dict] = None) -> None:
        """Main visualization method.
        
        Args:
            session: SessionData containing all sensor data
            analytics_results: Optional pre-computed analytics results
        """
        if not self.recording_stream:
            self.initialize_rerun(session.session_id or "ml2_gaze_viewer")
        
        print(f"\nVisualizing session: {session.session_id}")
        print(f"Duration: {session.duration_minutes:.1f} minutes")
        
        # Run analytics if not provided
        if analytics_results is None and self.plugins:
            analytics_results = self._run_analytics(session)
        
        # Visualize sensor data
        self._visualize_sensors(session)
        
        # Visualize analytics results
        if analytics_results:
            self._visualize_analytics(analytics_results)
        
        # Final flush to ensure all data is sent before exit
        print("Flushing remaining data...")
        flush_start = time.time()
        self.recording_stream.flush(blocking=True)
        flush_time = time.time() - flush_start
        print(f"  Final flush completed in {flush_time:.3f}s")
        
        print("Visualization complete! Use the Rerun viewer to explore the data.")
    
    def _run_analytics(self, session: SessionData) -> Dict:
        """Run analytics plugins on session data.
        
        Args:
            session: SessionData to analyze
            
        Returns:
            Dictionary of analytics results keyed by plugin name
        """
        print("\nRunning analytics plugins...")
        results = {}
        
        for plugin in self.plugins:
            if not plugin.enabled:
                continue
            
            print(f"  Running {plugin.name}...")
            try:
                # Get plugin-specific config
                plugin_config = self.config.get_plugin_config(
                    plugin.name.lower().replace(' ', '_')
                )
                
                # Run analysis
                plugin_results = plugin.process(session, plugin_config)
                results[plugin.name] = plugin_results
                
                summary = plugin.get_summary(plugin_results)
                if summary:
                    print(f"    {summary}")
                    
            except Exception as e:
                print(f"    Error in {plugin.name}: {e}")
                results[plugin.name] = {'error': str(e)}
        
        return results
    
    def _visualize_sensors(self, session: SessionData) -> None:
        """Visualize all sensor data.
        
        Args:
            session: SessionData containing sensor information
        """
        print("\nVisualizing sensor data...")
        
        # If no sensors explicitly added, create default ones
        if not self.sensors:
            self.sensors = [
                GazeSensor(),
                CameraSensor(),
                IMUSensor()
            ]
        
        # Visualize each sensor
        for sensor in self.sensors:
            if not sensor.enabled:
                continue
            
            print(f"  Visualizing {sensor.name}...")
            try:
                # Convert config to dict and add primary camera info
                config_dict = self.config.to_dict()
                config_dict['primary_camera'] = self.config.primary_camera
                
                sensor.log_to_rerun(session, config_dict)
                    
            except Exception as e:
                print(f"    Error visualizing {sensor.name}: {e}")
    
    def _visualize_analytics(self, results: Dict) -> None:
        """Visualize analytics results.
        
        Args:
            results: Dictionary of analytics results
        """
        print("\nVisualizing analytics results...")
        
        for plugin in self.plugins:
            if plugin.name not in results:
                continue
            
            plugin_results = results[plugin.name]
            if 'error' in plugin_results:
                continue
            
            print(f"  Visualizing {plugin.name} results...")
            try:
                plugin.visualize(plugin_results, self.recording_stream)
            except Exception as e:
                print(f"    Error visualizing {plugin.name}: {e}")