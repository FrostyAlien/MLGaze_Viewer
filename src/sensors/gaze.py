"""3D gaze sensor handler for MLGaze Viewer."""

import pandas as pd
import numpy as np
import rerun as rr
from typing import Dict, Any, List
from src.sensors.base import BaseSensor
from src.core import SessionData
from src.core.coordinate_utils import unity_to_rerun_position


class GazeSensor(BaseSensor):
    """Handler for 3D world-space eye gaze data visualization in Rerun.
    
    This sensor handles the 3D gaze rays and hit points that are shared across
    all cameras. Per-camera 2D gaze screen coordinates are handled by CameraSensor.
    """
    
    def __init__(self, entity_path: str = "/world/gaze"):
        """Initialize 3D gaze sensor.
        
        Args:
            entity_path: Base entity path for gaze data in Rerun
        """
        super().__init__(entity_path, "3D Gaze Sensor")
    
    def log_to_rerun(self, session: SessionData, config: Dict[str, Any]) -> None:
        """Log 3D gaze data to Rerun with rays, hit points, and trajectories.
        
        This method processes the sensors/gaze_data.csv data which contains 3D
        world-space gaze vectors and hit positions that are independent of any
        specific camera.
        
        Args:
            session: SessionData containing 3D gaze information
            config: Visualization configuration
        """
        if session.gaze.empty:
            print(f"No gaze data to visualize")
            return
        
        print(f"Logging {len(session.gaze)} gaze samples to Rerun...")
        
        # Collect data for batch processing
        all_positions = []
        all_colors = []
        all_radii = []
        
        # Process each gaze sample
        for idx, row in session.gaze.iterrows():
            timestamp_ns = int(row['timestamp'])
            
            rr.set_time("timestamp", timestamp=1e-9 * timestamp_ns)
            
            # Only log if tracking and has hit target
            if row.get('isTracking', True) and row.get('hasHitTarget', False):
                # Convert Unity coordinates to Rerun
                origin_unity = [row['gazeOriginX'], row['gazeOriginY'], row['gazeOriginZ']]
                pos_unity = [row['gazePositionX'], row['gazePositionY'], row['gazePositionZ']]
                
                origin = unity_to_rerun_position(origin_unity)
                position = unity_to_rerun_position(pos_unity)
                vector = [position[0] - origin[0], position[1] - origin[1], position[2] - origin[2]]
                
                # Get color based on gaze state
                color = self.get_color_for_state(row.get('gazeState', 'Unknown'))
                
                # Adjust visualization based on hit type
                hit_type = row.get('gazeHitType', 'none')
                if hit_type == 'mesh':
                    # Mesh hits are more precise - use smaller, brighter points
                    radius = 0.008
                    # Make color slightly brighter for mesh hits
                    color = [min(255, c + 30) for c in color]
                elif hit_type == 'bbox':
                    # Bounding box hits are less precise - use larger, dimmer points
                    radius = 0.012
                    # Make color slightly dimmer for bbox hits
                    color = [int(c * 0.8) for c in color]
                else:
                    # Unknown or no hit type - use default
                    radius = 0.01
                
                rr.log(
                    f"{self.entity_path}/rays",
                    rr.Arrows3D(
                        origins=[origin],
                        vectors=[vector],
                        colors=[color],
                        radii=0.002
                    )
                )
                
                rr.log(
                    f"{self.entity_path}/hits",
                    rr.Points3D(
                        positions=[position],
                        colors=[color],
                        radii=radius
                    )
                )
                
                # Collect for static visualizations
                all_positions.append(position)
                all_colors.append(color)
                all_radii.append(0.008)
            
            # Log progress
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(session.gaze)} gaze samples...")
        
        # Create static visualizations if enabled
        self._log_static_visualizations(all_positions, all_colors, all_radii, config)
    
    def _log_static_visualizations(self, positions: List, colors: List, radii: List, 
                                  config: Dict[str, Any]) -> None:
        """Log static visualizations like point clouds and trajectories.
        
        Args:
            positions: List of 3D positions
            colors: List of RGB colors
            radii: List of point radii
            config: Visualization configuration
        """
        if not positions:
            return
        
        # Point cloud visualization
        if config.get('show_point_cloud', True):
            print(f"  Creating gaze point cloud with {len(positions)} points...")
            rr.log(
                f"{self.entity_path}/point_cloud",
                rr.Points3D(
                    positions=positions,
                    colors=colors,
                    radii=radii
                ),
                static=True
            )
        
        # Trajectory visualization
        if config.get('show_gaze_trajectory', True) and len(positions) > 1:
            print(f"  Creating gaze trajectory...")
            rr.log(
                f"{self.entity_path}/trajectory",
                rr.LineStrips3D(
                    strips=[positions],
                    colors=[[128, 128, 255]],  # Light blue
                    radii=0.001
                ),
                static=True
            )
    
