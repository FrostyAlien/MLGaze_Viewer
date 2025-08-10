"""Gaze sensor handler for MLGaze Viewer."""

import pandas as pd
import numpy as np
import rerun as rr
from typing import Dict, Any, List
from src.sensors.base import BaseSensor
from src.core import SessionData
from src.core.coordinate_utils import unity_to_rerun_position


class GazeSensor(BaseSensor):
    """Handler for eye gaze data visualization in Rerun."""
    
    def __init__(self, entity_path: str = "/world/gaze"):
        """Initialize gaze sensor.
        
        Args:
            entity_path: Base entity path for gaze data in Rerun
        """
        super().__init__(entity_path, "Gaze Sensor")
    
    def log_to_rerun(self, session: SessionData, config: Dict[str, Any]) -> None:
        """Log gaze data to Rerun with rays, hit points, and trajectories.
        
        Args:
            session: SessionData containing gaze information
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
            
            # Set timeline
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
                
                # Log gaze ray
                rr.log(
                    f"{self.entity_path}/rays",
                    rr.Arrows3D(
                        origins=[origin],
                        vectors=[vector],
                        colors=[color],
                        radii=0.002
                    )
                )
                
                # Log hit point
                rr.log(
                    f"{self.entity_path}/hits",
                    rr.Points3D(
                        positions=[position],
                        colors=[color],
                        radii=0.01
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
    
    def log_screen_gaze(self, session: SessionData, config: Dict[str, Any]) -> None:
        """Log 2D screen gaze points to camera image space.
        
        Args:
            session: SessionData containing gaze information
            config: Visualization configuration
        """
        if session.gaze.empty:
            return
        
        # Get image dimensions from metadata
        if session.metadata.empty:
            return
        
        first_meta = session.metadata.iloc[0]
        image_height = int(first_meta['height'])
        
        print(f"Logging 2D screen gaze points...")
        
        for idx, row in session.gaze.iterrows():
            # Only log valid projections
            if row.get('isValidProjection', False) and not pd.isna(row.get('screenPixelX')):
                timestamp_ns = int(row['timestamp'])
                rr.set_time("timestamp", timestamp=1e-9 * timestamp_ns)
                
                screen_x = row['screenPixelX']
                screen_y = row['screenPixelY']
                
                # Apply Y-flip if configured
                if config.get('test_y_flip', False):
                    screen_y = image_height - screen_y
                
                color = self.get_color_for_state(row.get('gazeState', 'Unknown'))
                
                # Log 2D gaze point
                rr.log(
                    "/world/camera/image/gaze_2d",
                    rr.Points2D(
                        positions=[[screen_x, screen_y]],
                        colors=[color],
                        radii=10
                    )
                )