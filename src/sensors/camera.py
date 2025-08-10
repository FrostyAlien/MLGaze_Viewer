"""Camera sensor handler for MLGaze Viewer."""

import rerun as rr
import numpy as np
from typing import Dict, Any, List
from src.sensors.base import BaseSensor
from src.core import SessionData
from src.core.coordinate_utils import unity_to_rerun_position, unity_to_rerun_quaternion


class CameraSensor(BaseSensor):
    """Handler for camera data including poses and images."""
    
    def __init__(self, entity_path: str = "/world/camera"):
        """Initialize camera sensor.
        
        Args:
            entity_path: Base entity path for camera data in Rerun
        """
        super().__init__(entity_path, "Camera Sensor")
    
    def log_to_rerun(self, session: SessionData, config: Dict[str, Any]) -> None:
        """Log camera data to Rerun including poses, images, and trajectory.
        
        Args:
            session: SessionData containing camera information
            config: Visualization configuration
        """
        # Log camera intrinsics (static)
        self._log_camera_intrinsics(session)
        
        # Log camera poses and images
        self._log_camera_poses_and_images(session, config)
        
        # Log camera trajectory if enabled
        if config.get('show_camera_trajectory', True):
            self._log_camera_trajectory(session)
    
    def _log_camera_intrinsics(self, session: SessionData) -> None:
        """Log camera intrinsics as static data.
        
        Args:
            session: SessionData containing metadata
        """
        if session.metadata.empty:
            print("No camera metadata available")
            return
        
        # Get intrinsics from first frame metadata
        first_meta = session.metadata.iloc[0]
        image_width = int(first_meta['width'])
        image_height = int(first_meta['height'])
        focal_length = [first_meta['focalLengthX'], first_meta['focalLengthY']]
        principal_point = [first_meta['principalPointX'], first_meta['principalPointY']]
        
        print(f"Camera intrinsics: {image_width}x{image_height}, f={focal_length}, c={principal_point}")
        
        # Log pinhole camera model
        rr.log(
            f"{self.entity_path}/image",
            rr.Pinhole(
                focal_length=focal_length,
                principal_point=principal_point,
                width=image_width,
                height=image_height
            ),
            static=True
        )
    
    def _log_camera_poses_and_images(self, session: SessionData, config: Dict[str, Any]) -> None:
        """Log camera poses and associated images.
        
        Args:
            session: SessionData containing camera poses and frames
            config: Visualization configuration
        """
        if session.camera_poses.empty:
            print("No camera pose data available")
            return
        
        print(f"Logging {len(session.camera_poses)} camera poses and images...")
        
        camera_positions = []
        last_frame_id = None
        
        for idx, row in session.camera_poses.iterrows():
            frame_id = row.get('frameId')
            timestamp_ns = int(row['timestamp'])
            
            # Set timeline
            rr.set_time("timestamp", timestamp=1e-9 * timestamp_ns)
            
            # Convert Unity coordinates to Rerun
            cam_pos_unity = [
                row.get('cameraPositionX', 0),
                row.get('cameraPositionY', 0),
                row.get('cameraPositionZ', 0)
            ]
            cam_rot_unity = [
                row.get('cameraRotationX', 0),
                row.get('cameraRotationY', 0),
                row.get('cameraRotationZ', 0),
                row.get('cameraRotationW', 1)
            ]
            
            cam_pos = unity_to_rerun_position(cam_pos_unity)
            cam_rot = unity_to_rerun_quaternion(cam_rot_unity)
            
            # Log camera transform
            rr.log(
                self.entity_path,
                rr.Transform3D(
                    translation=cam_pos,
                    rotation=rr.Quaternion(xyzw=cam_rot)
                )
            )
            
            # Log associated image if available and changed
            if frame_id and frame_id != last_frame_id and frame_id in session.frames:
                rr.log(
                    f"{self.entity_path}/image",
                    rr.EncodedImage(
                        contents=session.frames[frame_id],
                        media_type="image/jpeg"
                    )
                )
                last_frame_id = frame_id
            
            # Collect positions for trajectory
            camera_positions.append(cam_pos)
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(session.camera_poses)} camera poses...")
        
        # Store positions for trajectory
        self._camera_positions = camera_positions
    
    def _log_camera_trajectory(self, session: SessionData) -> None:
        """Log camera movement trajectory as a static line.
        
        Args:
            session: SessionData (not used directly, positions from previous method)
        """
        if not hasattr(self, '_camera_positions') or len(self._camera_positions) < 2:
            return
        
        print(f"  Creating camera trajectory with {len(self._camera_positions)} points...")
        
        rr.log(
            "/world/camera_trajectory",
            rr.LineStrips3D(
                strips=[self._camera_positions],
                colors=[[255, 128, 0]],
                radii=0.0002
            ),
            static=True
        )