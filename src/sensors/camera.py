"""Multi-camera sensor handler for MLGaze Viewer."""

import rerun as rr
import numpy as np
from typing import Dict, Any, List
from src.sensors.base import BaseSensor
from src.core import SessionData
from src.core.coordinate_utils import unity_to_rerun_position, unity_to_rerun_quaternion


class CameraSensor(BaseSensor):
    """Handler for multi-camera data including poses, images, and trajectories."""
    
    def __init__(self, entity_path: str = "/cameras"):
        """Initialize multi-camera sensor.
        
        Args:
            entity_path: Base entity path for camera data in Rerun
        """
        super().__init__(entity_path, "Multi-Camera Sensor")
    
    def log_to_rerun(self, session: SessionData, config: Dict[str, Any]) -> None:
        """Log multi-camera data to Rerun with primary camera in 3D and all in 2D views.
        
        This creates two types of camera visualizations:
        - Primary camera: Logged at /world/camera with 3D poses and movement
        - All cameras: Logged at /cameras/{camera_name}/image for 2D views only
        
        NOTE: In Rerun viewer, you may want to hide the /cameras entity in the 3D view
        to avoid visual clutter, since only the primary camera should move in 3D space.
        
        Args:
            session: SessionData containing multi-camera information
            config: Visualization configuration
        """
        if not session.frames:
            print("No camera data available")
            return
        
        # Get primary camera from config or use first available
        primary_camera = config.get('primary_camera', '') or session.primary_camera
        if not primary_camera or primary_camera not in session.frames:
            available_cameras = list(session.frames.keys())
            primary_camera = sorted(available_cameras)[0] if available_cameras else None
        
        if not primary_camera:
            print("No cameras available for 3D visualization")
            return
        
        print(f"Processing {len(session.frames)} cameras, primary: {primary_camera}")
        print(f"Available cameras: {list(session.frames.keys())}")
        print(f"Primary camera from config: {config.get('primary_camera', 'NOT_SET')}")
        print(f"Primary camera from session: {session.primary_camera}")
        
        # Log primary camera in 3D world space
        self._log_3d_camera(session, primary_camera, config)
        
        # Log all cameras in 2D views with gaze overlays
        for camera_name in session.frames.keys():
            self._log_2d_camera_view(session, camera_name, config)
    
    def _log_3d_camera(self, session: SessionData, camera_name: str, config: Dict[str, Any]) -> None:
        """Log primary camera in 3D world space with poses and images.
        
        Args:
            session: SessionData containing camera information
            camera_name: Name of the camera to use for 3D visualization
            config: Visualization configuration
        """
        camera_metadata = session.get_metadata_for_camera(camera_name)
        camera_frames = session.get_frames_for_camera(camera_name)
        
        if camera_metadata is None or camera_metadata.empty:
            print(f"No metadata for camera: {camera_name}")
            return
        
        print(f"Logging primary camera '{camera_name}' in 3D space...")
        
        # Log camera intrinsics (static)
        self._log_camera_intrinsics_3d(camera_metadata, "/world/camera")
        
        # Log camera poses and images over time
        self._log_camera_poses_and_images_3d(camera_metadata, camera_frames, "/world/camera")
        
        # Log camera trajectory if enabled
        if config.get('show_camera_trajectory', True):
            self._log_camera_trajectory_3d(camera_metadata)
    
    def _log_2d_camera_view(self, session: SessionData, camera_name: str, config: Dict[str, Any]) -> None:
        """Log camera in its own 2D view with gaze overlay.
        
        Args:
            session: SessionData containing camera information
            camera_name: Name of the camera
            config: Visualization configuration
        """
        camera_metadata = session.get_metadata_for_camera(camera_name)
        camera_frames = session.get_frames_for_camera(camera_name)
        gaze_coords = session.get_gaze_coords_for_camera(camera_name)
        
        if camera_metadata is None or camera_metadata.empty:
            print(f"No metadata for camera: {camera_name}")
            return
        
        entity_path = f"{self.entity_path}/{camera_name}"
        print(f"Logging camera '{camera_name}' in 2D view at {entity_path}...")
        
        # Log camera intrinsics (static)
        self._log_camera_intrinsics_2d(camera_metadata, entity_path)
        
        # Log frames and gaze overlays over time
        self._log_camera_frames_2d(camera_metadata, camera_frames, gaze_coords, entity_path)
    
    def _log_camera_intrinsics_3d(self, camera_metadata, entity_path: str) -> None:
        """Log camera intrinsics for 3D visualization.
        
        Args:
            camera_metadata: DataFrame with camera metadata
            entity_path: Entity path for the camera
        """
        if camera_metadata.empty:
            return
        
        # Get intrinsics from first frame metadata
        first_meta = camera_metadata.iloc[0]
        image_width = int(first_meta.get('width', 1280))
        image_height = int(first_meta.get('height', 960))
        focal_length = [
            first_meta.get('focalLengthX', 500),
            first_meta.get('focalLengthY', 500)
        ]
        principal_point = [
            first_meta.get('principalPointX', image_width / 2),
            first_meta.get('principalPointY', image_height / 2)
        ]
        
        print(f"  3D Camera intrinsics: {image_width}x{image_height}, f={focal_length}")
        
        rr.log(
            f"{entity_path}/image",
            rr.Pinhole(
                focal_length=focal_length,
                principal_point=principal_point,
                width=image_width,
                height=image_height
            ),
            static=True
        )
    
    def _log_camera_intrinsics_2d(self, camera_metadata, entity_path: str) -> None:
        """Log camera intrinsics for 2D visualization.
        
        Args:
            camera_metadata: DataFrame with camera metadata
            entity_path: Entity path for the camera
        """
        if camera_metadata.empty:
            return
        
        # Get intrinsics from first frame metadata
        first_meta = camera_metadata.iloc[0]
        image_width = int(first_meta.get('width', 1280))
        image_height = int(first_meta.get('height', 960))
        focal_length = [
            first_meta.get('focalLengthX', 500),
            first_meta.get('focalLengthY', 500)
        ]
        principal_point = [
            first_meta.get('principalPointX', image_width / 2),
            first_meta.get('principalPointY', image_height / 2)
        ]
        
        print(f"  2D Camera intrinsics: {image_width}x{image_height}")
        
        rr.log(
            f"{entity_path}/image",
            rr.Pinhole(
                focal_length=focal_length,
                principal_point=principal_point,
                width=image_width,
                height=image_height
            ),
            static=True
        )
    
    def _log_camera_poses_and_images_3d(self, camera_metadata, camera_frames, entity_path: str) -> None:
        """Log camera poses and images for 3D visualization.
        
        Args:
            camera_metadata: DataFrame with camera metadata
            camera_frames: Dictionary of frame data
            entity_path: Entity path for the camera
        """
        print(f"  Logging {len(camera_metadata)} 3D camera poses and images...")
        
        camera_positions = []
        last_frame_id = None
        
        for idx, row in camera_metadata.iterrows():
            frame_id = row['frameId']  # Use frameId as-is from metadata (may include camera prefix)
            timestamp_ns = int(row['timestamp'])
            
            rr.set_time("timestamp", timestamp=timestamp_ns * 1e-9)
            
            # Extract camera pose from metadata if available
            cam_pos_unity = [
                row.get('posX', 0),
                row.get('posY', 0),
                row.get('posZ', 0)
            ]
            cam_rot_unity = [
                row.get('rotX', 0),
                row.get('rotY', 0),
                row.get('rotZ', 0),
                row.get('rotW', 1)
            ]
            
            cam_pos = unity_to_rerun_position(cam_pos_unity)
            cam_rot = unity_to_rerun_quaternion(cam_rot_unity)
            
            rr.log(
                entity_path,
                rr.Transform3D(
                    translation=cam_pos,
                    rotation=rr.Quaternion(xyzw=cam_rot)
                )
            )
            
            # Log associated image if available and changed
            if frame_id != last_frame_id and frame_id in camera_frames:
                rr.log(
                    f"{entity_path}/image",
                    rr.EncodedImage(
                        contents=camera_frames[frame_id],
                        media_type="image/jpeg"
                    )
                )
                last_frame_id = frame_id
            
            # Collect positions for trajectory
            camera_positions.append(cam_pos)
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx + 1}/{len(camera_metadata)} poses...")
        
        # Store positions for trajectory
        self._camera_positions = camera_positions
    
    def _log_camera_frames_2d(self, camera_metadata, camera_frames, gaze_coords, entity_path: str) -> None:
        """Log camera frames with gaze overlays for 2D visualization.
        
        Args:
            camera_metadata: DataFrame with camera metadata
            camera_frames: Dictionary of frame data
            gaze_coords: DataFrame with gaze screen coordinates (can be None)
            entity_path: Entity path for the camera
        """
        print(f"  Logging {len(camera_metadata)} 2D camera frames with gaze overlay...")
        
        last_frame_id = None
        
        for idx, row in camera_metadata.iterrows():
            frame_id = row['frameId']  # Use frameId as-is from metadata (may include camera prefix)
            timestamp_ns = int(row['timestamp'])
            
            rr.set_time("timestamp", timestamp=timestamp_ns * 1e-9)
            
            # Log frame if available and changed
            if frame_id != last_frame_id and frame_id in camera_frames:
                rr.log(
                    f"{entity_path}/image",
                    rr.EncodedImage(
                        contents=camera_frames[frame_id],
                        media_type="image/jpeg"
                    )
                )
                last_frame_id = frame_id
            
            # Log gaze point if available
            if gaze_coords is not None:
                gaze_at_time = gaze_coords[gaze_coords['timestamp'] == timestamp_ns]
                if not gaze_at_time.empty:
                    gaze_row = gaze_at_time.iloc[0]
                    if gaze_row.get('isTracking', True):
                        pixel_x = gaze_row.get('screenPixelX', 0)
                        pixel_y = gaze_row.get('screenPixelY', 0)
                        gaze_state = gaze_row.get('gazeState', 'Unknown')
                        
                        # Get color for gaze state
                        color = self.get_color_for_state(gaze_state)
                        
                        # Log gaze point
                        rr.log(
                            f"{entity_path}/gaze",
                            rr.Points2D(
                                positions=[[pixel_x, pixel_y]],
                                colors=[color],
                                radii=[10]
                            )
                        )
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx + 1}/{len(camera_metadata)} frames...")
    
    def _log_camera_trajectory_3d(self, camera_metadata) -> None:
        """Log camera movement trajectory as a static line for 3D visualization.
        
        Args:
            camera_metadata: DataFrame with camera metadata
        """
        if not hasattr(self, '_camera_positions') or len(self._camera_positions) < 2:
            return
        
        print(f"  Creating camera trajectory with {len(self._camera_positions)} points...")
        
        rr.log(
            "/world/camera_trajectory",
            rr.LineStrips3D(
                strips=[self._camera_positions],
                colors=[[255, 128, 0]],  # Orange
                radii=0.0002
            ),
            static=True
        )