"""Session data container for managing all sensor data in a recording session."""

from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from pathlib import Path
from .metadata import SessionMetadata


@dataclass
class SessionData:
    """Container for all sensor data in a multi-camera recording session.
    
    This class holds all the data from a single recording session,
    supporting multiple cameras with their own frames and gaze coordinates.
    Provides convenient properties for accessing common metrics and supports
    memory-efficient storage of large datasets.
    """
    
    # Multi-camera data
    frames: Dict[str, Dict[str, bytes]]  # {camera_name: {frame_id: jpeg_bytes}}
    camera_metadata: Dict[str, pd.DataFrame]  # {camera_name: frame_metadata}
    gaze_screen_coords: Dict[str, pd.DataFrame]  # {camera_name: gaze_coords}
    
    # Sensor data (shared across cameras)
    gaze: pd.DataFrame  # 3D world gaze from sensors/gaze_data.csv
    imu: Optional[pd.DataFrame] = None  # From sensors/imu_data.csv
    
    # Session information
    metadata: Optional[SessionMetadata] = None
    primary_camera: str = ""  # Selected camera for 3D visualization
    session_id: str = ""
    input_directory: str = ""
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize derived properties after dataclass creation."""
        if self.config is None:
            self.config = {}
        
        # Ensure timestamps are int64 for gaze data
        if not self.gaze.empty and 'timestamp' in self.gaze.columns:
            self.gaze['timestamp'] = self.gaze['timestamp'].astype(np.int64)
        
        # Ensure timestamps are int64 for camera metadata
        for camera_name, metadata_df in self.camera_metadata.items():
            if 'timestamp' in metadata_df.columns:
                metadata_df['timestamp'] = metadata_df['timestamp'].astype(np.int64)
        
        # Ensure timestamps are int64 for gaze screen coordinates
        for camera_name, gaze_coords_df in self.gaze_screen_coords.items():
            if gaze_coords_df is not None and 'timestamp' in gaze_coords_df.columns:
                gaze_coords_df['timestamp'] = gaze_coords_df['timestamp'].astype(np.int64)
        
        # Ensure timestamps are int64 for IMU data
        if self.imu is not None and 'timestamp' in self.imu.columns:
            self.imu['timestamp'] = self.imu['timestamp'].astype(np.int64)
        
        # Set primary camera if not specified
        if not self.primary_camera and self.frames:
            self.primary_camera = sorted(self.frames.keys())[0]
    
    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        if self.gaze.empty:
            return 0.0
        return (self.gaze['timestamp'].max() - self.gaze['timestamp'].min()) / 1e9
    
    @property
    def duration_minutes(self) -> float:
        """Get session duration in minutes."""
        return self.duration / 60.0
    
    @property
    def num_frames(self) -> int:
        """Get total number of camera frames across all cameras."""
        return sum(len(camera_frames) for camera_frames in self.frames.values())
    
    @property
    def num_cameras(self) -> int:
        """Get number of cameras in the session."""
        return len(self.frames)
    
    def get_camera_names(self) -> List[str]:
        """Get list of camera names in the session."""
        return list(self.frames.keys())
    
    def get_frames_for_camera(self, camera_name: str) -> Dict[str, bytes]:
        """Get frames for a specific camera.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            Dictionary of frame_id to JPEG bytes
        """
        return self.frames.get(camera_name, {})
    
    def get_metadata_for_camera(self, camera_name: str) -> Optional[pd.DataFrame]:
        """Get metadata for a specific camera.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            DataFrame with frame metadata or None if not found
        """
        return self.camera_metadata.get(camera_name)
    
    def get_gaze_coords_for_camera(self, camera_name: str) -> Optional[pd.DataFrame]:
        """Get gaze screen coordinates for a specific camera.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            DataFrame with gaze screen coordinates or None if not available
        """
        return self.gaze_screen_coords.get(camera_name)
    
    @property
    def num_gaze_samples(self) -> int:
        """Get number of gaze samples."""
        return len(self.gaze)
    
    @property
    def num_imu_samples(self) -> int:
        """Get number of IMU samples."""
        return len(self.imu) if self.imu is not None else 0
    
    @property
    def gaze_sampling_rate(self) -> float:
        """Calculate average gaze sampling rate in Hz."""
        if self.duration > 0:
            return self.num_gaze_samples / self.duration
        return 0.0
    
    @property
    def frame_rate(self) -> float:
        """Calculate average frame rate in FPS."""
        if self.duration > 0:
            return self.num_frames / self.duration
        return 0.0
    
    @property
    def imu_sampling_rate(self) -> float:
        """Calculate average IMU sampling rate in Hz."""
        if self.imu is not None and self.duration > 0:
            return self.num_imu_samples / self.duration
        return 0.0
    
    @property
    def start_timestamp(self) -> int:
        """Get session start timestamp in nanoseconds."""
        timestamps = []
        
        # Add gaze timestamps
        if not self.gaze.empty:
            timestamps.append(self.gaze['timestamp'].min())
        
        # Add camera metadata timestamps
        for metadata_df in self.camera_metadata.values():
            if not metadata_df.empty and 'timestamp' in metadata_df.columns:
                timestamps.append(metadata_df['timestamp'].min())
        
        # Add gaze screen coordinates timestamps
        for gaze_coords_df in self.gaze_screen_coords.values():
            if gaze_coords_df is not None and not gaze_coords_df.empty and 'timestamp' in gaze_coords_df.columns:
                timestamps.append(gaze_coords_df['timestamp'].min())
        
        # Add IMU timestamps
        if self.imu is not None and not self.imu.empty:
            timestamps.append(self.imu['timestamp'].min())
            
        return min(timestamps) if timestamps else 0
    
    @property
    def end_timestamp(self) -> int:
        """Get session end timestamp in nanoseconds."""
        timestamps = []
        
        # Add gaze timestamps
        if not self.gaze.empty:
            timestamps.append(self.gaze['timestamp'].max())
        
        # Add camera metadata timestamps
        for metadata_df in self.camera_metadata.values():
            if not metadata_df.empty and 'timestamp' in metadata_df.columns:
                timestamps.append(metadata_df['timestamp'].max())
        
        # Add gaze screen coordinates timestamps
        for gaze_coords_df in self.gaze_screen_coords.values():
            if gaze_coords_df is not None and not gaze_coords_df.empty and 'timestamp' in gaze_coords_df.columns:
                timestamps.append(gaze_coords_df['timestamp'].max())
        
        # Add IMU timestamps
        if self.imu is not None and not self.imu.empty:
            timestamps.append(self.imu['timestamp'].max())
            
        return max(timestamps) if timestamps else 0
    
    def get_time_range(self, start_s: float = None, end_s: float = None) -> 'SessionData':
        """Get a subset of the session data within a time range.
        
        Args:
            start_s: Start time in seconds from session start (None = from beginning)
            end_s: End time in seconds from session start (None = to end)
            
        Returns:
            New SessionData instance with filtered data
        """
        base_time = self.start_timestamp
        
        start_ns = base_time if start_s is None else base_time + int(start_s * 1e9)
        end_ns = self.end_timestamp if end_s is None else base_time + int(end_s * 1e9)
        
        # Filter 3D gaze data
        gaze_filtered = self.gaze[
            (self.gaze['timestamp'] >= start_ns) & 
            (self.gaze['timestamp'] <= end_ns)
        ].copy()
        
        # Filter camera metadata and frames for each camera
        camera_metadata_filtered = {}
        frames_filtered = {}
        gaze_screen_coords_filtered = {}
        
        for camera_name in self.camera_metadata.keys():
            # Filter camera metadata
            metadata_df = self.camera_metadata[camera_name]
            metadata_filtered = metadata_df[
                (metadata_df['timestamp'] >= start_ns) & 
                (metadata_df['timestamp'] <= end_ns)
            ].copy()
            camera_metadata_filtered[camera_name] = metadata_filtered
            
            # Filter frames based on filtered metadata frame IDs
            if not metadata_filtered.empty:
                frame_ids = set(metadata_filtered['frameId'].unique())  # frameId is already formatted
                camera_frames = self.frames.get(camera_name, {})
                frames_filtered[camera_name] = {k: v for k, v in camera_frames.items() if k in frame_ids}
            else:
                frames_filtered[camera_name] = {}
            
            # Filter gaze screen coordinates
            gaze_coords_df = self.gaze_screen_coords.get(camera_name)
            if gaze_coords_df is not None:
                gaze_coords_filtered = gaze_coords_df[
                    (gaze_coords_df['timestamp'] >= start_ns) & 
                    (gaze_coords_df['timestamp'] <= end_ns)
                ].copy()
                gaze_screen_coords_filtered[camera_name] = gaze_coords_filtered
            else:
                gaze_screen_coords_filtered[camera_name] = None
        
        # Filter IMU if present
        imu_filtered = None
        if self.imu is not None:
            imu_filtered = self.imu[
                (self.imu['timestamp'] >= start_ns) & 
                (self.imu['timestamp'] <= end_ns)
            ].copy()
        
        return SessionData(
            frames=frames_filtered,
            camera_metadata=camera_metadata_filtered,
            gaze_screen_coords=gaze_screen_coords_filtered,
            gaze=gaze_filtered,
            imu=imu_filtered,
            metadata=self.metadata,
            primary_camera=self.primary_camera,
            session_id=f"{self.session_id}_subset",
            input_directory=self.input_directory,
            config=self.config.copy() if self.config else {}
        )
    
    def get_gaze_state_distribution(self) -> Dict[str, int]:
        """Get distribution of gaze states in the session."""
        if 'gazeState' in self.gaze.columns:
            return self.gaze['gazeState'].value_counts().to_dict()
        return {}
    
    def get_tracking_quality(self) -> float:
        """Get percentage of samples with valid tracking."""
        if 'isTracking' in self.gaze.columns:
            return (self.gaze['isTracking'].sum() / len(self.gaze) * 100) if len(self.gaze) > 0 else 0.0
        return 100.0  # Assume all valid if no tracking column
    
    def summary(self) -> str:
        """Generate a summary string of the session data."""
        lines = [
            f"Session: {self.session_id or 'Unnamed'}",
            f"Duration: {self.duration_minutes:.1f} minutes",
            f"Cameras: {self.num_cameras}",
            f"Total frames: {self.num_frames} @ {self.frame_rate:.1f} FPS",
            f"3D gaze samples: {self.num_gaze_samples} @ {self.gaze_sampling_rate:.1f} Hz",
        ]
        
        # Add primary camera info
        if self.primary_camera:
            lines.append(f"Primary camera: {self.primary_camera}")
        
        # Add per-camera details
        if self.frames:
            lines.append("Camera details:")
            for camera_name in sorted(self.frames.keys()):
                num_frames = len(self.frames[camera_name])
                has_gaze = camera_name in self.gaze_screen_coords and self.gaze_screen_coords[camera_name] is not None
                gaze_status = "with 2D gaze" if has_gaze else "no 2D gaze"
                lines.append(f"  - {camera_name}: {num_frames} frames ({gaze_status})")
        
        # Add IMU info
        if self.imu is not None:
            lines.append(f"IMU samples: {self.num_imu_samples} @ {self.imu_sampling_rate:.1f} Hz")
        
        # Add tracking quality
        tracking_quality = self.get_tracking_quality()
        lines.append(f"Tracking quality: {tracking_quality:.1f}%")
        
        # Add gaze state distribution
        state_dist = self.get_gaze_state_distribution()
        if state_dist:
            lines.append("Gaze states:")
            for state, count in state_dist.items():
                percentage = (count / self.num_gaze_samples * 100) if self.num_gaze_samples > 0 else 0
                lines.append(f"  - {state}: {count} ({percentage:.1f}%)")
        
        return "\n".join(lines)