"""Session data container for managing all sensor data in a recording session."""

from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class SessionData:
    """Container for all sensor data in a recording session.
    
    This class holds all the data from a single recording session,
    including gaze, camera, IMU, and frame data. It provides convenient
    properties for accessing common metrics and supports memory-efficient
    storage of large datasets.
    """
    
    # Core data
    gaze: pd.DataFrame  # Gaze data with all fields
    frames: Dict[str, bytes]  # Compressed JPEG frames keyed by frame_id
    camera_poses: pd.DataFrame  # Camera pose data
    metadata: pd.DataFrame  # Frame metadata
    
    # Optional sensor data
    imu: Optional[pd.DataFrame] = None
    
    # Session information
    session_id: str = ""
    input_directory: str = ""
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize derived properties after dataclass creation."""
        if self.config is None:
            self.config = {}
        
        # Ensure timestamps are int64
        if 'timestamp' in self.gaze.columns:
            self.gaze['timestamp'] = self.gaze['timestamp'].astype(np.int64)
        if 'timestamp' in self.camera_poses.columns:
            self.camera_poses['timestamp'] = self.camera_poses['timestamp'].astype(np.int64)
        if self.imu is not None and 'timestamp' in self.imu.columns:
            self.imu['timestamp'] = self.imu['timestamp'].astype(np.int64)
    
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
        """Get number of camera frames."""
        return len(self.frames)
    
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
        if not self.gaze.empty:
            timestamps.append(self.gaze['timestamp'].min())
        if not self.camera_poses.empty:
            timestamps.append(self.camera_poses['timestamp'].min())
        if self.imu is not None and not self.imu.empty:
            timestamps.append(self.imu['timestamp'].min())
        return min(timestamps) if timestamps else 0
    
    @property
    def end_timestamp(self) -> int:
        """Get session end timestamp in nanoseconds."""
        timestamps = []
        if not self.gaze.empty:
            timestamps.append(self.gaze['timestamp'].max())
        if not self.camera_poses.empty:
            timestamps.append(self.camera_poses['timestamp'].max())
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
        
        # Filter dataframes
        gaze_filtered = self.gaze[
            (self.gaze['timestamp'] >= start_ns) & 
            (self.gaze['timestamp'] <= end_ns)
        ].copy()
        
        camera_filtered = self.camera_poses[
            (self.camera_poses['timestamp'] >= start_ns) & 
            (self.camera_poses['timestamp'] <= end_ns)
        ].copy()
        
        # Filter frames based on what's referenced in filtered gaze data
        frame_ids = set(gaze_filtered['frameId'].unique()) if 'frameId' in gaze_filtered.columns else set()
        frames_filtered = {k: v for k, v in self.frames.items() if k in frame_ids}
        
        # Filter IMU if present
        imu_filtered = None
        if self.imu is not None:
            imu_filtered = self.imu[
                (self.imu['timestamp'] >= start_ns) & 
                (self.imu['timestamp'] <= end_ns)
            ].copy()
        
        return SessionData(
            gaze=gaze_filtered,
            frames=frames_filtered,
            camera_poses=camera_filtered,
            metadata=self.metadata,
            imu=imu_filtered,
            session_id=f"{self.session_id}_subset",
            input_directory=self.input_directory,
            config=self.config.copy()
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
            f"Frames: {self.num_frames} @ {self.frame_rate:.1f} FPS",
            f"Gaze samples: {self.num_gaze_samples} @ {self.gaze_sampling_rate:.1f} Hz",
        ]
        
        if self.imu is not None:
            lines.append(f"IMU samples: {self.num_imu_samples} @ {self.imu_sampling_rate:.1f} Hz")
        
        tracking_quality = self.get_tracking_quality()
        lines.append(f"Tracking quality: {tracking_quality:.1f}%")
        
        state_dist = self.get_gaze_state_distribution()
        if state_dist:
            lines.append("Gaze states:")
            for state, count in state_dist.items():
                percentage = (count / self.num_gaze_samples * 100) if self.num_gaze_samples > 0 else 0
                lines.append(f"  - {state}: {count} ({percentage:.1f}%)")
        
        return "\n".join(lines)