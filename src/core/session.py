"""Session data container for managing all sensor data in a recording session."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from pathlib import Path
from .metadata import SessionMetadata
from .data_types import DetectedObject, ObjectInstance, GazeCluster


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
    
    # Object detection and tracking data
    detections: Dict[str, Dict[str, List[DetectedObject]]] = None  # {camera: {frame_id: [objects]}}
    object_instances: Dict[str, ObjectInstance] = None  # {instance_id: ObjectInstance}
    gaze_clusters: Dict[int, GazeCluster] = None  # {cluster_id: GazeCluster}
    
    # Plugin system results storage
    plugin_results: Dict[str, Any] = field(default_factory=dict)  # {plugin_name: results}
    
    def __post_init__(self):
        """Initialize derived properties after dataclass creation."""
        if self.config is None:
            self.config = {}
        
        # Initialize object detection data structures
        if self.detections is None:
            self.detections = {}
        if self.object_instances is None:
            self.object_instances = {}
        if self.gaze_clusters is None:
            self.gaze_clusters = {}
        
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
        """Get session duration in seconds based on the configured timestamp sync mode."""
        end_ts = self.end_timestamp
        start_ts = self.start_timestamp
        if end_ts == 0 and start_ts == 0:
            return 0.0
        return (end_ts - start_ts) / 1e9
    
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
        """Get session start timestamp in nanoseconds.
        
        Behavior depends on timestamp_sync_mode config:
        - "union": Start when first sensor starts (minimum timestamp)
        - "intersection": Start when all sensors start (maximum timestamp)
        """
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
        
        if not timestamps:
            return 0
        
        # Check timestamp synchronization mode from config
        sync_mode = self.config.get('timestamp_sync_mode', 'union')
        if sync_mode == 'intersection':
            # Start when all sensors start (latest start time)
            return max(timestamps)
        else:
            # Default: union mode - start when first sensor starts (earliest start time)
            return min(timestamps)
    
    @property
    def end_timestamp(self) -> int:
        """Get session end timestamp in nanoseconds.
        
        Behavior depends on timestamp_sync_mode config:
        - "union": End when last sensor stops (maximum timestamp)
        - "intersection": End when any sensor stops (minimum timestamp)
        """
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
        
        if not timestamps:
            return 0
            
        # Check timestamp synchronization mode from config
        sync_mode = self.config.get('timestamp_sync_mode', 'union')
        if sync_mode == 'intersection':
            # End when any sensor stops (earliest end time)
            return min(timestamps)
        else:
            # Default: union mode - end when last sensor stops (latest end time)
            return max(timestamps)
    
    def validate_timestamp_range(self) -> bool:
        """Validate that the timestamp range is valid for the current sync mode.
        
        Returns:
            True if the range is valid, False if end < start
        """
        return self.start_timestamp <= self.end_timestamp
    
    def get_timestamp_sync_info(self) -> Dict[str, Any]:
        """Get information about timestamp synchronization mode and its effects.
        
        Returns:
            Dictionary with sync mode info and data availability
        """
        sync_mode = self.config.get('timestamp_sync_mode', 'union')
        
        # Collect all start and end timestamps
        start_times = []
        end_times = []
        sensor_names = []
        
        if not self.gaze.empty:
            start_times.append(self.gaze['timestamp'].min())
            end_times.append(self.gaze['timestamp'].max())
            sensor_names.append('gaze')
        
        for camera_name, metadata_df in self.camera_metadata.items():
            if not metadata_df.empty and 'timestamp' in metadata_df.columns:
                start_times.append(metadata_df['timestamp'].min())
                end_times.append(metadata_df['timestamp'].max())
                sensor_names.append(f'camera_{camera_name}')
        
        for camera_name, gaze_coords_df in self.gaze_screen_coords.items():
            if gaze_coords_df is not None and not gaze_coords_df.empty and 'timestamp' in gaze_coords_df.columns:
                start_times.append(gaze_coords_df['timestamp'].min())
                end_times.append(gaze_coords_df['timestamp'].max())
                sensor_names.append(f'gaze_coords_{camera_name}')
        
        if self.imu is not None and not self.imu.empty:
            start_times.append(self.imu['timestamp'].min())
            end_times.append(self.imu['timestamp'].max())
            sensor_names.append('imu')
        
        if not start_times:
            return {
                'sync_mode': sync_mode,
                'valid_range': True,
                'sensors': [],
                'start_timestamp': 0,
                'end_timestamp': 0,
                'effective_duration_s': 0,
                'data_loss_s': 0
            }
        
        min_start = min(start_times)
        max_start = max(start_times)
        min_end = min(end_times)
        max_end = max(end_times)
        
        if sync_mode == 'intersection':
            effective_start = max_start
            effective_end = min_end
        else:  # union
            effective_start = min_start
            effective_end = max_end
        
        union_duration = (max_end - min_start) / 1e9
        effective_duration = max(0, (effective_end - effective_start) / 1e9)
        data_loss = union_duration - effective_duration
        
        return {
            'sync_mode': sync_mode,
            'valid_range': effective_start <= effective_end,
            'sensors': sensor_names,
            'start_timestamp': effective_start,
            'end_timestamp': effective_end,
            'effective_duration_s': effective_duration,
            'data_loss_s': data_loss,
            'earliest_start': min_start,
            'latest_start': max_start,
            'earliest_end': min_end,
            'latest_end': max_end
        }
    
    def get_filtered_by_sync_mode(self) -> 'SessionData':
        """Get session data filtered according to timestamp sync mode.
        
        Returns:
            SessionData filtered based on the sync mode:
            - Union mode: returns self (all data)
            - Intersection mode: returns data only in overlapping time range
        """
        sync_mode = self.config.get('timestamp_sync_mode', 'union')
        
        if sync_mode == 'union':
            return self  # No filtering needed for union mode
        
        # For intersection mode, check if range is valid
        if not self.validate_timestamp_range():
            print("WARNING: Intersection mode resulted in invalid range (end < start)")
            print("Falling back to union mode to show all data")
            return self
        
        # Calculate the intersection time range
        sync_info = self.get_timestamp_sync_info()
        intersection_start_ns = sync_info['start_timestamp']
        intersection_end_ns = sync_info['end_timestamp']
        
        # Find the earliest possible start time (for union mode calculation)
        union_start_ns = sync_info['earliest_start']
        
        # Calculate relative offsets for get_time_range
        start_offset_s = (intersection_start_ns - union_start_ns) / 1e9
        end_offset_s = (intersection_end_ns - union_start_ns) / 1e9
        
        # Create a temporary union-mode session to get the full range
        temp_config = self.config.copy()
        temp_config['timestamp_sync_mode'] = 'union'
        temp_session = SessionData(
            frames=self.frames,
            camera_metadata=self.camera_metadata,
            gaze_screen_coords=self.gaze_screen_coords,
            gaze=self.gaze,
            imu=self.imu,
            metadata=self.metadata,
            primary_camera=self.primary_camera,
            session_id=f"{self.session_id}_intersection",
            input_directory=self.input_directory,
            config=temp_config
        )
        
        # Filter to intersection range using the union-mode session as base
        return temp_session.get_time_range(start_offset_s, end_offset_s)
    
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
        
        # Add timestamp synchronization info
        sync_info = self.get_timestamp_sync_info()
        lines.append(f"Timestamp sync: {sync_info['sync_mode']}")
        if sync_info['data_loss_s'] > 0:
            lines.append(f"Data loss: {sync_info['data_loss_s']:.1f}s due to {sync_info['sync_mode']} mode")
        if not sync_info['valid_range']:
            lines.append("Invalid timestamp range - check sensor synchronization")
        
        return "\n".join(lines)
    
    # Object Detection and Tracking Methods
    
    def get_detections_for_frame(self, camera_name: str, frame_id: str) -> List[DetectedObject]:
        """Retrieve cached detections for a specific frame.
        
        Args:
            camera_name: Name of the camera
            frame_id: Frame identifier
            
        Returns:
            List of DetectedObject instances for the frame
        """
        return self.detections.get(camera_name, {}).get(frame_id, [])
    
    def get_detections_for_timestamp(self, camera_name: str, timestamp: int, tolerance_ms: int = 50) -> List[DetectedObject]:
        """Get detections for the frame closest to a given timestamp.
        
        Args:
            camera_name: Name of the camera
            timestamp: Target timestamp in nanoseconds
            tolerance_ms: Maximum time difference in milliseconds
            
        Returns:
            List of DetectedObject instances for the closest frame
        """
        if camera_name not in self.detections:
            return []
        
        tolerance_ns = tolerance_ms * 1e6
        best_frame = None
        min_diff = float('inf')
        
        # Find the frame with timestamp closest to target
        for frame_id, detections in self.detections[camera_name].items():
            if detections:  # Only consider frames with detections
                frame_timestamp = detections[0].timestamp  # All detections in frame have same timestamp
                diff = abs(frame_timestamp - timestamp)
                if diff < min_diff and diff <= tolerance_ns:
                    min_diff = diff
                    best_frame = frame_id
        
        if best_frame is not None:
            return self.detections[camera_name][best_frame]
        return []
    
    def add_detection_results(self, camera_name: str, frame_detections: Dict[str, List[DetectedObject]]) -> None:
        """Add detection results for a camera.
        
        Args:
            camera_name: Name of the camera
            frame_detections: Dictionary mapping frame_id to list of DetectedObject
        """
        if camera_name not in self.detections:
            self.detections[camera_name] = {}
        
        self.detections[camera_name].update(frame_detections)
    
    def get_instance_by_id(self, instance_id: str) -> Optional[ObjectInstance]:
        """Get object instance by ID.
        
        Args:
            instance_id: Unique instance identifier
            
        Returns:
            ObjectInstance if found, None otherwise
        """
        return self.object_instances.get(instance_id)
    
    def get_active_instances(self, timestamp: int, timeout_ms: int = 10000) -> List[ObjectInstance]:
        """Get object instances that are active at a given timestamp.
        
        Args:
            timestamp: Current timestamp in nanoseconds
            timeout_ms: Instance timeout in milliseconds
            
        Returns:
            List of active ObjectInstance objects
        """
        active_instances = []
        for instance in self.object_instances.values():
            if instance.is_active(timestamp, timeout_ms):
                active_instances.append(instance)
        return active_instances
    
    def add_object_instance(self, instance: ObjectInstance) -> None:
        """Add or update an object instance.
        
        Args:
            instance: ObjectInstance to add or update
        """
        self.object_instances[instance.instance_id] = instance
    
    def add_gaze_cluster(self, cluster: GazeCluster) -> None:
        """Add a gaze cluster.
        
        Args:
            cluster: GazeCluster to add
        """
        self.gaze_clusters[cluster.cluster_id] = cluster
    
    def get_gaze_cluster(self, cluster_id: int) -> Optional[GazeCluster]:
        """Get gaze cluster by ID.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            GazeCluster if found, None otherwise
        """
        return self.gaze_clusters.get(cluster_id)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get statistics about object detection results.
        
        Returns:
            Dictionary with detection statistics
        """
        stats = {
            'total_cameras_with_detections': len(self.detections),
            'total_frames_with_detections': 0,
            'total_detected_objects': 0,
            'detections_per_camera': {},
            'class_distribution': {},
            'confidence_stats': {
                'mean': 0.0,
                'min': 1.0,
                'max': 0.0,
                'std': 0.0
            }
        }
        
        all_confidences = []
        
        for camera_name, camera_detections in self.detections.items():
            camera_stats = {
                'frames_with_detections': len(camera_detections),
                'total_objects': 0,
                'classes': set()
            }
            
            for frame_id, detections in camera_detections.items():
                camera_stats['total_objects'] += len(detections)
                stats['total_detected_objects'] += len(detections)
                
                for detection in detections:
                    # Update class distribution
                    class_name = detection.class_name
                    if class_name not in stats['class_distribution']:
                        stats['class_distribution'][class_name] = 0
                    stats['class_distribution'][class_name] += 1
                    camera_stats['classes'].add(class_name)
                    
                    # Collect confidence scores
                    all_confidences.append(detection.confidence)
            
            camera_stats['unique_classes'] = len(camera_stats['classes'])
            camera_stats['classes'] = list(camera_stats['classes'])
            stats['detections_per_camera'][camera_name] = camera_stats
            stats['total_frames_with_detections'] += camera_stats['frames_with_detections']
        
        # Calculate confidence statistics
        if all_confidences:
            stats['confidence_stats']['mean'] = float(np.mean(all_confidences))
            stats['confidence_stats']['min'] = float(np.min(all_confidences))
            stats['confidence_stats']['max'] = float(np.max(all_confidences))
            stats['confidence_stats']['std'] = float(np.std(all_confidences))
        
        return stats
    
    # Plugin System Methods
    
    def get_plugin_result(self, plugin_name: str) -> Optional[Any]:
        """Get results from a plugin by name.
        
        Args:
            plugin_name: Name of the plugin (usually class name)
            
        Returns:
            Plugin results if available, None otherwise
        """
        return self.plugin_results.get(plugin_name)
    
    def set_plugin_result(self, plugin_name: str, result: Any) -> None:
        """Store plugin results.
        
        Args:
            plugin_name: Name of the plugin (usually class name)
            result: Results dictionary from plugin.process()
        """
        self.plugin_results[plugin_name] = result
    
    def has_plugin_result(self, plugin_name: str) -> bool:
        """Check if plugin results are available.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            True if plugin results exist and don't contain errors
        """
        result = self.plugin_results.get(plugin_name)
        if result is None:
            return False
        # Consider results with errors as not available
        return not (isinstance(result, dict) and "error" in result)
    
    def clear_plugin_results(self) -> None:
        """Clear all plugin results."""
        self.plugin_results = {}
    
    def get_plugin_results_summary(self) -> Dict[str, str]:
        """Get summary of all plugin results.
        
        Returns:
            Dictionary mapping plugin names to status strings
        """
        summary = {}
        for plugin_name, result in self.plugin_results.items():
            if isinstance(result, dict):
                if "error" in result:
                    summary[plugin_name] = f"Error: {result['error']}"
                elif "status" in result:
                    summary[plugin_name] = result["status"]
                else:
                    summary[plugin_name] = "Completed"
            else:
                summary[plugin_name] = "Completed"
        return summary