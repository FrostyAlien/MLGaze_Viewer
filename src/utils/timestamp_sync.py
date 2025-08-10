"""Timestamp synchronization utilities for multi-sensor data."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class TimestampSynchronizer:
    """Synchronize timestamps across multiple sensors."""
    
    def __init__(self, tolerance_ms: float = 16.67):
        """Initialize synchronizer.
        
        Args:
            tolerance_ms: Maximum time difference in milliseconds to consider synchronized
                         Default is 16.67ms (one frame at 60 FPS)
        """
        self.tolerance_ns = int(tolerance_ms * 1e6)  # Convert to nanoseconds
    
    def find_nearest_timestamp(self, target_time: int, timestamps: np.ndarray) -> Tuple[int, int]:
        """Find the nearest timestamp in an array.
        
        Args:
            target_time: Target timestamp in nanoseconds
            timestamps: Array of timestamps to search
            
        Returns:
            Tuple of (nearest_timestamp, index)
        """
        if len(timestamps) == 0:
            return None, -1
        
        idx = np.argmin(np.abs(timestamps - target_time))
        return timestamps[idx], idx
    
    def synchronize_to_frames(self, gaze_df: pd.DataFrame, 
                             frame_timestamps: pd.Series) -> pd.DataFrame:
        """Synchronize gaze data to frame timestamps.
        
        Args:
            gaze_df: DataFrame with gaze data including 'timestamp' column
            frame_timestamps: Series of frame timestamps
            
        Returns:
            DataFrame with added 'sync_frame_timestamp' and 'sync_time_diff' columns
        """
        gaze_sync = gaze_df.copy()
        frame_times = frame_timestamps.values
        
        sync_frame_times = []
        sync_diffs = []
        
        for gaze_time in gaze_df['timestamp'].values:
            nearest_frame, _ = self.find_nearest_timestamp(gaze_time, frame_times)
            if nearest_frame is not None:
                sync_frame_times.append(nearest_frame)
                sync_diffs.append(abs(gaze_time - nearest_frame))
            else:
                sync_frame_times.append(None)
                sync_diffs.append(None)
        
        gaze_sync['sync_frame_timestamp'] = sync_frame_times
        gaze_sync['sync_time_diff_ns'] = sync_diffs
        gaze_sync['is_synchronized'] = gaze_sync['sync_time_diff_ns'] <= self.tolerance_ns
        
        return gaze_sync
    
    def calculate_sensor_offsets(self, session_data) -> Dict[str, float]:
        """Calculate time offsets between different sensors.
        
        Args:
            session_data: SessionData object
            
        Returns:
            Dictionary of sensor offsets in milliseconds
        """
        offsets = {}
        
        # Use first gaze timestamp as reference
        if not session_data.gaze.empty:
            ref_time = session_data.gaze['timestamp'].iloc[0]
            
            # Camera offset
            if not session_data.camera_poses.empty:
                cam_first = session_data.camera_poses['timestamp'].iloc[0]
                offsets['camera_offset_ms'] = (cam_first - ref_time) / 1e6
            
            # IMU offset
            if session_data.imu is not None and not session_data.imu.empty:
                imu_first = session_data.imu['timestamp'].iloc[0]
                offsets['imu_offset_ms'] = (imu_first - ref_time) / 1e6
        
        return offsets
    
    def analyze_synchronization(self, session_data, sample_rate: float = 0.1) -> Dict:
        """Analyze synchronization quality between sensors.
        
        Args:
            session_data: SessionData object
            sample_rate: Fraction of data to sample for analysis (0.0-1.0)
            
        Returns:
            Dictionary with synchronization metrics
        """
        metrics = {}
        
        # Analyze gaze-frame synchronization
        if 'frameId' in session_data.gaze.columns and not session_data.metadata.empty:
            # Create frame timestamp lookup
            frame_lookup = dict(zip(
                session_data.metadata['frameId'], 
                session_data.metadata['timestamp']
            ))
            
            # Find gaze samples with matching frames
            gaze_with_frames = session_data.gaze[
                session_data.gaze['frameId'].isin(frame_lookup.keys())
            ].copy()
            
            if len(gaze_with_frames) > 0:
                # Sample for efficiency
                sample_size = max(100, int(len(gaze_with_frames) * sample_rate))
                sample_size = min(sample_size, len(gaze_with_frames))
                sampled = gaze_with_frames.sample(n=sample_size)
                
                # Calculate time differences
                sampled['frame_timestamp'] = sampled['frameId'].map(frame_lookup)
                sampled['time_diff_ms'] = (
                    sampled['timestamp'] - sampled['frame_timestamp']
                ) / 1e6
                
                metrics['gaze_frame_sync'] = {
                    'mean_diff_ms': float(sampled['time_diff_ms'].mean()),
                    'std_diff_ms': float(sampled['time_diff_ms'].std()),
                    'max_diff_ms': float(sampled['time_diff_ms'].abs().max()),
                    'samples_analyzed': len(sampled)
                }
        
        # Analyze IMU synchronization if available
        if session_data.imu is not None and not session_data.imu.empty:
            # Calculate IMU sampling rate
            if len(session_data.imu) > 1:
                imu_intervals = np.diff(session_data.imu['timestamp'].values) / 1e6
                metrics['imu_sampling'] = {
                    'mean_interval_ms': float(np.mean(imu_intervals)),
                    'std_interval_ms': float(np.std(imu_intervals)),
                    'sampling_rate_hz': float(1000 / np.mean(imu_intervals))
                }
        
        # Add sensor offsets
        metrics['sensor_offsets'] = self.calculate_sensor_offsets(session_data)
        
        # Determine synchronization quality
        if 'gaze_frame_sync' in metrics:
            mean_diff = abs(metrics['gaze_frame_sync']['mean_diff_ms'])
            if mean_diff < 1.0:
                metrics['sync_quality'] = 'Excellent'
            elif mean_diff < 5.0:
                metrics['sync_quality'] = 'Good'
            elif mean_diff < 16.67:
                metrics['sync_quality'] = 'Acceptable'
            else:
                metrics['sync_quality'] = 'Poor'
        
        return metrics