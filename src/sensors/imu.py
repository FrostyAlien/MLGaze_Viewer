"""IMU sensor handler for MLGaze Viewer."""

import rerun as rr
import numpy as np
from typing import Dict, Any, List
from src.sensors.base import BaseSensor
from src.core import SessionData


class IMUSensor(BaseSensor):
    """Handler for IMU (Inertial Measurement Unit) sensor data."""
    
    def __init__(self, entity_path: str = "/sensors/imu"):
        """Initialize IMU sensor.
        
        Args:
            entity_path: Base entity path for IMU data in Rerun
        """
        super().__init__(entity_path, "IMU Sensor")
        
        # Define axis colors for consistent visualization
        self.axis_colors = {
            'X': [255, 80, 80],   # Bright Red
            'Y': [80, 200, 80],   # Bright Green
            'Z': [80, 120, 255]   # Bright Blue
        }
    
    def log_to_rerun(self, session: SessionData, config: Dict[str, Any]) -> None:
        """Log IMU data to Rerun as time series scalars.
        
        Args:
            session: SessionData containing IMU information
            config: Visualization configuration
        """
        if not config.get('show_imu_data', True):
            print("IMU visualization disabled in config")
            return
        
        if session.imu is None or session.imu.empty:
            print("No IMU data available")
            return
        
        print(f"Logging {len(session.imu)} IMU samples using columnar method...")
        
        # Use efficient columnar logging for time series data
        self._log_accelerometer_data(session.imu)
        self._log_gyroscope_data(session.imu)
        
        print(f"  IMU data logged successfully")
    
    def _log_accelerometer_data(self, imu_df) -> None:
        """Log accelerometer data as time series.
        
        Args:
            imu_df: DataFrame containing IMU data
        """
        # Create time column (convert nanoseconds to seconds)
        times = rr.TimeColumn("timestamp", timestamp=imu_df["timestamp"].values * 1e-9)
        
        # Log each axis separately for proper visualization
        for axis in ['X', 'Y', 'Z']:
            column_name = f"accel{axis}"
            if column_name in imu_df.columns:
                entity_path = f"/sensors/accelerometer/{axis}"
                
                # Send columnar data
                rr.send_columns(
                    entity_path,
                    indexes=[times],
                    columns=rr.Scalars.columns(scalars=imu_df[column_name].values)
                )
                
                # Apply styling
                self._apply_axis_styling(entity_path, axis)
        
        print(f"  Accelerometer: {len(imu_df)} samples to /sensors/accelerometer/[X,Y,Z]")
    
    def _log_gyroscope_data(self, imu_df) -> None:
        """Log gyroscope data as time series.
        
        Args:
            imu_df: DataFrame containing IMU data
        """
        # Create time column (convert nanoseconds to seconds)
        times = rr.TimeColumn("timestamp", timestamp=imu_df["timestamp"].values * 1e-9)
        
        # Log each axis separately
        for axis in ['X', 'Y', 'Z']:
            column_name = f"gyro{axis}"
            if column_name in imu_df.columns:
                entity_path = f"/sensors/gyroscope/{axis}"
                
                # Send columnar data
                rr.send_columns(
                    entity_path,
                    indexes=[times],
                    columns=rr.Scalars.columns(scalars=imu_df[column_name].values)
                )
                
                # Apply styling
                self._apply_axis_styling(entity_path, axis)
        
        print(f"  Gyroscope: {len(imu_df)} samples to /sensors/gyroscope/[X,Y,Z]")
    
    def _apply_axis_styling(self, entity_path: str, axis: str) -> None:
        """Apply SeriesLines styling to IMU axis with proper color.
        
        Args:
            entity_path: Rerun entity path for the data
            axis: Axis name (X, Y, Z)
        """
        try:
            color = self.axis_colors.get(axis, [128, 128, 128])
            rr.log(
                entity_path,
                rr.SeriesLines(colors=color, names=axis),
                static=True
            )
        except Exception as e:
            # SeriesLines might not be available in all Rerun versions
            pass
    
    def get_imu_statistics(self, session: SessionData) -> Dict[str, Any]:
        """Calculate IMU sensor statistics.
        
        Args:
            session: SessionData containing IMU information
            
        Returns:
            Dictionary with IMU statistics
        """
        if session.imu is None or session.imu.empty:
            return {}
        
        stats = {
            'num_samples': len(session.imu),
            'sampling_rate_hz': session.imu_sampling_rate,
        }
        
        # Accelerometer statistics
        for axis in ['X', 'Y', 'Z']:
            col = f"accel{axis}"
            if col in session.imu.columns:
                stats[f'accel_{axis.lower()}_mean'] = float(session.imu[col].mean())
                stats[f'accel_{axis.lower()}_std'] = float(session.imu[col].std())
                stats[f'accel_{axis.lower()}_range'] = [
                    float(session.imu[col].min()),
                    float(session.imu[col].max())
                ]
        
        # Gyroscope statistics  
        for axis in ['X', 'Y', 'Z']:
            col = f"gyro{axis}"
            if col in session.imu.columns:
                stats[f'gyro_{axis.lower()}_mean'] = float(session.imu[col].mean())
                stats[f'gyro_{axis.lower()}_std'] = float(session.imu[col].std())
                stats[f'gyro_{axis.lower()}_range'] = [
                    float(session.imu[col].min()),
                    float(session.imu[col].max())
                ]
        
        return stats