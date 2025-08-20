"""Base class for gaze processing plugins with shared functionality."""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from abc import abstractmethod

from src.plugin_sys.base import AnalyticsPlugin
from src.core.coordinate_utils import unity_to_rerun_position


class BaseGazeProcessor(AnalyticsPlugin):
    """Base class for analytics plugins that process 3D gaze data.
    
    Provides common functionality for:
    - Gaze data validation and extraction
    - Gaze state filtering
    - Hit type filtering
    - Coordinate conversion
    """
    
    def __init__(self, name: str):
        """Initialize base gaze processor.
        
        Args:
            name: Name of the plugin
        """
        super().__init__(name)
        
        # Common filtering parameters
        self.filter_enabled = False
        self.gaze_states_filter: List[str] = []
        self.hit_type_filter: List[str] = []
    
    def extract_common_config(self, plugin_config: dict) -> None:
        """Extract common configuration parameters.
        
        Args:
            plugin_config: Plugin-specific configuration dictionary
        """
        self.filter_enabled = plugin_config.get('filter_enabled', False)
        self.gaze_states_filter = plugin_config.get('gaze_states_filter', [])
        self.hit_type_filter = plugin_config.get('hit_type_filter', [])
    
    def extract_and_filter_3d_gaze(self, gaze_df: pd.DataFrame, 
                                   include_timestamps: bool = False) -> Tuple[List[np.ndarray], Optional[List[int]]]:
        """Extract and filter 3D gaze points with common validation and filtering.
        
        Args:
            gaze_df: DataFrame with gaze data
            include_timestamps: Whether to also return timestamps
            
        Returns:
            Tuple of (3D positions in Rerun coordinates, timestamps if requested)
        """
        # Check required columns exist
        required_cols = ['isTracking', 'hasHitTarget', 'gazePositionX', 
                        'gazePositionY', 'gazePositionZ']
        if include_timestamps:
            required_cols.append('timestamp')
            
        missing_cols = [col for col in required_cols if col not in gaze_df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return [], [] if include_timestamps else []
        
        # Filter for valid hit points
        valid_gaze = gaze_df[
            (gaze_df['isTracking'] == True) & 
            (gaze_df['hasHitTarget'] == True)
        ]
        
        # Apply gaze state filter if enabled
        if self.filter_enabled and self.gaze_states_filter:
            if 'gazeState' in valid_gaze.columns:
                valid_gaze = valid_gaze[valid_gaze['gazeState'].isin(self.gaze_states_filter)]
                self.logger.info(f"Filtering for gaze states: {self.gaze_states_filter}")
            else:
                self.logger.warning("gazeState column not found, skipping state filtering")
        
        # Apply hit type filter if specified
        if self.hit_type_filter:
            if 'gazeHitType' in valid_gaze.columns:
                valid_gaze = valid_gaze[valid_gaze['gazeHitType'].isin(self.hit_type_filter)]
                self.logger.info(f"Filtering for hit types: {self.hit_type_filter}")
            else:
                self.logger.debug("gazeHitType column not found, skipping hit type filter")
        
        if len(valid_gaze) == 0:
            self.logger.warning("No valid gaze hit points found after filtering")
            return [], [] if include_timestamps else []
        
        self.logger.info(f"Found {len(valid_gaze)} valid gaze hit points")
        
        # Extract positions and convert to Rerun coordinates
        positions = valid_gaze[['gazePositionX', 'gazePositionY', 'gazePositionZ']].values
        gaze_points_3d = []
        
        for unity_pos in positions:
            rerun_pos = unity_to_rerun_position(unity_pos.tolist())
            gaze_points_3d.append(np.array(rerun_pos))
        
        if include_timestamps:
            timestamps = valid_gaze['timestamp'].values.astype(np.int64).tolist()
            return gaze_points_3d, timestamps
        
        return gaze_points_3d, None
    
    def log_filter_info(self) -> str:
        """Get filter information string for logging.
        
        Returns:
            String describing active filters
        """
        filter_parts = []
        if self.filter_enabled and self.gaze_states_filter:
            filter_parts.append(f"states={self.gaze_states_filter}")
        if self.hit_type_filter:
            filter_parts.append(f"hit_types={self.hit_type_filter}")
        
        return f", filters: {', '.join(filter_parts)}" if filter_parts else ""
    
    @abstractmethod
    def process(self, session, config=None):
        """Process method to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def visualize(self, results, rr_stream=None):
        """Visualization method to be implemented by subclasses."""
        pass