"""3D Gaze Heatmap plugin for spatial density visualization."""

import numpy as np
import pandas as pd
import rerun as rr
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from src.plugin_sys.base import AnalyticsPlugin
from src.core import SessionData
from src.core.coordinate_utils import unity_to_rerun_position


@dataclass
class VoxelData:
    """Data for a single voxel in the heatmap grid."""
    x: int
    y: int
    z: int
    count: int
    center_position: np.ndarray
    

class Gaze3DHeatmap(AnalyticsPlugin):
    """Generate 3D heatmap visualization of gaze density using voxel grid.
    
    This plugin creates a voxel-based density visualization showing WHERE
    in 3D space users focus their attention. It divides 3D space into
    configurable voxels and counts gaze points within each voxel.
    
    Key features:
    - Simple voxel counting algorithm (no complex KDE)
    - Configurable voxel size and minimum density threshold
    - Turbo colormap for intuitive heat visualization
    - Toggleable visualization in Rerun
    """
    
    def __init__(self):
        """Initialize 3D heatmap plugin."""
        super().__init__("Gaze3DHeatmap")
        
        # Storage for voxel grid
        self.voxel_grid: Dict[Tuple[int, int, int], VoxelData] = {}
        self.gaze_points_3d: List[np.ndarray] = []
        
        # Config will be set in process()
        self.config = {}
        self.voxel_size = 0.1  # Default 10cm
        self.min_density = 5  # Default min 5 points to show
        self.show_heatmap = True  # Default show heatmap
        self.use_boxes = None
        self.opacity_min = 0.3  # Default min opacity
        self.opacity_max = 1.0  # Default max opacity
        self.fill_mode = 'solid'  # Default fill mode
        self.filter_enabled = False  # Default filtering off
        self.gaze_states_filter: List[str] = []  # Default no filter
    
    def get_dependencies(self) -> List[str]:
        """No dependencies - works directly with gaze data."""
        return []
    
    def get_optional_dependencies(self) -> List[str]:
        """No optional dependencies."""
        return []
    
    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process 3D gaze data to create voxel-based heatmap.
        
        Args:
            session: SessionData containing 3D gaze information
            config: Optional configuration dictionary with plugin settings
            
        Returns:
            Dictionary containing:
                - voxel_grid: Dict of voxel coordinates to counts
                - total_voxels: Number of voxels with data
                - max_density: Maximum points in any voxel
                - hotspot_voxels: Voxels above 90th percentile
                - coverage_volume: Approximate volume covered
        """
        # Get plugin-specific configuration from passed config
        if config and 'plugin_configs' in config:
            plugin_config = config['plugin_configs'].get(self.__class__.__name__, {})
        else:
            plugin_config = {}
        
        # Extract configuration parameters
        self.config = plugin_config
        self.voxel_size = plugin_config.get('voxel_size', 0.1)  # 10cm default
        self.min_density = plugin_config.get('min_density', 5)  # Min 5 points to show
        self.show_heatmap = plugin_config.get('show_heatmap', True)
        
        # New visualization parameters with defaults
        self.use_boxes = plugin_config.get('use_boxes', True)
        self.opacity_min = plugin_config.get('opacity_min', 0.3)
        self.opacity_max = plugin_config.get('opacity_max', 1.0)
        self.fill_mode = plugin_config.get('fill_mode', 'solid')
        
        # Gaze state filtering parameters
        self.filter_enabled = plugin_config.get('filter_enabled', False)
        self.gaze_states_filter = plugin_config.get('gaze_states_filter', [])
        
        filter_info = f", filter_states={self.gaze_states_filter}" if self.filter_enabled else ""
        self.logger.info(f"Processing with voxel_size={self.voxel_size}m, "
                        f"min_density={self.min_density}, use_boxes={self.use_boxes}{filter_info}")
        
        if session.gaze.empty:
            self.logger.warning("No gaze data available for heatmap generation")
            return {
                'voxel_grid': {},
                'total_voxels': 0,
                'max_density': 0,
                'hotspot_voxels': [],
                'coverage_volume': 0.0
            }
        
        self.logger.info(f"Processing {len(session.gaze)} gaze samples for 3D heatmap")
        
        # Extract 3D gaze hit points
        self._extract_3d_gaze_points(session.gaze)
        
        # Create voxel grid
        self._create_voxel_grid()
        
        # Calculate metrics
        metrics = self._calculate_heatmap_metrics()
        
        # Store results in session
        session.set_plugin_result('Gaze3DHeatmap', metrics)
        
        # Generate reports
        self._generate_reports(metrics, session)
        
        self.logger.info(f"Created heatmap with {metrics['total_voxels']} voxels, "
                        f"max density: {metrics['max_density']}")
        
        return metrics
    
    def _extract_3d_gaze_points(self, gaze_df: pd.DataFrame) -> None:
        """Extract 3D gaze hit points from dataframe.
        
        Args:
            gaze_df: DataFrame with gaze data including position columns
        """
        self.gaze_points_3d.clear()
        
        # Check required columns exist
        required_cols = ['isTracking', 'hasHitTarget', 'gazePositionX', 'gazePositionY', 'gazePositionZ']
        missing_cols = [col for col in required_cols if col not in gaze_df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return
        
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
        
        if len(valid_gaze) == 0:
            self.logger.warning("No valid gaze hit points found")
            return
        
        self.logger.info(f"Found {len(valid_gaze)} valid gaze hit points")
        
        # Vectorized extraction and conversion
        positions = valid_gaze[['gazePositionX', 'gazePositionY', 'gazePositionZ']].values
        
        # Convert all positions from Unity to Rerun coordinates
        for unity_pos in positions:
            rerun_pos = unity_to_rerun_position(unity_pos.tolist())
            self.gaze_points_3d.append(np.array(rerun_pos))
    
    def _create_voxel_grid(self) -> None:
        """Create voxel grid by counting points in each voxel."""
        self.voxel_grid.clear()
        
        for point in self.gaze_points_3d:
            # Quantize to voxel coordinates
            voxel_key = (
                int(point[0] / self.voxel_size),
                int(point[1] / self.voxel_size),
                int(point[2] / self.voxel_size)
            )
            
            # Update or create voxel
            if voxel_key in self.voxel_grid:
                self.voxel_grid[voxel_key].count += 1
            else:
                # Calculate voxel center position
                center = np.array([
                    (voxel_key[0] + 0.5) * self.voxel_size,
                    (voxel_key[1] + 0.5) * self.voxel_size,
                    (voxel_key[2] + 0.5) * self.voxel_size
                ])
                self.voxel_grid[voxel_key] = VoxelData(
                    x=voxel_key[0],
                    y=voxel_key[1],
                    z=voxel_key[2],
                    count=1,
                    center_position=center
                )
    
    def _calculate_heatmap_metrics(self) -> Dict[str, Any]:
        """Calculate heatmap metrics and statistics.
        
        Returns:
            Dictionary with heatmap metrics
        """
        if not self.voxel_grid:
            return {
                'voxel_grid': {},
                'total_voxels': 0,
                'max_density': 0,
                'hotspot_voxels': [],
                'coverage_volume': 0.0,
                'total_points': len(self.gaze_points_3d),
                'voxel_size': self.voxel_size,
                'min_density': self.min_density
            }
        
        # Filter voxels by minimum density
        filtered_voxels = {
            k: v for k, v in self.voxel_grid.items() 
            if v.count >= self.min_density
        }
        
        # Calculate statistics
        densities = [v.count for v in filtered_voxels.values()]
        max_density = max(densities) if densities else 0
        
        # Find hotspots (90th percentile)
        if densities:
            threshold = np.percentile(densities, 90)
            hotspot_voxels = [
                k for k, v in filtered_voxels.items() 
                if v.count >= threshold
            ]
        else:
            hotspot_voxels = []
        
        # Estimate coverage volume
        coverage_volume = len(filtered_voxels) * (self.voxel_size ** 3)
        
        return {
            'voxel_grid': filtered_voxels,
            'total_voxels': len(filtered_voxels),
            'max_density': max_density,
            'hotspot_voxels': hotspot_voxels,
            'coverage_volume': coverage_volume,
            'total_points': len(self.gaze_points_3d),
            'voxel_size': self.voxel_size,
            'min_density': self.min_density
        }
    
    def visualize(self, results: Dict, rr_stream=None) -> None:
        """Visualize heatmap in Rerun using Boxes3D or Points3D.
        
        Args:
            results: Dictionary containing processed heatmap data
            rr_stream: Optional Rerun stream to use
        """
        if not self.show_heatmap:
            self.logger.info("Heatmap visualization disabled by configuration")
            return
        
        # Get voxel grid from results
        if not results or not results.get('voxel_grid'):
            self.logger.warning("No heatmap data to visualize")
            return
        
        voxel_grid = results['voxel_grid']
        max_density = results['max_density']
        
        self.logger.info(f"Visualizing {len(voxel_grid)} voxels in Rerun")
        
        # Prepare data for visualization
        positions = []
        colors = []
        radii = []
        half_sizes = []  # For Boxes3D
        
        for voxel_data in voxel_grid.values():
            positions.append(voxel_data.center_position)
            
            # Map density to color using turbo colormap
            normalized_density = voxel_data.count / max_density if max_density > 0 else 0
            color = self._density_to_turbo_color(normalized_density)
            
            # Logarithmic opacity scaling for better visual range
            if max_density > 0:
                log_density = np.log1p(voxel_data.count)  # log(1 + count)
                log_max = np.log1p(max_density)
                opacity = self.opacity_min + (self.opacity_max - self.opacity_min) * (log_density / log_max)
            else:
                opacity = self.opacity_min
            
            # Add alpha channel to color for transparency
            alpha = int(opacity * 255)  # Convert 0-1 to 0-255
            color_with_alpha = [color[0], color[1], color[2], alpha]
            colors.append(color_with_alpha)
            
            # For Points3D fallback
            radius = 0.005 + (normalized_density * 0.01)  # 5-15mm
            radii.append(radius)
            
            # For Boxes3D
            half_size = [self.voxel_size / 2, self.voxel_size / 2, self.voxel_size / 2]
            half_sizes.append(half_size)
        
        # Use Boxes3D or Points3D based on configuration
        if self.use_boxes:
            # Log voxels as boxes for better visualization
            rr.log(
                "/world/spatial/heatmap/voxels",
                rr.Boxes3D(
                    centers=positions,
                    half_sizes=half_sizes,
                    colors=colors,  # Now includes alpha channel
                    fill_mode=self.fill_mode
                ),
                static=True  # Static for performance
            )
        else:
            # Fallback to Points3D
            rr.log(
                "/world/spatial/heatmap/density",
                rr.Points3D(
                    positions=positions,
                    colors=colors,
                    radii=radii
                ),
                static=True  # Static for performance
            )
        
        # Log hotspots separately for emphasis
        hotspot_positions = []
        for voxel_key in results['hotspot_voxels']:
            if voxel_key in voxel_grid:
                hotspot_positions.append(voxel_grid[voxel_key].center_position)
        
        if hotspot_positions:
            rr.log(
                "/world/spatial/heatmap/hotspots",
                rr.Points3D(
                    positions=hotspot_positions,
                    colors=[[255, 0, 0]],  # Red for hotspots
                    radii=0.02  # Larger for visibility
                ),
                static=False
            )
        
        self.logger.info(f"Logged heatmap with {len(hotspot_positions)} hotspots")
    
    def _density_to_turbo_color(self, normalized_value: float) -> List[int]:
        """Convert normalized density (0-1) to turbo colormap RGB.
        
        Turbo colormap: Blue (cold) -> Cyan -> Green -> Yellow -> Red (hot)
        
        Args:
            normalized_value: Value between 0 and 1
            
        Returns:
            RGB color as [R, G, B] with values 0-255
        """
        # Clamp value
        value = max(0.0, min(1.0, normalized_value))
        
        # Simple turbo approximation
        if value < 0.25:
            # Blue to Cyan
            r = 0
            g = int(value * 4 * 255)
            b = 255
        elif value < 0.5:
            # Cyan to Green
            r = 0
            g = 255
            b = int((1 - (value - 0.25) * 4) * 255)
        elif value < 0.75:
            # Green to Yellow
            r = int((value - 0.5) * 4 * 255)
            g = 255
            b = 0
        else:
            # Yellow to Red
            r = 255
            g = int((1 - (value - 0.75) * 4) * 255)
            b = 0
        
        return [r, g, b]
    
    def _generate_reports(self, metrics: Dict[str, Any], session: SessionData) -> None:
        """Generate CSV report for heatmap data.
        
        Args:
            metrics: Heatmap metrics dictionary
            session: SessionData for session info
        """
        if not metrics.get('voxel_grid'):
            return
        
        # Create report directory
        report_dir = Path(f"reports/{session.session_id}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create voxel data DataFrame
        voxel_data = []
        for voxel_key, voxel in metrics['voxel_grid'].items():
            voxel_data.append({
                'voxel_x': voxel.x,
                'voxel_y': voxel.y,
                'voxel_z': voxel.z,
                'center_x': voxel.center_position[0],
                'center_y': voxel.center_position[1],
                'center_z': voxel.center_position[2],
                'density': voxel.count,
                'is_hotspot': voxel_key in metrics['hotspot_voxels']
            })
        
        voxel_df = pd.DataFrame(voxel_data)
        voxel_df = voxel_df.sort_values('density', ascending=False)
        
        # Save voxel report
        voxel_path = report_dir / "spatial_heatmap_voxels.csv"
        voxel_df.to_csv(voxel_path, index=False)
        
        # Create summary report
        summary_data = {
            'metric': ['total_voxels', 'max_density', 'hotspot_count', 
                      'coverage_volume_m3', 'total_gaze_points', 
                      'voxel_size_m', 'min_density_threshold'],
            'value': [
                metrics['total_voxels'],
                metrics['max_density'],
                len(metrics['hotspot_voxels']),
                metrics['coverage_volume'],
                metrics['total_points'],
                metrics['voxel_size'],
                metrics['min_density']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = report_dir / "spatial_heatmap_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        self.logger.info(f"Saved heatmap reports to {report_dir}")
        
        # Console summary
        self.logger.info("="*60)
        self.logger.info("3D GAZE HEATMAP SUMMARY")
        self.logger.info(f"Total voxels with activity: {metrics['total_voxels']}")
        self.logger.info(f"Maximum density: {metrics['max_density']} points/voxel")
        self.logger.info(f"Hotspot regions: {len(metrics['hotspot_voxels'])}")
        self.logger.info(f"Coverage volume: {metrics['coverage_volume']:.2f} m³")
        self.logger.info(f"Configuration: voxel_size={metrics['voxel_size']}m, "
                        f"min_density={metrics['min_density']}")
        
        if len(voxel_df) > 0:
            self.logger.info("Top 5 Densest Voxels:")
            for idx, row in voxel_df.head(5).iterrows():
                self.logger.info(f"  Position: ({row['center_x']:.2f}, {row['center_y']:.2f}, "
                                f"{row['center_z']:.2f}) - Density: {row['density']}")
        self.logger.info("="*60)