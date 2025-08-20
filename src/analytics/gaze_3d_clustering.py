"""3D Gaze Clustering plugin using HDBSCAN for spatial region identification."""

import numpy as np
import pandas as pd
import hdbscan
import rerun as rr
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.analytics.base_gaze_processor import BaseGazeProcessor
from src.core import SessionData
from src.core.data_types import GazeCluster
from src.core.coordinate_utils import unity_to_rerun_position


class Gaze3DClustering(BaseGazeProcessor):
    """Create spatial clusters from 3D gaze points using HDBSCAN.
    
    This plugin groups 3D gaze points into discrete spatial regions using
    density-based clustering. These clusters persist across frames and
    provide spatial anchors for object instance tracking.
    
    Key features:
    - HDBSCAN clustering for robust density-based grouping
    - Configurable parameters for different scene scales
    - Process ALL gaze points without downsampling
    - Store clusters for use by instance tracking
    """
    
    def __init__(self):
        """Initialize 3D clustering plugin."""
        super().__init__("Gaze3DClustering")
        
        # Storage
        self.gaze_points_3d: List[np.ndarray] = []
        self.timestamps: List[int] = []
        self.clusters: Dict[int, GazeCluster] = {}
        
        # Config will be set in process()
        self.config = {}
        self.min_cluster_size = 30
        self.min_samples = 5
        self.epsilon = 0.15
        self.show_bounds = True
        self.bound_opacity = 0.3
        self.point_opacity = 0.8

    
    def get_dependencies(self) -> List[str]:
        """No hard dependencies - works directly with gaze data."""
        return []
    
    def get_optional_dependencies(self) -> List[str]:
        """Optional dependency on heatmap for validation."""
        return ["Gaze3DHeatmap"]
    
    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process 3D gaze data to create spatial clusters.
        
        Args:
            session: SessionData containing 3D gaze information
            config: Optional configuration dictionary with plugin settings
            
        Returns:
            Dictionary containing:
                - clusters: Dict of cluster ID to GazeCluster objects
                - num_clusters: Number of clusters found
                - noise_ratio: Percentage of points marked as noise
                - largest_cluster_size: Points in largest cluster
        """
        # Get plugin-specific configuration from passed config
        if config and 'plugin_configs' in config:
            plugin_config = config['plugin_configs'].get(self.__class__.__name__, {})
        else:
            plugin_config = {}
        
        # Extract configuration parameters
        self.config = plugin_config
        self.min_cluster_size = plugin_config.get('min_cluster_size', 10)
        self.min_samples = plugin_config.get('min_samples', 5)
        self.epsilon = plugin_config.get('epsilon_m', 0.05)
        
        # New visualization parameters
        self.show_bounds = plugin_config.get('show_bounds', True)
        self.bound_opacity = plugin_config.get('bound_opacity', 0.3)
        self.point_opacity = plugin_config.get('point_opacity', 0.8)
        
        # Extract common filtering configuration
        self.extract_common_config(plugin_config)
        
        self.logger.info(f"Processing with min_cluster_size={self.min_cluster_size}, "
                        f"min_samples={self.min_samples}, epsilon={self.epsilon}m{self.log_filter_info()}")
        
        if session.gaze.empty:
            self.logger.warning("No gaze data available for clustering")
            return {
                'clusters': {},
                'num_clusters': 0,
                'noise_ratio': 0.0,
                'largest_cluster_size': 0
            }
        
        self.logger.info(f"Processing {len(session.gaze)} gaze samples for clustering")
        
        # Extract 3D gaze points using base class method
        try:
            self.gaze_points_3d, self.timestamps = self.extract_and_filter_3d_gaze(session.gaze, include_timestamps=True)
        except Exception as e:
            self.logger.error(f"Error extracting gaze points: {e}")
            return {
                'clusters': {},
                'num_clusters': 0,
                'num_points': 0,
                'num_noise': 0,
                'noise_ratio': 0.0,
                'largest_cluster_size': 0,
                'avg_cluster_quality': 0.0,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'epsilon': self.epsilon
            }
        
        # Validate gaze points for NaN/Inf values
        if self.gaze_points_3d:
            points_array = np.array(self.gaze_points_3d)
            if np.any(~np.isfinite(points_array)):
                self.logger.warning("Found NaN/Inf values in gaze points, filtering...")
                valid_mask = np.all(np.isfinite(points_array), axis=1)
                self.gaze_points_3d = [self.gaze_points_3d[i] for i in range(len(self.gaze_points_3d)) if valid_mask[i]]
                self.timestamps = [self.timestamps[i] for i in range(len(self.timestamps)) if valid_mask[i]]
                self.logger.info(f"Filtered to {len(self.gaze_points_3d)} valid points")
        
        if len(self.gaze_points_3d) < self.min_cluster_size:
            self.logger.warning(f"Not enough points ({len(self.gaze_points_3d)}) "
                               f"for clustering (min: {self.min_cluster_size})")
            return {
                'clusters': {},
                'num_clusters': 0,
                'num_points': len(self.gaze_points_3d),
                'num_noise': len(self.gaze_points_3d),
                'noise_ratio': 100.0,
                'largest_cluster_size': 0,
                'avg_cluster_quality': 0.0,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'epsilon': self.epsilon
            }
        
        # Perform HDBSCAN clustering
        try:
            labels = self._perform_clustering()
        except Exception as e:
            self.logger.error(f"HDBSCAN clustering failed: {e}")
            return {
                'clusters': {},
                'num_clusters': 0,
                'num_points': len(self.gaze_points_3d),
                'num_noise': len(self.gaze_points_3d),
                'noise_ratio': 100.0,
                'largest_cluster_size': 0,
                'avg_cluster_quality': 0.0,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'epsilon': self.epsilon
            }
        
        # Create GazeCluster objects
        try:
            self._create_cluster_objects(labels)
        except Exception as e:
            self.logger.error(f"Error creating cluster objects: {e}")
            # Continue with empty clusters rather than failing completely
            self.clusters = {}
        
        # Calculate metrics
        metrics = self._calculate_clustering_metrics(labels)
        
        # Store clusters in session
        session.gaze_clusters = self.clusters
        session.set_plugin_result('Gaze3DClustering', metrics)
        
        # Generate reports
        self._generate_reports(metrics, session)
        
        self.logger.info(f"Found {metrics['num_clusters']} clusters, "
                        f"noise ratio: {metrics['noise_ratio']:.1f}%")
        
        return metrics
    
    
    def _perform_clustering(self) -> np.ndarray:
        """Perform HDBSCAN clustering on 3D gaze points.
        
        Returns:
            Array of cluster labels (-1 for noise)
        """
        # Convert to numpy array for HDBSCAN
        points_array = np.array(self.gaze_points_3d)
        
        self.logger.info(f"Running HDBSCAN on {len(points_array)} points...")
        
        # Create and fit HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.epsilon,
            cluster_selection_method='eom',  # Excess of Mass
            metric='euclidean',
            algorithm='best',
            prediction_data=False  # Don't need soft clustering
        )
        
        labels = clusterer.fit_predict(points_array)
        
        # Log clustering results
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        self.logger.info(f"HDBSCAN found {n_clusters} clusters, "
                        f"{n_noise} noise points")
        
        return labels
    
    def _create_cluster_objects(self, labels: np.ndarray) -> None:
        """Create GazeCluster objects from clustering results.
        
        Args:
            labels: Array of cluster labels from HDBSCAN
        """
        self.clusters.clear()
        points_array = np.array(self.gaze_points_3d)
        timestamps_array = np.array(self.timestamps)
        
        # Get unique cluster IDs (excluding noise)
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for cluster_id in unique_labels:
            # Get points belonging to this cluster
            mask = labels == cluster_id
            cluster_points = points_array[mask]
            cluster_timestamps = timestamps_array[mask]
            
            # Calculate centroid
            centroid = np.mean(cluster_points, axis=0)
            
            # Calculate radius (distance to furthest point)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            radius = float(np.max(distances))
            
            # Calculate density (points per cubic meter)
            # Approximate volume as sphere
            volume = (4/3) * np.pi * (radius ** 3) if radius > 0 else 1.0
            density = len(cluster_points) / volume
            
            # Quality score based on normalized size and density
            # Normalize size: expect clusters of 30-500 points
            size_score = min(1.0, len(cluster_points) / 500)
            # Normalize density: expect 50-500 points/m³
            density_score = min(1.0, density / 500)
            # Combined quality score
            quality_score = (size_score * 0.6) + (density_score * 0.4)
            
            # Create GazeCluster object
            cluster = GazeCluster(
                cluster_id=int(cluster_id),
                points_3d=cluster_points,
                timestamps=cluster_timestamps,
                centroid=centroid,
                radius=radius,
                quality_score=quality_score,
                point_count=len(cluster_points),
                density=density,
                start_timestamp=int(cluster_timestamps.min()),
                end_timestamp=int(cluster_timestamps.max()),
                duration_ms=float((cluster_timestamps.max() - cluster_timestamps.min()) / 1e6)
            )
            
            self.clusters[int(cluster_id)] = cluster
            
            self.logger.debug(f"Cluster {cluster_id}: {len(cluster_points)} points, "
                            f"radius={radius:.2f}m, quality={quality_score:.2f}")
    
    def _calculate_clustering_metrics(self, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate clustering metrics and statistics.
        
        Args:
            labels: Cluster labels from HDBSCAN
            
        Returns:
            Dictionary with clustering metrics
        """
        # Count clusters and noise
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_noise = list(labels).count(-1)
        noise_ratio = (num_noise / len(labels)) * 100 if len(labels) > 0 else 0
        
        # Find largest cluster
        largest_cluster_size = 0
        if num_clusters > 0:
            for cluster_id in unique_labels:
                if cluster_id != -1:
                    cluster_size = list(labels).count(cluster_id)
                    largest_cluster_size = max(largest_cluster_size, cluster_size)
        
        # Calculate average cluster quality
        avg_quality = 0.0
        if self.clusters:
            avg_quality = np.mean([c.quality_score for c in self.clusters.values()])
        
        return {
            'clusters': self.clusters,
            'num_clusters': num_clusters,
            'num_points': len(labels),
            'num_noise': num_noise,
            'noise_ratio': noise_ratio,
            'largest_cluster_size': largest_cluster_size,
            'avg_cluster_quality': avg_quality,
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'epsilon': self.epsilon
        }
    
    def visualize(self, results: Dict, rr_stream=None) -> None:
        """Visualize clusters in Rerun.
        
        Args:
            results: Dictionary containing cluster results
            rr_stream: Optional Rerun stream to use
        """
        clusters = results.get('clusters', {})
        if not clusters:
            self.logger.warning("No clusters to visualize")
            return
        
        self.logger.info(f"Visualizing {len(clusters)} clusters in Rerun")
        
        # Colors for different clusters (cycle through if many clusters)
        cluster_colors = [
            [255, 100, 100],  # Red
            [100, 255, 100],  # Green
            [100, 100, 255],  # Blue
            [255, 255, 100],  # Yellow
            [255, 100, 255],  # Magenta
            [100, 255, 255],  # Cyan
            [255, 165, 0],    # Orange
            [128, 0, 128],    # Purple
        ]
        
        for cluster_id, cluster in clusters.items():
            # Select color for this cluster
            color = cluster_colors[cluster_id % len(cluster_colors)]
            
            # Add alpha channel for points opacity
            point_alpha = int(self.point_opacity * 255)
            color_with_alpha = [color[0], color[1], color[2], point_alpha]
            
            # Log cluster points with RGBA color
            rr.log(
                f"/world/spatial/clusters/{cluster_id}/points",
                rr.Points3D(
                    positions=cluster.points_3d.tolist(),
                    colors=[color_with_alpha],  # Now includes alpha
                    radii=0.003  # Small points
                ),
                static=False
            )
            
            # Add bounding box if enabled
            if self.show_bounds:
                # Calculate bounding box dimensions
                points = cluster.points_3d
                min_bounds = np.min(points, axis=0)
                max_bounds = np.max(points, axis=0)
                center = (min_bounds + max_bounds) / 2
                half_sizes = (max_bounds - min_bounds) / 2
                
                # Add alpha channel for bounding box opacity
                bound_alpha = int(self.bound_opacity * 255)
                bound_color_with_alpha = [color[0], color[1], color[2], bound_alpha]
                
                # Log bounding box as wireframe with RGBA color
                rr.log(
                    f"/world/spatial/clusters/{cluster_id}/bounds",
                    rr.Boxes3D(
                        centers=[center.tolist()],
                        half_sizes=[half_sizes.tolist()],
                        colors=[bound_color_with_alpha],  # Now includes alpha
                        fill_mode="wireframe"
                    ),
                    static=True
                )
            
            # Log cluster centroid (larger, brighter)
            bright_color = [min(255, c + 50) for c in color]
            rr.log(
                f"/world/spatial/clusters/{cluster_id}/centroid",
                rr.Points3D(
                    positions=[cluster.centroid.tolist()],
                    colors=[bright_color],
                    radii=0.02
                ),
                static=True
            )
            
            # Log cluster info as text
            info_text = (f"Cluster {cluster_id}\n"
                        f"Points: {cluster.point_count}\n"
                        f"Radius: {cluster.radius:.2f}m\n"
                        f"Quality: {cluster.quality_score:.2f}")
            
            rr.log(
                f"/world/spatial/clusters/{cluster_id}/info",
                rr.TextDocument(info_text),
                static=False
            )
        
        # Log overall cluster visualization
        all_centroids = [c.centroid.tolist() for c in clusters.values()]
        if all_centroids:
            rr.log(
                "/world/spatial/clusters/all",
                rr.Points3D(
                    positions=all_centroids,
                    colors=[[255, 255, 255]],  # White for all centroids
                    radii=0.015
                ),
                static=False
            )
    
    def _generate_reports(self, metrics: Dict[str, Any], session: SessionData) -> None:
        """Generate CSV reports for clustering results.
        
        Args:
            metrics: Clustering metrics
            session: SessionData for session info
        """
        if not metrics.get('clusters'):
            return
        
        # Create report directory
        report_dir = Path(f"reports/{session.session_id}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cluster data DataFrame
        cluster_data = []
        for cluster_id, cluster in metrics['clusters'].items():
            cluster_data.append({
                'cluster_id': cluster_id,
                'centroid_x': cluster.centroid[0],
                'centroid_y': cluster.centroid[1],
                'centroid_z': cluster.centroid[2],
                'point_count': cluster.point_count,
                'radius_m': cluster.radius,
                'density': cluster.density,
                'quality_score': cluster.quality_score,
                'duration_ms': cluster.duration_ms,
                'start_timestamp': cluster.start_timestamp,
                'end_timestamp': cluster.end_timestamp
            })
        
        cluster_df = pd.DataFrame(cluster_data)
        cluster_df = cluster_df.sort_values('point_count', ascending=False)
        
        # Save cluster report
        cluster_path = report_dir / "spatial_clusters.csv"
        cluster_df.to_csv(cluster_path, index=False)
        
        # Create summary report
        summary_data = {
            'metric': ['num_clusters', 'total_points', 'noise_points', 
                      'noise_ratio_pct', 'largest_cluster_size', 
                      'avg_cluster_quality', 'min_cluster_size_param',
                      'min_samples_param', 'epsilon_m_param'],
            'value': [
                metrics['num_clusters'],
                metrics['num_points'],
                metrics['num_noise'],
                metrics['noise_ratio'],
                metrics['largest_cluster_size'],
                metrics['avg_cluster_quality'],
                metrics['min_cluster_size'],
                metrics['min_samples'],
                metrics['epsilon']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = report_dir / "spatial_clustering_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        self.logger.info(f"Saved clustering reports to {report_dir}")
        
        # Console summary
        self.logger.info("="*60)
        self.logger.info("3D GAZE CLUSTERING SUMMARY")
        self.logger.info(f"Clusters found: {metrics['num_clusters']}")
        self.logger.info(f"Total points: {metrics['num_points']}")
        self.logger.info(f"Noise points: {metrics['num_noise']} ({metrics['noise_ratio']:.1f}%)")
        self.logger.info(f"Largest cluster: {metrics['largest_cluster_size']} points")
        self.logger.info(f"Average quality: {metrics['avg_cluster_quality']:.2f}")
        self.logger.info(f"Parameters: min_size={metrics['min_cluster_size']}, "
                        f"min_samples={metrics['min_samples']}, epsilon={metrics['epsilon']}m")
        
        if len(cluster_df) > 0:
            self.logger.info("Top 5 Clusters by Size:")
            for idx, row in cluster_df.head(5).iterrows():
                self.logger.info(f"  Cluster {row['cluster_id']}: {row['point_count']} points, "
                                f"radius={row['radius_m']:.2f}m, quality={row['quality_score']:.2f}")
        self.logger.info("="*60)