"""Object instance tracking plugin that associates detected objects across frames and with gaze clusters."""

import numpy as np
import pandas as pd
import rerun as rr
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

from src.plugin_sys.base import AnalyticsPlugin
from src.core import SessionData
from src.core.data_types import DetectedObject, GazeCluster, SessionMetrics
from src.core.coordinate_utils import unity_to_rerun_position


@dataclass
class TrackedInstance:
    """Represents a tracked object instance across multiple frames."""
    instance_id: int
    class_name: str
    first_seen_timestamp: int
    last_seen_timestamp: int
    frame_count: int = 0
    detections: List[DetectedObject] = field(default_factory=list)
    associated_clusters: List[int] = field(default_factory=list)  # GazeCluster IDs
    confidence_scores: List[float] = field(default_factory=list)
    bbox_3d: Optional[Dict] = None  # 3D bounding box if sufficient gaze data
    gaze_quality: float = 0.0  # Average quality of associated gaze clusters
    gaze_point_count: int = 0  # Number of gaze points used for 3D bbox
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence score across all detections."""
        return np.mean(self.confidence_scores) if self.confidence_scores else 0.0
    
    @property
    def duration_ms(self) -> float:
        """Duration this instance was tracked in milliseconds."""
        return (self.last_seen_timestamp - self.first_seen_timestamp) / 1e6
    
    @property
    def is_active(self) -> bool:
        """Check if instance is still active (seen recently).
        
        Note: This is currently unused as we track based on frame gaps.
        Could be used for real-time tracking scenarios.
        """
        # This would need session context to work properly
        # For now, tracking is handled by frame gap logic
        return True


class ObjectInstanceTracker(AnalyticsPlugin):
    """Track object instances across frames and associate with gaze clusters.
    
    This plugin:
    1. Associates detected objects across frames to create persistent instances
    2. Links instances with nearby gaze clusters for spatial context
    3. Tracks instance lifecycle (appearance, persistence, disappearance)
    4. Provides metrics on instance-gaze relationships
    """
    
    def __init__(self):
        """Initialize the ObjectInstanceTracker."""
        super().__init__("Object Instance Tracker")
        self.tracked_instances: Dict[int, TrackedInstance] = {}
        self.next_instance_id = 1
        self.class_instance_counters = defaultdict(int)  # Track per-class instance counts
    
    def get_dependencies(self) -> List[str]:
        """Return required dependencies.
        
        Returns:
            List of required plugin class names
        """
        return ["ObjectDetector"]  # Only ObjectDetector is required
    
    def get_optional_dependencies(self) -> List[str]:
        """Return optional dependencies for enhanced functionality.
        
        Returns:
            List of optional plugin class names
        """
        return ["Gaze3DClustering", "GazeObjectInteraction"]
    
    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process detected objects to create tracked instances.
        
        Args:
            session: SessionData containing detected objects and gaze clusters
            config: Optional configuration parameters
            
        Returns:
            Dictionary containing tracked instances and metrics
        """
        if config is None:
            config = {}
        
        # Check prerequisites
        if session.gaze.empty:
            error_msg = "No 3D gaze data available for instance tracking"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        if not session.get_camera_names():
            error_msg = "No camera data available for instance tracking"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        # Get plugin-specific config
        plugin_config = config.get('plugin_configs', {}).get(self.__class__.__name__, {})
        
        # Configuration parameters
        self.iou_threshold = plugin_config.get('iou_threshold', 0.3)  # IoU for matching
        self.max_frame_gap = plugin_config.get('max_frame_gap', 10)  # Max frames between detections
        self.min_detections = plugin_config.get('min_detections', 3)  # Min detections to confirm instance
        self.cluster_distance_threshold = plugin_config.get('cluster_distance_threshold', 0.2)  # 20cm default
        
        # 3D visualization configuration
        self.min_gaze_points = plugin_config.get('min_gaze_points', 50)  # Min points for 3D bbox
        self.min_cluster_quality = plugin_config.get('min_cluster_quality', 0.5)
        self.min_gaze_duration_ms = plugin_config.get('min_gaze_duration_ms', 200)
        self.gaze_time_window_ms = plugin_config.get('gaze_time_window_ms', 100)  # Time window for gaze collection
        self.gaze_state_filter = plugin_config.get('gaze_state_filter', ['Fixation', 'Pursuit'])
        self.bbox_padding_m = plugin_config.get('bbox_padding_m', 0.1)  # 10cm padding
        self.outlier_method = plugin_config.get('outlier_method', 'iqr')
        self.outlier_threshold = plugin_config.get('outlier_threshold', 1.5)
        
        self.logger.info(f"Processing object instance tracking with IoU={self.iou_threshold}, "
                        f"max_gap={self.max_frame_gap} frames")
        
        # Check required dependencies
        deps = config.get("dependencies", {})
        if "ObjectDetector" not in deps or "error" in deps.get("ObjectDetector", {}):
            error_msg = "ObjectDetector dependency required but not available or has errors"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        # Get detected objects from ObjectDetector results
        detector_results = deps["ObjectDetector"]
        if not detector_results or "error" in detector_results:
            self.logger.warning("No detected objects available for instance tracking")
            return self._create_empty_results()
        
        # Check if session has detection data
        if not hasattr(session, 'detections') or not session.detections:
            self.logger.warning("No detected objects found in session")
            return self._create_empty_results()
        
        # Count total detections across all cameras
        total_detections = 0
        for camera_name, camera_detections in session.detections.items():
            for frame_id, detections in camera_detections.items():
                total_detections += len(detections)
        
        if total_detections == 0:
            self.logger.warning("No detected objects found in session")
            return self._create_empty_results()
        
        self.logger.info(f"Found {total_detections} total detections across {len(session.detections)} cameras")
        
        # Check optional dependencies
        has_clusters = "Gaze3DClustering" in deps and "error" not in deps.get("Gaze3DClustering", {})
        has_interactions = "GazeObjectInteraction" in deps and "error" not in deps.get("GazeObjectInteraction", {})
        
        if not has_clusters:
            self.logger.info("Gaze3DClustering not available, tracking objects without cluster association")
        if has_interactions:
            self.logger.info("GazeObjectInteraction available, will use Visit-based association")
        
        # Process detections frame by frame
        self._track_instances_across_frames(session)
        
        # Associate with gaze clusters if available
        if has_clusters:
            self._associate_with_gaze_clusters(session, deps)
        else:
            self._association_method = 'none'
        
        # Calculate 3D bounding boxes for well-attended objects
        self._calculate_3d_bounding_boxes(session, deps)
        
        # Calculate metrics
        metrics = self._calculate_tracking_metrics()
        
        # Store results in session
        session.object_instances = self.tracked_instances
        session.set_plugin_result('ObjectInstanceTracker', metrics)
        
        # Generate report
        self._generate_tracking_report(metrics, session)
        
        # Generate CSV reports
        self._generate_csv_reports(metrics, session)
        
        return metrics
    
    def _track_instances_across_frames(self, session: SessionData) -> None:
        """Track object instances across frames using IoU matching.
        
        Args:
            session: SessionData with detected objects organized by frame
        """
        # Collect all detections across all cameras and organize by timestamp
        all_detections = []
        frames_with_detections = defaultdict(list)
        
        for camera_name, camera_detections in session.detections.items():
            for frame_id, detections in camera_detections.items():
                for detection in detections:
                    # Camera context should already be set during detection
                    # Do not overwrite existing camera_name to prevent tracking issues
                    all_detections.append(detection)
                    # Use timestamp as the key for chronological sorting
                    frames_with_detections[detection.timestamp].append(detection)
        
        # Sort frames chronologically by timestamp
        sorted_timestamps = sorted(frames_with_detections.keys())
        
        self.logger.info(f"Tracking {len(all_detections)} detections across {len(sorted_timestamps)} timestamps")
        
        # Track frame by frame
        active_instances = {}  # Maps instance_id to last detection
        frame_gaps = {}  # Track how long each instance has been missing
        
        for timestamp in sorted_timestamps:
            detections = frames_with_detections[timestamp]
            matched_instances = set()  # Track which instances were matched this frame
            
            for det in detections:
                # Try to match with existing active instances
                best_match_id = self._find_best_match(det, active_instances)
                
                if best_match_id is not None:
                    # Update existing instance
                    instance = self.tracked_instances[best_match_id]
                    instance.detections.append(det)
                    instance.confidence_scores.append(det.confidence)
                    instance.last_seen_timestamp = det.timestamp
                    instance.frame_count += 1
                    active_instances[best_match_id] = det
                    matched_instances.add(best_match_id)
                    # Reset frame gap counter
                    frame_gaps.pop(best_match_id, None)
                else:
                    # Create new instance
                    instance_id = self._create_new_instance(det)
                    active_instances[instance_id] = det
                    matched_instances.add(instance_id)
            
            # Update frame gaps for unmatched instances
            for instance_id in list(active_instances.keys()):
                if instance_id not in matched_instances:
                    frame_gaps[instance_id] = frame_gaps.get(instance_id, 0) + 1
                    
                    # Remove instances that have been missing too long
                    if frame_gaps[instance_id] > self.max_frame_gap:
                        active_instances.pop(instance_id, None)
                        frame_gaps.pop(instance_id, None)
        
        self.logger.info(f"Created {len(self.tracked_instances)} total object instances")
        
        # Clean up inactive instances
        self._cleanup_inactive_instances()
    
    def _find_best_match(self, detection: DetectedObject, 
                        active_instances: Dict[int, DetectedObject]) -> Optional[int]:
        """Find best matching instance for a detection.
        
        Args:
            detection: Current detection to match
            active_instances: Dictionary mapping instance_id to last detection
            
        Returns:
            Instance ID of best match, or None if no match found
        """
        best_iou = 0.0
        best_instance_id = None
        
        for instance_id, prev_det in active_instances.items():
            # Check class match
            if prev_det.class_name != detection.class_name:
                continue
            
            # Prefer same camera, but don't strictly require it for cross-camera tracking
            camera_match_bonus = 1.0 if prev_det.camera_name == detection.camera_name else 0.8
            
            # Calculate IoU with camera preference bonus
            iou = self._calculate_iou(prev_det.bbox, detection.bbox)
            weighted_iou = iou * camera_match_bonus
            
            if weighted_iou > self.iou_threshold and weighted_iou > best_iou:
                best_iou = weighted_iou
                best_instance_id = instance_id
        
        return best_instance_id
    
    def _calculate_iou(self, bbox1, bbox2) -> float:
        """Calculate Intersection over Union for two bounding boxes.
        
        Args:
            bbox1, bbox2: BoundingBox objects
            
        Returns:
            IoU score between 0 and 1
        """
        # Get coordinates
        x1_min, y1_min = bbox1.x, bbox1.y
        x1_max, y1_max = bbox1.x + bbox1.width, bbox1.y + bbox1.height
        x2_min, y2_min = bbox2.x, bbox2.y
        x2_max, y2_max = bbox2.x + bbox2.width, bbox2.y + bbox2.height
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _create_new_instance(self, detection: DetectedObject) -> int:
        """Create a new tracked instance.
        
        Args:
            detection: Initial detection for the instance
            
        Returns:
            New instance ID
        """
        instance_id = self.next_instance_id
        self.next_instance_id += 1
        
        # Update per-class counter
        self.class_instance_counters[detection.class_name] += 1
        
        instance = TrackedInstance(
            instance_id=instance_id,
            class_name=detection.class_name,
            first_seen_timestamp=detection.timestamp,
            last_seen_timestamp=detection.timestamp,
            frame_count=1,
            detections=[detection],
            confidence_scores=[detection.confidence]
        )
        
        self.tracked_instances[instance_id] = instance
        return instance_id
    
    def _cleanup_inactive_instances(self) -> None:
        """Remove instances that haven't been seen recently."""
        # For now, we keep all instances for analysis
        # In a real-time system, we would remove inactive ones
        pass
    
    def _filter_gaze_by_hit_type(self, gaze_df: pd.DataFrame) -> pd.DataFrame:
        """Filter gaze data preferring mesh hits over bbox hits.
        
        Args:
            gaze_df: DataFrame with gaze data
            
        Returns:
            Filtered DataFrame with preferred hit types
        """
        if 'gazeHitType' not in gaze_df.columns:
            return gaze_df
        
        mesh_hits = gaze_df[gaze_df['gazeHitType'] == 'mesh']
        if not mesh_hits.empty:
            return mesh_hits
            
        bbox_hits = gaze_df[gaze_df['gazeHitType'] == 'bbox']
        if not bbox_hits.empty:
            return bbox_hits
            
        return gaze_df
    
    def _check_gaze_cluster_proximity(self, gaze_df: pd.DataFrame, 
                                     cluster: GazeCluster) -> bool:
        """Check if any gaze points are within proximity of a cluster.
        
        Args:
            gaze_df: DataFrame with gaze points to check
            cluster: GazeCluster to check proximity against
            
        Returns:
            True if any gaze point is within cluster distance threshold
        """
        for _, gaze_point in gaze_df.iterrows():
            # Get gaze position in Unity coordinates
            gaze_pos = np.array([
                gaze_point.get('posX', 0),
                gaze_point.get('posY', 0),
                gaze_point.get('posZ', 0)
            ])
            
            # Convert to Rerun coordinates
            gaze_pos_rerun = unity_to_rerun_position(gaze_pos.tolist())
            
            # Check distance to cluster centroid
            distance = np.linalg.norm(np.array(gaze_pos_rerun) - cluster.centroid)
            if distance <= self.cluster_distance_threshold:
                return True
        
        return False
    
    def _associate_with_gaze_clusters(self, session: SessionData, deps: Dict[str, Any]) -> None:
        """Associate tracked instances with gaze clusters using Visit data or spatial proximity.
        
        Args:
            session: SessionData with gaze clusters
            deps: Dependency results dictionary
        """
        if not session.gaze_clusters:
            self.logger.info("No gaze clusters available for association")
            return
        
        self.logger.info(f"Associating {len(self.tracked_instances)} instances with "
                        f"{len(session.gaze_clusters)} gaze clusters")
        
        # Try Visit-based association first (most accurate)
        if "GazeObjectInteraction" in deps and "error" not in deps.get("GazeObjectInteraction", {}):
            interaction_results = deps["GazeObjectInteraction"]
            if interaction_results and "metrics" in interaction_results:
                self.logger.info("Using Visit-based cluster association")
                self._association_method = 'visit-based'
                self._associate_using_visits(session, interaction_results["metrics"])
                return
        
        # Fallback to spatial proximity
        self.logger.info("Using spatial proximity for cluster association")
        self._association_method = 'spatial-proximity'
        self._associate_using_spatial_proximity(session)
    
    def _associate_using_visits(self, session: SessionData, interaction_metrics: SessionMetrics) -> None:
        """Use Visit timestamps to find which clusters objects were viewed with.
        
        This creates behaviorally-grounded associations based on actual gaze interactions.
        
        Args:
            session: SessionData with gaze and cluster data
            interaction_metrics: SessionMetrics from GazeObjectInteraction
        """
        associations_made = 0
        
        for instance in self.tracked_instances.values():
            associated_clusters = set()
            
            # Find visits to this object class during instance lifetime
            for camera_name, camera_metrics in interaction_metrics.camera_metrics.items():
                for obj_key_str, obj_metrics in camera_metrics.object_metrics.items():
                    # Check if this is the same object class
                    if obj_metrics.object_key.class_name != instance.class_name:
                        continue
                    
                    # Check each visit for temporal overlap with instance
                    for visit in obj_metrics.visits:
                        # Check if visit overlaps with instance lifetime
                        visit_end = visit.end_timestamp if visit.end_timestamp else session.end_timestamp
                        
                        if (visit.start_timestamp <= instance.last_seen_timestamp and
                            visit_end >= instance.first_seen_timestamp):
                            
                            # Find 3D gaze points during this visit
                            gaze_mask = (session.gaze['timestamp'] >= visit.start_timestamp) & \
                                       (session.gaze['timestamp'] <= visit_end)
                            visit_gaze = session.gaze[gaze_mask]
                            
                            if visit_gaze.empty:
                                continue
                            
                            # Check which clusters these gaze points belong to
                            for cluster_id, cluster in session.gaze_clusters.items():
                                # Filter gaze by hit type preference and check proximity
                                filtered_gaze = self._filter_gaze_by_hit_type(visit_gaze)
                                if self._check_gaze_cluster_proximity(filtered_gaze, cluster):
                                    associated_clusters.add(cluster_id)
            
            # Store associations
            instance.associated_clusters = list(associated_clusters)
            if associated_clusters:
                associations_made += 1
                self.logger.debug(f"Instance {instance.instance_id} ({instance.class_name}) "
                                f"associated with clusters: {associated_clusters}")
        
        self.logger.info(f"Visit-based association: {associations_made}/{len(self.tracked_instances)} "
                        f"instances associated with clusters")
    
    def _associate_using_spatial_proximity(self, session: SessionData) -> None:
        """Fallback: Associate instances with clusters based on spatial proximity.
        
        Uses 3D gaze points during instance lifetime to find nearby clusters.
        
        Args:
            session: SessionData with gaze and cluster data
        """
        associations_made = 0
        
        for instance in self.tracked_instances.values():
            associated_clusters = set()
            
            # Get gaze points during instance lifetime
            time_mask = (session.gaze['timestamp'] >= instance.first_seen_timestamp) & \
                       (session.gaze['timestamp'] <= instance.last_seen_timestamp)
            relevant_gaze = session.gaze[time_mask]
            
            if relevant_gaze.empty:
                self.logger.debug(f"No gaze data during instance {instance.instance_id} lifetime")
                continue
            
            # Check each cluster for proximity to relevant gaze points
            for cluster_id, cluster in session.gaze_clusters.items():
                # Filter gaze by hit type preference and check proximity
                filtered_gaze = self._filter_gaze_by_hit_type(relevant_gaze)
                if self._check_gaze_cluster_proximity(filtered_gaze, cluster):
                    associated_clusters.add(cluster_id)
            
            # Store associations
            instance.associated_clusters = list(associated_clusters)
            if associated_clusters:
                associations_made += 1
                self.logger.debug(f"Instance {instance.instance_id} ({instance.class_name}) "
                                f"spatially associated with clusters: {associated_clusters}")
        
        self.logger.info(f"Spatial proximity association: {associations_made}/{len(self.tracked_instances)} "
                        f"instances associated with clusters")
    
    def _calculate_tracking_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for tracked instances.
        
        Returns:
            Dictionary with tracking metrics
        """
        if not self.tracked_instances:
            return self._create_empty_results()
        
        # Filter by minimum detections if configured
        filtered_instances = {}
        for instance_id, instance in self.tracked_instances.items():
            if instance.frame_count >= self.min_detections:
                filtered_instances[instance_id] = instance
        
        if not filtered_instances:
            self.logger.info(f"No instances met minimum detection threshold of {self.min_detections}")
            return self._create_empty_results()
        
        # Calculate per-class statistics
        class_stats = defaultdict(lambda: {
            'count': 0,
            'total_frames': 0,
            'avg_confidence': 0.0,
            'with_gaze': 0,
            'avg_clusters_per_instance': 0.0
        })
        
        total_with_gaze = 0
        confidence_scores = []
        durations = []
        cluster_counts = []
        association_method = getattr(self, '_association_method', 'none')
        
        for instance in filtered_instances.values():
            stats = class_stats[instance.class_name]
            stats['count'] += 1
            stats['total_frames'] += instance.frame_count
            
            confidence_scores.append(instance.avg_confidence)
            durations.append(instance.duration_ms)
            
            if instance.associated_clusters:
                stats['with_gaze'] += 1
                total_with_gaze += 1
                cluster_counts.append(len(instance.associated_clusters))
            else:
                cluster_counts.append(0)
        
        # Calculate averages
        for class_name, stats in class_stats.items():
            if stats['count'] > 0:
                stats['avg_frames'] = stats['total_frames'] / stats['count']
                stats['gaze_association_rate'] = stats['with_gaze'] / stats['count']
                # Calculate average clusters per instance for this class
                class_instances = [i for i in filtered_instances.values() if i.class_name == class_name]
                class_cluster_counts = [len(i.associated_clusters) for i in class_instances]
                stats['avg_clusters_per_instance'] = np.mean(class_cluster_counts) if class_cluster_counts else 0
        
        return {
            'total_instances': len(filtered_instances),
            'instances_filtered_out': len(self.tracked_instances) - len(filtered_instances),
            'instances_with_gaze': total_with_gaze,
            'gaze_association_rate': total_with_gaze / len(filtered_instances) if filtered_instances else 0,
            'association_method': association_method,
            'avg_clusters_per_instance': np.mean(cluster_counts) if cluster_counts else 0,
            'class_statistics': dict(class_stats),
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'avg_duration_ms': np.mean(durations) if durations else 0,
            'unique_classes': len(self.class_instance_counters),
            'class_counts': dict(self.class_instance_counters),
            'tracked_instances': filtered_instances,
            'configuration': {
                'iou_threshold': self.iou_threshold,
                'max_frame_gap': self.max_frame_gap,
                'min_detections': self.min_detections,
                'cluster_distance_threshold': self.cluster_distance_threshold
            }
        }
    
    def _create_empty_results(self) -> Dict[str, Any]:
        """Create empty results structure.
        
        Returns:
            Empty metrics dictionary
        """
        return {
            'total_instances': 0,
            'instances_filtered_out': 0,
            'instances_with_gaze': 0,
            'gaze_association_rate': 0.0,
            'association_method': 'none',
            'avg_clusters_per_instance': 0.0,
            'class_statistics': {},
            'avg_confidence': 0.0,
            'avg_duration_ms': 0.0,
            'unique_classes': 0,
            'class_counts': {},
            'tracked_instances': {},
            'configuration': {
                'iou_threshold': getattr(self, 'iou_threshold', 0.3),
                'max_frame_gap': getattr(self, 'max_frame_gap', 10),
                'min_detections': getattr(self, 'min_detections', 3),
                'cluster_distance_threshold': getattr(self, 'cluster_distance_threshold', 0.2)
            }
        }
    
    def _generate_tracking_report(self, metrics: Dict[str, Any], session: SessionData) -> None:
        """Generate a tracking report.
        
        Args:
            metrics: Calculated tracking metrics
            session: SessionData for context
        """
        report_lines = [
            f"=== Object Instance Tracking Report ===",
            f"Session: {session.session_id}",
            f"",
            f"Summary:",
            f"  Total Instances Tracked: {metrics['total_instances']}",
            f"  Instances Filtered Out: {metrics.get('instances_filtered_out', 0)}",
            f"  Unique Object Classes: {metrics['unique_classes']}",
            f"  Association Method: {metrics.get('association_method', 'none')}",
            f"  Instances with Gaze Association: {metrics['instances_with_gaze']} "
            f"({metrics['gaze_association_rate']:.1%})",
            f"  Average Clusters per Instance: {metrics.get('avg_clusters_per_instance', 0):.1f}",
            f"  Average Confidence: {metrics['avg_confidence']:.2f}",
            f"  Average Track Duration: {metrics['avg_duration_ms']:.1f} ms",
            f"",
            f"Configuration:",
            f"  IoU Threshold: {metrics['configuration']['iou_threshold']}",
            f"  Max Frame Gap: {metrics['configuration']['max_frame_gap']}",
            f"  Min Detections: {metrics['configuration']['min_detections']}",
            f"  Cluster Distance: {metrics['configuration']['cluster_distance_threshold']}m",
            f"",
            f"Per-Class Statistics:"
        ]
        
        for class_name, stats in metrics['class_statistics'].items():
            report_lines.extend([
                f"  {class_name}:",
                f"    Instances: {stats['count']}",
                f"    Avg Frames per Instance: {stats.get('avg_frames', 0):.1f}",
                f"    Gaze Association Rate: {stats.get('gaze_association_rate', 0):.1%}"
            ])
        
        report_lines.extend([
            f"",
            f"Instance Counts by Class:",
        ])
        
        for class_name, count in sorted(metrics['class_counts'].items()):
            report_lines.append(f"  {class_name}: {count}")
        
        # Log report
        report = "\n".join(report_lines)
        self.logger.info(f"\n{report}")
        
        # Store report in session
        session.tracking_report = report
    
    def visualize(self, results: Dict, rr_stream=None) -> None:
        """Visualize tracked instances in Rerun with consolidated entity paths and proper temporal logging.
        
        Uses batch logging to consolidate all instances under organized 3D world space paths
        instead of creating individual entity paths for each instance. Objects appear at their
        correct detection timestamps throughout the session.
        
        Args:
            results: Results from process method
            rr_stream: Rerun stream for logging
        """
        if not results['tracked_instances']:
            self.logger.info("No tracked instances to visualize")
            return
        
        from src.core.data_types import get_coco_class_color, get_coco_class_id
        
        # Organize instances by class for efficient batching
        instances_by_class = {}
        instances_3d = []
        trajectories_2d = {}
        
        for instance_id, instance in results['tracked_instances'].items():
            class_name = instance.class_name
            if class_name not in instances_by_class:
                instances_by_class[class_name] = []
            instances_by_class[class_name].append((instance_id, instance))
            
            # Collect 2D trajectories by class
            if len(instance.detections) > 1:
                if class_name not in trajectories_2d:
                    trajectories_2d[class_name] = []
                
                # Extract center points of bounding boxes
                centers = []
                for det in instance.detections:
                    center_x = det.bbox.x + det.bbox.width / 2
                    center_y = det.bbox.y + det.bbox.height / 2
                    centers.append([center_x, center_y])
                
                trajectories_2d[class_name].append({
                    'instance_id': instance_id,
                    'centers': centers,
                    'metadata': {
                        'instance_id': instance_id,
                        'frames': instance.frame_count,
                        'confidence': instance.avg_confidence,
                        'duration_ms': instance.duration_ms,
                        'clusters': len(instance.associated_clusters)
                    }
                })
            
            # Collect 3D instances
            if hasattr(instance, 'bbox_3d') and instance.bbox_3d is not None:
                instances_3d.append({
                    'instance_id': instance_id,
                    'instance': instance,
                    'class_name': class_name
                })
        
        # Log 2D trajectories consolidated by class
        for class_name, class_trajectories in trajectories_2d.items():
            if not class_trajectories:
                continue
                
            sanitized_name = self._sanitize_class_name_for_path(class_name)
            
            # Batch all trajectories for this class
            line_strips = [traj['centers'] for traj in class_trajectories]
            
            rr.log(
                f"/tracking/2d_trajectories/{sanitized_name}",
                rr.LineStrips2D(line_strips),
                rr.AnyValues(
                    instance_id=[traj['metadata']['instance_id'] for traj in class_trajectories],
                    frames=[traj['metadata']['frames'] for traj in class_trajectories],
                    confidence=[traj['metadata']['confidence'] for traj in class_trajectories],
                    duration_ms=[traj['metadata']['duration_ms'] for traj in class_trajectories],
                    gaze_clusters=[traj['metadata']['clusters'] for traj in class_trajectories]
                ),
                static=True
            )
        
        # Log 3D objects with proper temporal visualization
        if instances_3d:
            self._visualize_3d_objects_temporally(instances_3d, get_coco_class_color, get_coco_class_id)
        
        visualized_3d = len(instances_3d)
        num_2d_trajectories = sum(len(trajs) for trajs in trajectories_2d.values())
        
        self.logger.info(f"Visualized tracked instances in consolidated paths: "
                        f"2D trajectories: {num_2d_trajectories} across {len(trajectories_2d)} classes, "
                        f"3D objects: {visualized_3d} instances with temporal logging")
    
    def _visualize_3d_objects_temporally(self, instances_3d: List[Dict], get_coco_class_color, get_coco_class_id) -> None:
        """Visualize 3D objects with proper temporal logging.
        
        Objects appear at their detection timestamps instead of all at once.
        
        Args:
            instances_3d: List of 3D instance data
            get_coco_class_color: Function to get COCO class colors
            get_coco_class_id: Function to get COCO class IDs
        """
        # Collect all timestamps where objects are visible
        visibility_by_timestamp = {}  # timestamp -> list of visible instances
        
        for item in instances_3d:
            instance = item['instance']
            
            # Add instance to all timestamps during its lifetime
            for detection in instance.detections:
                timestamp = detection.timestamp
                if timestamp not in visibility_by_timestamp:
                    visibility_by_timestamp[timestamp] = []
                visibility_by_timestamp[timestamp].append(item)
        
        # Log objects at each timestamp where they appear
        for timestamp in sorted(visibility_by_timestamp.keys()):
            visible_instances = visibility_by_timestamp[timestamp]
            
            # Set timeline to this timestamp (convert nanoseconds to seconds)
            rr.set_time("timestamp", timestamp=timestamp / 1e9)
            
            # Prepare batch data for all visible instances at this timestamp
            centers_3d = []
            half_sizes_3d = []
            colors_3d = []
            instance_metadata = {
                'instance_id': [],
                'class_name': [],
                'confidence': [],
                'gaze_quality': [],
                'gaze_points': [],
                'duration_ms': []
            }
            
            for item in visible_instances:
                instance_id = item['instance_id']
                instance = item['instance']
                class_name = item['class_name']
                
                # Get class color
                class_id = get_coco_class_id(class_name)
                color = get_coco_class_color(class_id)
                
                # Adjust opacity based on gaze quality
                if hasattr(instance, 'gaze_quality'):
                    opacity = int(128 + instance.gaze_quality * 127)  # 0.5-1.0 range
                else:
                    opacity = 200
                color_with_alpha = [color[0], color[1], color[2], opacity]
                
                centers_3d.append(instance.bbox_3d['center'])
                half_sizes_3d.append(instance.bbox_3d['half_sizes'])
                colors_3d.append(color_with_alpha)
                
                # Collect metadata
                instance_metadata['instance_id'].append(instance_id)
                instance_metadata['class_name'].append(class_name)
                instance_metadata['confidence'].append(instance.avg_confidence)
                instance_metadata['gaze_quality'].append(getattr(instance, 'gaze_quality', 0.0))
                instance_metadata['gaze_points'].append(getattr(instance, 'gaze_point_count', 0))
                instance_metadata['duration_ms'].append(instance.duration_ms)
            
            # Log all visible instances at this timestamp
            if centers_3d:  # Only log if there are instances to show
                rr.log(
                    "/world/objects/tracked_instances",
                    rr.Boxes3D(
                        centers=centers_3d,
                        half_sizes=half_sizes_3d,
                        colors=colors_3d,
                        fill_mode="DenseWireframe"
                    ),
                    rr.AnyValues(**instance_metadata),
                    static=False
                )
                
                # Log consolidated centroids for easier visualization
                rr.log(
                    "/world/objects/tracked_centroids",
                    rr.Points3D(
                        positions=centers_3d,
                        colors=[color[:3] for color in colors_3d],  # Remove alpha for points
                        radii=0.02
                    ),
                    rr.AnyValues(**instance_metadata),
                    static=False
                )
    
    def _calculate_3d_bounding_boxes(self, session: SessionData, deps: Dict) -> None:
        """Calculate 3D bounding boxes for well-attended instances.
        
        Args:
            session: SessionData with gaze data
            deps: Dependency results including clusters
        """
        has_clusters = "Gaze3DClustering" in deps and "error" not in deps.get("Gaze3DClustering", {})
        
        for instance_id, instance in self.tracked_instances.items():
            # Skip if instance has insufficient gaze association
            if not instance.associated_clusters and not has_clusters:
                continue
            
            # Collect all gaze points within instance's 2D bounding boxes
            gaze_points_3d = []
            
            for det in instance.detections:
                # Get gaze points for this detection's timestamp
                # Convert ms to nanoseconds
                time_window_ns = self.gaze_time_window_ms * 1e6
                frame_gaze = session.gaze[
                    (session.gaze['timestamp'] >= det.timestamp - time_window_ns) &
                    (session.gaze['timestamp'] <= det.timestamp + time_window_ns)
                ]
                
                if frame_gaze.empty:
                    continue
                
                # Filter gaze by hit type if configured
                if hasattr(self, 'gaze_state_filter') and 'gazeState' in frame_gaze.columns:
                    frame_gaze = frame_gaze[frame_gaze['gazeState'].isin(self.gaze_state_filter)]
                
                # Check which gaze points fall within 2D bbox
                for _, gaze_row in frame_gaze.iterrows():
                    if not gaze_row.get('hasHitTarget', False):
                        continue
                    
                    # Get 3D position
                    if 'gazePositionX' in gaze_row and 'gazePositionY' in gaze_row and 'gazePositionZ' in gaze_row:
                        pos_3d = [gaze_row['gazePositionX'], gaze_row['gazePositionY'], gaze_row['gazePositionZ']]
                        
                        # Check for NaN/Inf values
                        if not all(np.isfinite(pos_3d)):
                            continue
                        
                        try:
                            # Convert to Rerun coordinates
                            rerun_pos = unity_to_rerun_position(pos_3d)
                            gaze_points_3d.append(rerun_pos)
                        except Exception as e:
                            self.logger.warning(f"Failed to convert gaze position {pos_3d}: {e}")
                            continue
            
            # Check if we have enough points for reliable 3D bbox
            if len(gaze_points_3d) < self.min_gaze_points:
                continue
            
            # Check cluster quality if available
            if has_clusters and instance.associated_clusters:
                cluster_results = deps["Gaze3DClustering"]
                if 'clusters' in cluster_results:
                    avg_quality = np.mean([
                        cluster_results['clusters'][cid].quality_score
                        for cid in instance.associated_clusters
                        if cid in cluster_results['clusters']
                    ])
                    if avg_quality < self.min_cluster_quality:
                        continue
                    instance.gaze_quality = avg_quality
                else:
                    instance.gaze_quality = 0.5
            else:
                instance.gaze_quality = 0.5
            
            # Calculate 3D bounding box from gaze points
            try:
                bbox_3d = self._compute_gaze_based_bbox(instance, gaze_points_3d)
                if bbox_3d is not None:
                    instance.bbox_3d = bbox_3d
                    instance.gaze_point_count = len(gaze_points_3d)
            except Exception as e:
                self.logger.warning(f"Failed to compute 3D bbox for instance {instance_id}: {e}")
                continue
    
    def _sanitize_class_name_for_path(self, class_name: str) -> str:
        """Sanitize class name for use in entity paths by replacing spaces with underscores.
        
        Args:
            class_name: Original class name (may contain spaces)
            
        Returns:
            Sanitized class name suitable for entity paths
        """
        return class_name.replace(" ", "_")
    
    def _compute_gaze_based_bbox(self, instance: TrackedInstance, gaze_points: List[np.ndarray]) -> Optional[Dict]:
        """Compute 3D bounding box from gaze points.
        
        Args:
            instance: Tracked instance
            gaze_points: List of 3D gaze positions
            
        Returns:
            Dict with 'center' and 'half_sizes' or None if insufficient data
        """
        if len(gaze_points) < 3:  # Need minimum points
            return None
        
        points = np.array(gaze_points)
        
        # Remove outliers using IQR method
        if self.outlier_method == 'iqr' and len(points) > 10:
            q1 = np.percentile(points, 25, axis=0)
            q3 = np.percentile(points, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - self.outlier_threshold * iqr
            upper_bound = q3 + self.outlier_threshold * iqr
            
            # Filter to inliers only
            mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
            filtered_points = points[mask]
            
            if len(filtered_points) >= 3:
                points = filtered_points
        
        # Calculate bounds with padding
        min_bounds = np.min(points, axis=0) - self.bbox_padding_m
        max_bounds = np.max(points, axis=0) + self.bbox_padding_m
        
        center = (min_bounds + max_bounds) / 2
        half_sizes = (max_bounds - min_bounds) / 2
        
        return {
            'center': center.tolist(),
            'half_sizes': half_sizes.tolist(),
            'min_bounds': min_bounds.tolist(),
            'max_bounds': max_bounds.tolist()
        }
    
    def _generate_csv_reports(self, metrics: Dict[str, Any], session: SessionData) -> None:
        """Generate CSV reports for spatial analysis.
        
        Args:
            metrics: Calculated tracking metrics
            session: SessionData for context
        """
        from pathlib import Path
        
        # Create report directory
        report_dir = Path(f"reports/{session.session_id}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. spatial_instances.csv - Instance details with 2D and 3D metrics
        instance_data = []
        for instance_id, instance in metrics['tracked_instances'].items():
            # Calculate 2D metrics
            avg_bbox_width = np.mean([d.bbox.width for d in instance.detections]) if instance.detections else 0
            avg_bbox_height = np.mean([d.bbox.height for d in instance.detections]) if instance.detections else 0
            
            # Get 3D metrics if available
            bbox_3d = getattr(instance, 'bbox_3d', None)
            
            instance_data.append({
                'instance_id': instance_id,
                'class_name': instance.class_name,
                'first_seen_timestamp': instance.first_seen_timestamp,
                'last_seen_timestamp': instance.last_seen_timestamp,
                'duration_ms': instance.duration_ms,
                'frame_count': instance.frame_count,
                'avg_confidence': instance.avg_confidence,
                # 2D metrics
                'avg_bbox_width_2d': avg_bbox_width,
                'avg_bbox_height_2d': avg_bbox_height,
                # 3D metrics
                'bbox_width_m': bbox_3d['half_sizes'][0] * 2 if bbox_3d else None,
                'bbox_height_m': bbox_3d['half_sizes'][1] * 2 if bbox_3d else None,
                'bbox_depth_m': bbox_3d['half_sizes'][2] * 2 if bbox_3d else None,
                'centroid_x': bbox_3d['center'][0] if bbox_3d else None,
                'centroid_y': bbox_3d['center'][1] if bbox_3d else None,
                'centroid_z': bbox_3d['center'][2] if bbox_3d else None,
                # Gaze metrics
                'associated_clusters': ','.join(map(str, instance.associated_clusters)),
                'num_clusters': len(instance.associated_clusters),
                'gaze_quality': getattr(instance, 'gaze_quality', None),
                'gaze_point_count': getattr(instance, 'gaze_point_count', 0),
                'is_visualized_3d': bbox_3d is not None
            })
        
        if instance_data:
            instance_df = pd.DataFrame(instance_data)
            instance_df = instance_df.sort_values('duration_ms', ascending=False)
            instance_path = report_dir / "spatial_instances.csv"
            instance_df.to_csv(instance_path, index=False)
            self.logger.info(f"Saved instance report to {instance_path}")
        
        # 2. spatial_summary.csv - Overview statistics
        num_visualized_3d = sum(1 for i in metrics['tracked_instances'].values() 
                                if hasattr(i, 'bbox_3d') and i.bbox_3d is not None)
        
        summary_data = {
            'metric': [
                'total_instances',
                'instances_filtered_out',
                'instances_visualized_3d',
                'instances_with_gaze',
                'gaze_association_rate',
                'association_method',
                'avg_clusters_per_instance',
                'avg_confidence',
                'avg_duration_ms',
                'unique_classes',
                'min_gaze_points_threshold',
                'min_cluster_quality_threshold',
                'min_gaze_duration_ms_threshold'
            ],
            'value': [
                metrics['total_instances'],
                metrics.get('instances_filtered_out', 0),
                num_visualized_3d,
                metrics['instances_with_gaze'],
                f"{metrics['gaze_association_rate']:.2%}",
                metrics.get('association_method', 'none'),
                f"{metrics.get('avg_clusters_per_instance', 0):.2f}",
                f"{metrics['avg_confidence']:.2f}",
                f"{metrics['avg_duration_ms']:.1f}",
                metrics['unique_classes'],
                self.min_gaze_points,
                self.min_cluster_quality,
                self.min_gaze_duration_ms
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = report_dir / "spatial_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"Saved summary report to {summary_path}")