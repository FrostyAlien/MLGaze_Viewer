"""Gaze-Object Interaction Analysis Plugin for MLGaze Viewer."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core import SessionData
from src.core.data_types import (
    DetectedObject, ObjectInteractionMetrics, ObjectKey, CameraMetrics, SessionMetrics, Visit
)
from src.plugin_sys.base import AnalyticsPlugin


class GazeObjectInteraction(AnalyticsPlugin):
    """Analyze interactions between 3D gaze data and detected 2D objects using Visit-based tracking.
    
    This plugin maps gaze points to detected objects, calculates dwell times,
    tracks fixations, and analyzes transitions between objects. It depends on
    ObjectDetector results and processes each camera's gaze screen coordinates.
    
    Architecture:
    - Uses Visit-based tracking system for accurate dwell time calculations
    - Tracks simultaneous visits to ALL overlapping objects (not just primary)
    - Location-based visits only (not dependent on eye status)
    - Maintains camera separation (no cross-camera transitions)
    - Counts fixation samples with configurable interruption tolerance
    """
    
    def __init__(self, 
                 overlap_strategy: str = "highest_confidence",
                 dwell_interruption_ms: float = 100.0,
                 minimum_dwell_ms: float = 50.0,
                 chunk_size: int = 1000):
        """Initialize the Gaze-Object Interaction analyzer.
        
        Args:
            overlap_strategy: How to handle overlapping objects:
                             "highest_confidence", "smallest_bbox", or "all"
            dwell_interruption_ms: Max gap to consider continuous dwell (for blinks)
            minimum_dwell_ms: Minimum dwell time to count as valid interaction
            chunk_size: Number of gaze samples to process at once for performance
        """
        super().__init__("Gaze-Object Interaction")
        self.overlap_strategy = overlap_strategy
        self.dwell_interruption_ms = dwell_interruption_ms
        self.minimum_dwell_ms = minimum_dwell_ms
        self.chunk_size = chunk_size
        
        # Validate configuration
        self._validate_config()
    
    def get_dependencies(self) -> List[str]:
        """Requires ObjectDetector to provide detected objects."""
        return ["ObjectDetector"]
    
    def get_optional_dependencies(self) -> List[str]:
        """No optional dependencies."""
        return []
    
    def _validate_config(self):
        """Validate configuration parameters."""
        valid_strategies = {"highest_confidence", "smallest_bbox", "all"}
        if self.overlap_strategy not in valid_strategies:
            raise ValueError(f"overlap_strategy must be one of {valid_strategies}")
        
        if self.dwell_interruption_ms < 0:
            raise ValueError("dwell_interruption_ms must be non-negative")
            
        if self.minimum_dwell_ms < 0:
            raise ValueError("minimum_dwell_ms must be non-negative")
            
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
    
    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process gaze-object interactions for all cameras.
        
        Args:
            session: SessionData containing gaze coordinates and object detections
            config: Configuration dictionary with dependency results
            
        Returns:
            Dictionary containing interaction metrics and analysis results
        """
        start_time = time.time()
        
        # Update configuration if provided
        if config:
            deps = config.get("dependencies", {})
            if "ObjectDetector" not in deps:
                error_msg = "ObjectDetector dependency required but not found"
                self.logger.error(error_msg)
                return {"error": error_msg}
        
        # Get object detection results
        detection_results = config["dependencies"]["ObjectDetector"]
        if "error" in detection_results:
            error_msg = f"ObjectDetector dependency has error: {detection_results['error']}"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        self.logger.info("Starting gaze-object interaction analysis...")
        self.logger.info(f"Configuration: overlap={self.overlap_strategy}, "
                        f"interruption_tolerance={self.dwell_interruption_ms}ms, "
                        f"min_dwell={self.minimum_dwell_ms}ms")
        
        # Use new SessionMetrics structure for clean organization
        session_metrics = SessionMetrics(
            session_id=session.session_id,
            configuration={
                "overlap_strategy": self.overlap_strategy,
                "dwell_interruption_ms": self.dwell_interruption_ms,
                "minimum_dwell_ms": self.minimum_dwell_ms,
                "chunk_size": self.chunk_size
            }
        )
        
        # Process each camera
        cameras_to_process = [cam for cam in session.get_camera_names() 
                             if cam in session.gaze_screen_coords and 
                                session.gaze_screen_coords[cam] is not None]
        
        if not cameras_to_process:
            self.logger.warning("No cameras with gaze screen coordinates found")
            return {"metrics": session_metrics}
        
        self.logger.info(f"Processing {len(cameras_to_process)} cameras with gaze coordinates")
        
        for camera_name in cameras_to_process:
            self.logger.info(f"Processing camera: {camera_name}")
            camera_metrics = self._process_camera(session, camera_name)
            session_metrics.camera_metrics[camera_name] = camera_metrics
        
        # Calculate final statistics and complete session metrics
        session_metrics.processing_time_s = time.time() - start_time
        session_metrics.global_statistics = {
            "total_gaze_samples": session_metrics.total_gaze_samples,
            "samples_on_objects": session_metrics.total_samples_on_objects,
            "hit_rate": session_metrics.overall_hit_rate,
            "cameras_processed": session_metrics.total_cameras
        }
        
        # Create clean return format
        results = {
            "metrics": session_metrics,
            "session_data": session
        }
        
        # Store results in session for other plugins
        session.set_plugin_result("GazeObjectInteraction", results)
        
        self.logger.info(f"Interaction analysis complete in {session_metrics.processing_time_s:.2f}s")
        self.logger.info(f"Hit rate: {session_metrics.overall_hit_rate:.1%} "
                        f"({session_metrics.total_samples_on_objects:,}/{session_metrics.total_gaze_samples:,} samples)")
        self.logger.info(f"Found {len(session_metrics.get_all_objects())} unique objects with gaze interactions")
        
        # Generate and save reports
        reports = self._generate_reports(session_metrics, session)
        saved_files = self._save_reports(reports, session)
        if saved_files:
            self.logger.info(f"Saved {len(saved_files)} report files")
        
        # Print summary to console
        self._print_console_summary(reports)
        
        return results
    
    def _process_camera(self, session: SessionData, camera_name: str) -> CameraMetrics:
        """Process gaze interactions for a single camera using Visit-based tracking.
        
        Implements a clean state machine that tracks multiple simultaneous visits
        to overlapping objects with proper interruption tolerance.
        
        Args:
            session: SessionData containing all sensor data
            camera_name: Name of camera to process
            
        Returns:
            CameraMetrics with interaction results
        """
        # Get gaze coordinates and object detections for this camera
        gaze_coords = session.get_gaze_coords_for_camera(camera_name)
        camera_detections = session.detections.get(camera_name, {})
        
        if gaze_coords is None or gaze_coords.empty:
            self.logger.warning(f"No gaze coordinates found for camera {camera_name}")
            return CameraMetrics(camera_name=camera_name)
        
        if not camera_detections:
            self.logger.warning(f"No object detections found for camera {camera_name}")
            camera_metrics = CameraMetrics(camera_name=camera_name)
            camera_metrics.total_gaze_samples = len(gaze_coords)
            return camera_metrics
        
        self.logger.debug(f"Processing {len(gaze_coords)} gaze samples against "
                         f"{len(camera_detections)} frames with detections")
        
        # Initialize clean Visit-based tracking
        object_metrics = {}  # Dict[str, ObjectInteractionMetrics]
        current_visits = {}  # Dict[str, Visit] - Track ALL ongoing visits
        last_object_timestamps = {}  # Dict[str, int] - For interruption tolerance
        
        camera_metrics = CameraMetrics(camera_name=camera_name)
        
        # Sort gaze coordinates by timestamp for proper sequence analysis
        gaze_coords_sorted = gaze_coords.sort_values('timestamp')
        camera_metrics.total_gaze_samples = len(gaze_coords_sorted)
        
        # Process in chunks with progress tracking
        total_chunks = len(gaze_coords_sorted) // self.chunk_size + 1
        
        for chunk_start in tqdm(range(0, len(gaze_coords_sorted), self.chunk_size),
                               desc=f"  Processing {camera_name}",
                               total=total_chunks,
                               leave=False):
            
            chunk_end = min(chunk_start + self.chunk_size, len(gaze_coords_sorted))
            chunk = gaze_coords_sorted.iloc[chunk_start:chunk_end]
            
            for idx, gaze_sample in chunk.iterrows():
                timestamp = gaze_sample['timestamp']
                gaze_state = gaze_sample.get('gazeState', 'Unknown')
                
                # Find ALL objects containing this gaze point
                target_objects = self._find_gaze_targets(gaze_sample, camera_detections, session)
                
                # Create ObjectKeys for ALL target objects
                current_object_keys = set()
                if target_objects:
                    camera_metrics.samples_on_objects += 1
                    for obj in target_objects:
                        obj_key = ObjectKey(
                            camera_name=camera_name,
                            class_name=obj.class_name,
                            instance_id=obj.instance_id
                        )
                        current_object_keys.add(obj_key)
                
                # Convert to string keys for dictionary usage
                current_key_strs = {str(k) for k in current_object_keys}
                previous_key_strs = set(current_visits.keys())
                
                # Process visits ending (objects no longer being looked at)
                for obj_key_str in previous_key_strs - current_key_strs:
                    if obj_key_str in last_object_timestamps:
                        time_gap_ns = timestamp - last_object_timestamps[obj_key_str]
                        time_gap_ms = time_gap_ns / 1_000_000  # Convert to ms
                        
                        # Check interruption tolerance
                        if time_gap_ms > self.dwell_interruption_ms:
                            # End visit - gap too large
                            self._complete_visit(obj_key_str, timestamp, current_visits, object_metrics, camera_metrics)
                
                # Process visits starting (new objects being looked at)
                for obj_key_str in current_key_strs - previous_key_strs:
                    # Start new visit
                    object_key = next(k for k in current_object_keys if str(k) == obj_key_str)
                    self._start_visit(obj_key_str, object_key, timestamp, current_visits, object_metrics, camera_metrics)
                
                # Update ALL ongoing visits
                for obj_key_str in current_key_strs:
                    if obj_key_str in current_visits:
                        # Add sample to ongoing visit
                        current_visits[obj_key_str].gaze_samples.append(timestamp)
                        
                        # Count fixation samples (as requested)
                        if gaze_state == 'Fixation':
                            current_visits[obj_key_str].fixation_sample_count += 1
                        
                        # Update gaze state distribution
                        if obj_key_str in object_metrics:
                            metrics = object_metrics[obj_key_str]
                            if gaze_state not in metrics.gaze_state_distribution:
                                metrics.gaze_state_distribution[gaze_state] = 0
                            metrics.gaze_state_distribution[gaze_state] += 1
                
                # Update last seen timestamps for interruption tolerance
                for obj_key_str in current_key_strs:
                    last_object_timestamps[obj_key_str] = timestamp
                
                # Store frame interaction data for visualization
                if target_objects:
                    frame_id = target_objects[0].frame_id  # Use first object's frame_id
                    camera_metrics.frame_interactions.setdefault(frame_id, []).append({
                        "timestamp": timestamp,
                        "objects_hit": list(current_key_strs),
                        "gaze_state": gaze_state,
                        "screen_coords": [gaze_sample.get('x', 0), gaze_sample.get('y', 0)]
                    })
        
        # Complete all remaining visits at end of processing
        if gaze_coords_sorted.empty:
            final_timestamp = 0
        else:
            final_timestamp = gaze_coords_sorted.iloc[-1]['timestamp']
            
        for obj_key_str in list(current_visits.keys()):
            self._complete_visit(obj_key_str, final_timestamp, current_visits, object_metrics, camera_metrics)
        
        # Set final statistics
        camera_metrics.object_metrics = object_metrics
        camera_metrics.unique_objects = len(object_metrics)
        
        return camera_metrics
    
    def _start_visit(self, obj_key_str: str, object_key: ObjectKey, timestamp: int,
                    current_visits: Dict[str, Visit], 
                    object_metrics: Dict[str, ObjectInteractionMetrics],
                    camera_metrics: CameraMetrics) -> None:
        """Start a new visit to an object."""
        # Initialize object metrics if needed
        if obj_key_str not in object_metrics:
            object_metrics[obj_key_str] = ObjectInteractionMetrics(object_key=object_key)
        
        # Start new visit
        new_visit = Visit(start_timestamp=timestamp)
        new_visit.gaze_samples.append(timestamp)
        current_visits[obj_key_str] = new_visit
        
        # Update entry count and record transition
        object_metrics[obj_key_str].total_entry_count += 1
        camera_metrics.add_transition(None, object_key.class_name, timestamp, None, obj_key_str)
    
    def _complete_visit(self, obj_key_str: str, end_timestamp: int,
                       current_visits: Dict[str, Visit],
                       object_metrics: Dict[str, ObjectInteractionMetrics], 
                       camera_metrics: CameraMetrics) -> None:
        """Complete a visit and update metrics."""
        if obj_key_str not in current_visits:
            return
            
        visit = current_visits[obj_key_str]
        visit.end_timestamp = end_timestamp
        
        # Apply minimum dwell time filtering
        visit_duration_ms = visit.duration_ns / 1_000_000  # Convert to milliseconds
        
        if visit_duration_ms >= self.minimum_dwell_ms:
            # Visit meets minimum duration requirement, add to metrics
            if obj_key_str in object_metrics:
                object_metrics[obj_key_str].add_visit(visit)
                
                # Update exit count and record transition
                object_metrics[obj_key_str].total_exit_count += 1
                class_name = object_metrics[obj_key_str].class_name
                camera_metrics.add_transition(class_name, None, end_timestamp, obj_key_str, None)
        else:
            # Visit too short, just update exit count without adding visit
            if obj_key_str in object_metrics:
                object_metrics[obj_key_str].total_exit_count += 1
                class_name = object_metrics[obj_key_str].class_name
                camera_metrics.add_transition(class_name, None, end_timestamp, obj_key_str, None)
        
        # Remove from current visits
        del current_visits[obj_key_str]
    
    def _find_gaze_targets(self, gaze_sample: pd.Series, camera_detections: Dict, 
                          session: SessionData) -> List[DetectedObject]:
        """Find which detected objects contain the gaze point.
        
        Args:
            gaze_sample: Single gaze sample with screen coordinates
            camera_detections: Dictionary of frame_id -> List[DetectedObject]
            session: SessionData for frame metadata
            
        Returns:
            List of DetectedObject instances that contain the gaze point
        """
        try:
            # Validate input data
            if not camera_detections:
                return []
                
            # Extract gaze screen coordinates
            gaze_x = gaze_sample.get('screenPixelX')
            gaze_y = gaze_sample.get('screenPixelY')
            gaze_timestamp = gaze_sample.get('timestamp')
            
            if pd.isna(gaze_x) or pd.isna(gaze_y) or gaze_timestamp is None:
                return []
            
            gaze_point = np.array([gaze_x, gaze_y])
            
            # Find the closest frame by timestamp
            closest_frame_id = self._find_closest_frame(gaze_timestamp, camera_detections)
            
            if not closest_frame_id:
                return []
            
            detections = camera_detections.get(closest_frame_id, [])
            if not detections:
                return []
                
            target_objects = []
            
            # Check which objects contain the gaze point
            for detection in detections:
                try:
                    if detection.bbox and detection.bbox.contains_point(gaze_point):
                        target_objects.append(detection)
                except (AttributeError, IndexError, TypeError) as e:
                    # Log malformed bounding box but continue processing
                    self.logger.debug(f"Skipping malformed detection bbox: {e}")
                    continue
            
            # Apply overlap strategy to filter objects if needed
            return self._apply_overlap_strategy(target_objects)
            
        except Exception as e:
            self.logger.error(f"Error finding gaze targets: {e}")
            return []
    
    def _find_closest_frame(self, timestamp: int, detections: Dict) -> Optional[str]:
        """Find the frame_id with timestamp closest to the gaze timestamp.
        
        Args:
            timestamp: Gaze timestamp in nanoseconds
            detections: Dictionary of frame_id -> List[DetectedObject]
            
        Returns:
            frame_id of closest frame, or None if no suitable frame found
        """
        if not detections:
            return None
        
        # Find closest frame by comparing detection timestamps
        best_frame = None
        min_diff = float('inf')
        
        for frame_id, detection_list in detections.items():
            if detection_list:  # Make sure frame has detections
                # Use timestamp from first detection in frame
                frame_timestamp = detection_list[0].timestamp
                time_diff = abs(timestamp - frame_timestamp)
                
                if time_diff < min_diff:
                    min_diff = time_diff
                    best_frame = frame_id
        
        # Only return frame if within reasonable tolerance (e.g., 100ms for 10fps minimum)
        if min_diff <= 100_000_000:  # 100ms in nanoseconds
            return best_frame
        
        return None
    
    def _apply_overlap_strategy(self, target_objects: List[DetectedObject]) -> List[DetectedObject]:
        """Apply overlap strategy to filter overlapping objects.
        
        Args:
            target_objects: List of overlapping detected objects
            
        Returns:
            Filtered list based on overlap strategy
        """
        if not target_objects or len(target_objects) <= 1:
            return target_objects
            
        if self.overlap_strategy == "all":
            # Track all overlapping objects simultaneously
            return target_objects
        elif self.overlap_strategy == "highest_confidence":
            # Select object with highest confidence score
            return [max(target_objects, key=lambda obj: obj.confidence)]
        elif self.overlap_strategy == "smallest_bbox":
            # Select object with smallest bounding box area
            def bbox_area(obj):
                bounds = obj.bbox.bounds
                return bounds[2] * bounds[3]  # width * height
            return [min(target_objects, key=bbox_area)]
        else:
            # Fallback to all objects if strategy not recognized
            self.logger.warning(f"Unknown overlap strategy '{self.overlap_strategy}', using 'all'")
            return target_objects
    
    def _generate_reports(self, session_metrics: SessionMetrics, session: SessionData) -> Dict[str, pd.DataFrame]:
        """Generate DataFrame reports for visualization and saving.
        
        Args:
            session_metrics: SessionMetrics with all interaction data
            session: SessionData for metadata
            
        Returns:
            Dictionary containing different report DataFrames
        """
        reports = {}
        
        try:
            # Object Metrics Report - collect from all cameras
            object_data = []
            for camera_name, camera_metrics in session_metrics.camera_metrics.items():
                for obj_key, metrics in camera_metrics.object_metrics.items():
                    class_name = metrics.class_name
                    object_camera = metrics.object_key.camera_name
                    
                    object_data.append({
                        "class_name": class_name,
                        "camera_name": object_camera,  # Use camera from ObjectKey
                        "total_dwell_s": round(metrics.total_dwell_s, 3),
                        "fixation_count": metrics.fixation_event_count,
                        "entry_count": metrics.total_entry_count,
                        "exit_count": metrics.total_exit_count,
                        "avg_dwell_per_visit_s": round(metrics.average_dwell_per_visit_s, 3),
                        "gaze_samples": metrics.total_gaze_samples,
                        "first_look_timestamp": metrics.first_look_timestamp,
                        "last_look_timestamp": metrics.last_look_timestamp
                    })
            
            if object_data:
                
                reports["object_metrics"] = pd.DataFrame(object_data)
                # Sort by total dwell time descending
                reports["object_metrics"] = reports["object_metrics"].sort_values(
                    "total_dwell_s", ascending=False
                ).reset_index(drop=True)
            
            # Transition Matrix Report - use actual camera information from session_metrics
            transition_counts = {}  # (from_class, to_class, camera) -> count
            
            # Collect transition details from all cameras
            self.logger.debug(f"Processing transitions for {len(session_metrics.camera_metrics)} cameras")
            for camera_name, camera_metrics in session_metrics.camera_metrics.items():
                self.logger.debug(f"Camera {camera_name} has {len(camera_metrics.transition_details)} transitions")
                for detail in camera_metrics.transition_details:
                    # Use camera name from detail instead of loop variable for accuracy
                    key = (detail["from_class"], detail["to_class"], detail["camera_name"])
                    transition_counts[key] = transition_counts.get(key, 0) + 1
            
            # Build DataFrame with proper camera information
            if transition_counts:
                transition_data = []
                for (from_class, to_class, camera), count in transition_counts.items():
                    transition_data.append({
                        "from_class": from_class,
                        "to_class": to_class,
                        "camera": camera,  # Single camera column
                        "transition_count": count
                    })
                
                reports["transitions"] = pd.DataFrame(transition_data)
                # Sort by transition count descending
                reports["transitions"] = reports["transitions"].sort_values(
                    "transition_count", ascending=False
                ).reset_index(drop=True)
            
            # Summary Statistics Report from SessionMetrics
            stats = {
                "total_gaze_samples": session_metrics.total_gaze_samples,
                "samples_on_objects": session_metrics.total_samples_on_objects,
                "hit_rate": session_metrics.overall_hit_rate,
                "cameras_processed": session_metrics.total_cameras,
                "processing_time_s": session_metrics.processing_time_s
            }
            config = session_metrics.configuration
            
            summary_data = [{
                "metric": "Total Gaze Samples",
                "value": f"{stats.get('total_gaze_samples', 0):,}"
            }, {
                "metric": "Samples on Objects",
                "value": f"{stats.get('samples_on_objects', 0):,}"
            }, {
                "metric": "Hit Rate",
                "value": f"{stats.get('hit_rate', 0):.1%}"
            }, {
                "metric": "Unique Objects",
                "value": str(len(session_metrics.get_all_objects()))
            }, {
                "metric": "Cameras Processed",
                "value": str(stats.get('cameras_processed', 0))
            }, {
                "metric": "Processing Time",
                "value": f"{stats.get('processing_time_s', 0):.2f}s"
            }, {
                "metric": "Session ID",
                "value": session.session_id or "Unknown"
            }, {
                "metric": "Overlap Strategy",
                "value": config.get('overlap_strategy', 'Unknown')
            }, {
                "metric": "Dwell Interruption (ms)",
                "value": str(config.get('dwell_interruption_ms', 'Unknown'))
            }, {
                "metric": "Min Dwell (ms)",
                "value": str(config.get('minimum_dwell_ms', 'Unknown'))
            }]
            
            reports["summary"] = pd.DataFrame(summary_data)
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
        
        return reports
    
    def _save_reports(self, reports: Dict[str, pd.DataFrame], session: SessionData) -> List[str]:
        """Save reports to files in reports/{session_id}/ directory.
        
        Args:
            reports: Dictionary of report DataFrames
            session: SessionData for session ID
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        try:
            # Create reports directory
            session_id = session.session_id or "unknown_session"
            reports_dir = Path("reports") / session_id
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each report as CSV
            for report_name, df in reports.items():
                if df is not None and not df.empty:
                    csv_path = reports_dir / f"gaze_object_interaction_{report_name}.csv"
                    df.to_csv(csv_path, index=False)
                    saved_files.append(str(csv_path))
                    self.logger.info(f"Saved {report_name} report to {csv_path}")
            
            # Save complete results as JSON
            json_path = reports_dir / "gaze_object_interaction_complete.json"
            with open(json_path, 'w') as f:
                # Convert numpy types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                # Create a serializable copy of results
                serializable_results = {}
                for key, value in reports.items():
                    if isinstance(value, pd.DataFrame):
                        serializable_results[key] = value.to_dict('records')
                    else:
                        serializable_results[key] = value
                
                json.dump(serializable_results, f, indent=2, default=convert_numpy)
            
            saved_files.append(str(json_path))
            self.logger.info(f"Saved complete results to {json_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving reports: {e}")
        
        return saved_files

    def visualize(self, results: Dict[str, Any], recording_stream=None) -> None:
        """No-op visualization method - reports generated in process()."""
        pass
    
    def _print_console_summary(self, reports: Dict[str, pd.DataFrame]) -> None:
        """Print summary report to console.
        
        Args:
            reports: Generated DataFrame reports
        """
        try:
            print("\n" + "=" * 60)
            print("           GAZE-OBJECT INTERACTION SUMMARY")
            print("=" * 60)
            
            # Summary statistics
            summary_df = reports.get("summary")
            if summary_df is not None:
                for _, row in summary_df.iterrows():
                    print(f"{row['metric']:<25}: {row['value']}")
            
            # Top objects by dwell time
            object_df = reports.get("object_metrics")
            if object_df is not None and not object_df.empty:
                print(f"\nTop Objects by Dwell Time:")
                print("-" * 60)
                top_objects = object_df.head(10)
                for _, obj in top_objects.iterrows():
                    camera = obj.get('camera_name', 'unknown')
                    dwell_ms = obj['total_dwell_s'] * 1000  # Convert seconds to ms for display
                    print(f"{obj['class_name']:<15} [{camera:<12}]: {dwell_ms:.0f}ms "
                          f"({obj['fixation_count']} fixations, {obj['entry_count']} entries)")
            
            # Top transitions
            transition_df = reports.get("transitions")
            if transition_df is not None and not transition_df.empty:
                print(f"\nTop Object Transitions:")
                print("-" * 60)
                top_transitions = transition_df.head(10)
                for _, trans in top_transitions.iterrows():
                    camera = trans.get('camera', 'unknown')
                    print(f"{trans['from_class']} → {trans['to_class']} [{camera}]: {trans['transition_count']} times")
            
            print("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Error printing console summary: {e}")
