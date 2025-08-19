"""Data type definitions for MLGaze Viewer sensors and analytics."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


# Base Classes for Common Patterns

@dataclass
class TimestampedData:
    """Base class for all timestamped data."""
    timestamp: int  # Nanoseconds


@dataclass
class SpatialEntity:
    """Base class for entities with spatial properties."""
    camera_name: str
    frame_id: str


# Composite Key Classes for Multi-dimensional Data

@dataclass(frozen=True)
class ObjectKey:
    """Immutable key for unique object identification across cameras."""
    camera_name: str
    class_name: str
    instance_id: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation for logging and display."""
        if self.instance_id:
            return f"{self.class_name}[{self.instance_id}]@{self.camera_name}"
        return f"{self.class_name}@{self.camera_name}"
    
    @property
    def class_camera_key(self) -> Tuple[str, str]:
        """Get (class_name, camera_name) tuple for compatibility."""
        return (self.class_name, self.camera_name)


@dataclass(frozen=True)
class TransitionKey:
    """Immutable key for object transitions."""
    from_object: ObjectKey
    to_object: ObjectKey
    
    def __str__(self) -> str:
        """String representation for logging and display."""
        return f"{self.from_object} → {self.to_object}"
    
    @property
    def is_cross_camera(self) -> bool:
        """Check if transition crosses camera boundaries."""
        return self.from_object.camera_name != self.to_object.camera_name


# Core Sensor Data Types

@dataclass
class GazeSample(TimestampedData):
    """Single gaze sample with 3D and 2D projection data."""
    frame_id: str
    origin: np.ndarray  # 3D position [x, y, z]
    direction: np.ndarray  # 3D vector [x, y, z]
    hit_point: Optional[np.ndarray] = None  # 3D position where gaze hits
    screen_point: Optional[np.ndarray] = None  # 2D projection [x, y]
    state: str = "Unknown"  # Fixation/Saccade/Pursuit/etc
    confidence: float = 1.0
    is_tracking: bool = True
    has_hit_target: bool = False
    is_valid_projection: bool = False


@dataclass
class CameraPose(TimestampedData, SpatialEntity):
    """Camera position and orientation in world space."""
    position: np.ndarray  # 3D position [x, y, z]
    rotation: np.ndarray  # Quaternion [x, y, z, w]
    
    # Camera intrinsics
    focal_length: Optional[np.ndarray] = None  # [fx, fy]
    principal_point: Optional[np.ndarray] = None  # [cx, cy]
    image_width: Optional[int] = None
    image_height: Optional[int] = None


@dataclass
class IMUSample(TimestampedData):
    """IMU sensor reading with accelerometer and gyroscope data."""
    accelerometer: np.ndarray  # [ax, ay, az] in m/s²
    gyroscope: np.ndarray  # [gx, gy, gz] in rad/s
    has_valid_data: bool = True


@dataclass
class BoundingBox:
    """2D or 3D bounding box for AOI and object detection."""
    name: str
    bounds: np.ndarray  # [x, y, w, h] for 2D or [x, y, z, w, h, d] for 3D
    confidence: float = 1.0
    category: Optional[str] = None
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    @property
    def is_3d(self) -> bool:
        """Check if this is a 3D bounding box."""
        return len(self.bounds) == 6
    
    @property
    def center(self) -> np.ndarray:
        """Get center point of bounding box."""
        if self.is_3d:
            return self.bounds[:3] + self.bounds[3:] / 2
        else:
            return self.bounds[:2] + self.bounds[2:] / 2
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside this bounding box."""
        try:
            if self.is_3d and len(point) == 3:
                # Ensure bounds has 6 elements for 3D
                if len(self.bounds) < 6:
                    return False
                return (self.bounds[0] <= point[0] <= self.bounds[0] + self.bounds[3] and
                        self.bounds[1] <= point[1] <= self.bounds[1] + self.bounds[4] and
                        self.bounds[2] <= point[2] <= self.bounds[2] + self.bounds[5])
            elif not self.is_3d and len(point) == 2:
                # Ensure bounds has 4 elements for 2D
                if len(self.bounds) < 4:
                    return False
                return (self.bounds[0] <= point[0] <= self.bounds[0] + self.bounds[2] and
                        self.bounds[1] <= point[1] <= self.bounds[1] + self.bounds[3])
        except (IndexError, TypeError):
            return False
        return False


@dataclass
class Fixation:
    """Detected fixation event with spatial and temporal properties."""
    start_timestamp: int
    end_timestamp: int
    duration_ms: float
    position: np.ndarray  # Average position during fixation
    dispersion: float  # Spatial dispersion
    samples: List[GazeSample]  # Raw samples in this fixation
    
    @property
    def center(self) -> np.ndarray:
        """Get the centroid of fixation."""
        return self.position


@dataclass
class Saccade:
    """Detected saccade event with movement properties."""
    start_timestamp: int
    end_timestamp: int
    duration_ms: float
    start_position: np.ndarray
    end_position: np.ndarray
    peak_velocity: float  # degrees/second
    amplitude: float  # degrees
    samples: List[GazeSample]


# Object Detection Data Types

@dataclass
class DetectedObject(TimestampedData, SpatialEntity):
    """Enhanced 2D detected object with camera context.
    
    Inherits timestamp and camera/frame context from base classes.
    Separates detection properties from interaction metrics.
    """
    # Detection properties
    bbox: BoundingBox  # 2D bounding box in image coordinates
    class_name: str
    class_id: int
    confidence: float
    
    # Optional tracking
    instance_id: Optional[str] = None  # For tracking across frames
    
    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate detection properties."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.bbox.is_3d:  # Detection bboxes should be 2D
            raise ValueError("Detection bounding boxes must be 2D, not 3D")
        if len(self.bbox.bounds) != 4:
            raise ValueError(f"2D bounding box must have 4 elements, got {len(self.bbox.bounds)}")
    
    @property
    def object_key(self) -> ObjectKey:
        """Get ObjectKey for this detection."""
        return ObjectKey(
            camera_name=self.camera_name,
            class_name=self.class_name,
            instance_id=self.instance_id
        )


@dataclass
class Visit:
    """Individual visit to an object with precise timing."""
    start_timestamp: int  # Nanoseconds
    end_timestamp: Optional[int] = None  # Nanoseconds, None if ongoing
    gaze_samples: List[int] = field(default_factory=list)  # Sample timestamps
    fixation_sample_count: int = 0  # Number of fixation samples during this visit
    
    @property
    def duration_ns(self) -> int:
        """Get visit duration in nanoseconds."""
        if self.end_timestamp is None:
            return 0
        return self.end_timestamp - self.start_timestamp
    
    @property
    def duration_s(self) -> float:
        """Get visit duration in seconds for reports."""
        return self.duration_ns / 1e9
    
    @property
    def is_ongoing(self) -> bool:
        """Check if visit is still ongoing."""
        return self.end_timestamp is None


@dataclass 
class ObjectInteractionMetrics:
    """Enhanced metrics for gaze interaction with a specific object.
    
    Uses composite ObjectKey for proper identification and tracks
    individual visits for accurate dwell time calculations.
    """
    # Identity using composite key
    object_key: ObjectKey
    
    # Visit tracking (corrected approach)
    visits: List[Visit] = field(default_factory=list)
    
    # Derived temporal metrics (calculated from visits)
    total_dwell_ns: int = 0  # Nanoseconds - internal precision
    first_look_timestamp: Optional[int] = None  # Nanoseconds
    last_look_timestamp: Optional[int] = None   # Nanoseconds
    
    # Interaction counts (corrected)
    visit_count: int = 0  # Number of discrete visits
    total_entry_count: int = 0  # Total entries (including re-entries)
    total_exit_count: int = 0   # Total exits
    fixation_event_count: int = 0  # Number of fixation events (not samples)
    
    # Sample tracking
    total_gaze_samples: int = 0  # Total samples on this object
    gaze_state_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Extensible metadata for plugins
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived properties from visits."""
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """Recalculate all derived metrics from visit data."""
        if not self.visits:
            return
            
        # Calculate total dwell from all visits
        self.total_dwell_ns = sum(visit.duration_ns for visit in self.visits if not visit.is_ongoing)
        
        # Update visit count
        self.visit_count = len(self.visits)
        
        # Update first/last timestamps
        all_timestamps = []
        for visit in self.visits:
            all_timestamps.extend(visit.gaze_samples)
        
        if all_timestamps:
            self.first_look_timestamp = min(all_timestamps)
            self.last_look_timestamp = max(all_timestamps)
        
        # Sum fixation samples
        self.fixation_event_count = sum(visit.fixation_sample_count for visit in self.visits)
        
        # Sum total samples
        self.total_gaze_samples = sum(len(visit.gaze_samples) for visit in self.visits)
    
    @property
    def class_name(self) -> str:
        """Get class name from object key for compatibility."""
        return self.object_key.class_name
    
    @property
    def camera_name(self) -> str:
        """Get camera name from object key."""
        return self.object_key.camera_name
    
    @property
    def total_dwell_s(self) -> float:
        """Get total dwell time in seconds for reports."""
        return self.total_dwell_ns / 1e9
    
    @property
    def average_dwell_per_visit_s(self) -> float:
        """Get average dwell per visit in seconds."""
        if self.visit_count == 0:
            return 0.0
        return self.total_dwell_s / self.visit_count
    
    @property
    def revisit_rate(self) -> float:
        """Calculate revisit rate (visits after first)."""
        return max(0, self.visit_count - 1)
    
    def add_visit(self, visit: Visit):
        """Add a new visit and recalculate metrics."""
        self.visits.append(visit)
        self._recalculate_metrics()
    
    def complete_current_visit(self, end_timestamp: int):
        """Complete the ongoing visit if it exists."""
        if self.visits and self.visits[-1].is_ongoing:
            self.visits[-1].end_timestamp = end_timestamp
            self._recalculate_metrics()


@dataclass
class ObjectInstance:
    """Tracked object instance with 3D persistence and history."""
    instance_id: str  # Unique identifier across the entire session
    class_name: str
    class_id: int
    cluster_id: Optional[int] = None  # Associated 3D gaze cluster
    centroid_3d: Optional[np.ndarray] = None  # 3D spatial centroid
    bounding_boxes: List[BoundingBox] = None  # History of 2D detections
    timestamps: List[int] = None  # Timestamps when object was detected
    confidence_scores: List[float] = None  # Detection confidence over time
    
    # Interaction metrics
    total_dwell_ms: float = 0.0
    fixation_count: int = 0
    first_detection_time: Optional[int] = None
    last_detection_time: Optional[int] = None
    last_seen_timestamp: int = 0
    
    # Tracking quality
    detection_count: int = 0
    average_confidence: float = 0.0
    tracking_quality: float = 1.0  # 0-1 score for tracking consistency
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.bounding_boxes is None:
            self.bounding_boxes = []
        if self.timestamps is None:
            self.timestamps = []
        if self.confidence_scores is None:
            self.confidence_scores = []
    
    def add_detection(self, detected_obj: DetectedObject) -> None:
        """Add a new detection to this instance."""
        self.bounding_boxes.append(detected_obj.bbox)
        self.timestamps.append(detected_obj.timestamp)
        self.confidence_scores.append(detected_obj.confidence)
        self.last_seen_timestamp = detected_obj.timestamp
        self.detection_count += 1
        
        # Update first detection time
        if self.first_detection_time is None:
            self.first_detection_time = detected_obj.timestamp
        
        # Update average confidence
        self.average_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
    
    def is_active(self, current_timestamp: int, timeout_ms: int = 10000) -> bool:
        """Check if instance is still considered active."""
        if self.last_seen_timestamp == 0:
            return False
        time_since_last_seen = (current_timestamp - self.last_seen_timestamp) / 1e6  # Convert to ms
        return time_since_last_seen <= timeout_ms
    
    @property
    def duration_ms(self) -> float:
        """Get the duration this instance was tracked."""
        if self.first_detection_time is None or self.last_detection_time is None:
            return 0.0
        return (self.last_detection_time - self.first_detection_time) / 1e6


@dataclass
class GazeCluster:
    """3D cluster of gaze points with spatial and temporal properties."""
    cluster_id: int
    points_3d: np.ndarray  # N x 3 array of 3D gaze points
    timestamps: np.ndarray  # N array of timestamps
    centroid: np.ndarray  # 3D centroid position
    radius: float  # Cluster radius (distance from centroid to furthest point)
    quality_score: float  # Cluster quality metric (0-1)
    point_count: int
    density: float  # Points per cubic meter
    
    # Temporal properties
    start_timestamp: int
    end_timestamp: int
    duration_ms: float
    
    # Associated object information
    associated_class: Optional[str] = None
    associated_instance_id: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived properties."""
        if len(self.points_3d) != len(self.timestamps):
            raise ValueError("Points and timestamps must have same length")
        
        self.point_count = len(self.points_3d)
        self.start_timestamp = int(self.timestamps.min())
        self.end_timestamp = int(self.timestamps.max())
        self.duration_ms = (self.end_timestamp - self.start_timestamp) / 1e6
        
        # Calculate radius
        if self.point_count > 0:
            distances = np.linalg.norm(self.points_3d - self.centroid, axis=1)
            self.radius = float(distances.max())
            
            # Estimate density (rough approximation)
            if self.radius > 0:
                volume = (4/3) * np.pi * (self.radius ** 3)
                self.density = self.point_count / volume
            else:
                self.density = float('inf')
        else:
            self.radius = 0.0
            self.density = 0.0


# COCO Class Definitions and Utilities
# Note: These are the standard 80 COCO classes. This provides compatibility
# across different object detection models (RF-DETR, YOLO, etc.) by using
# a standardized class mapping. Models can map their outputs to these standard classes.

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Color palette for COCO classes (RGB values)
COCO_COLORS = [
    [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228],
    [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30],
    [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42],
    [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157],
    [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240],
    [0, 125, 92], [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164],
    [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0], [174, 255, 243],
    [45, 89, 255], [134, 134, 103], [145, 148, 174], [255, 208, 186], [197, 226, 255],
    [171, 134, 1], [109, 63, 54], [207, 138, 255], [151, 0, 95], [9, 80, 61],
    [84, 105, 51], [74, 65, 105], [166, 196, 102], [208, 195, 210], [255, 109, 65],
    [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0], [227, 255, 205],
    [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0], [119, 0, 170],
    [0, 182, 199], [0, 165, 120], [183, 130, 88], [95, 32, 0], [130, 114, 135],
    [110, 129, 133], [166, 74, 118], [219, 142, 185], [79, 210, 114], [178, 90, 62],
    [65, 70, 15], [127, 167, 115], [59, 105, 106], [142, 108, 45], [196, 172, 0],
    [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122], [191, 162, 208]
]


def get_coco_class_id(class_name: str) -> int:
    """Get COCO class ID from class name."""
    try:
        return COCO_CLASSES.index(class_name)
    except ValueError:
        return -1


def get_coco_class_name(class_id: int) -> str:
    """Get COCO class name from class ID."""
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return "unknown"


def get_coco_class_color(class_id: int) -> List[int]:
    """Get RGB color for COCO class."""
    if 0 <= class_id < len(COCO_COLORS):
        return COCO_COLORS[class_id]
    return [128, 128, 128]  # Default gray


def validate_detection_bbox(bbox: BoundingBox, image_width: int, image_height: int) -> bool:
    """Validate that a detection bounding box is within image bounds."""
    if bbox.is_3d:
        return False  # Detection bboxes should be 2D
    
    x, y, w, h = bbox.bounds
    return (0 <= x < image_width and 
            0 <= y < image_height and
            x + w <= image_width and
            y + h <= image_height and
            w > 0 and h > 0)


# Analytics Aggregation Classes

@dataclass
class CameraMetrics:
    """Metrics aggregated per camera for clear separation.
    
    Provides a structured way to organize all camera-specific metrics
    without mixing data across cameras. Tracks transitions within camera.
    """
    camera_name: str
    
    # Object interaction data
    object_metrics: Dict[str, ObjectInteractionMetrics] = field(default_factory=dict)
    
    # Transition tracking - corrected to show actual camera context
    transition_matrix: Dict[Tuple[str, str], int] = field(default_factory=dict)  # (from_class, to_class) -> count
    transition_details: List[Dict[str, Any]] = field(default_factory=list)  # Detailed transition records
    
    # Camera statistics
    total_gaze_samples: int = 0
    samples_on_objects: int = 0
    unique_objects: int = 0
    
    # Frame interaction data
    frame_interactions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # frame_id -> interactions
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate for this camera."""
        if self.total_gaze_samples == 0:
            return 0.0
        return self.samples_on_objects / self.total_gaze_samples
    
    def get_objects_by_class(self, class_name: str) -> List[ObjectInteractionMetrics]:
        """Get all objects of a specific class in this camera."""
        return [metrics for metrics in self.object_metrics.values() 
                if metrics.class_name == class_name]
    
    def add_transition(self, from_class: Optional[str], to_class: Optional[str], 
                      timestamp: int, from_key: Optional[str] = None, to_key: Optional[str] = None):
        """Add a transition between objects with proper camera context."""
        # Handle None cases (no object)
        from_display = from_class if from_class else "NO_OBJECT"
        to_display = to_class if to_class else "NO_OBJECT"
        
        # Update transition matrix
        key = (from_display, to_display)
        if key not in self.transition_matrix:
            self.transition_matrix[key] = 0
        self.transition_matrix[key] += 1
        
        # Store detailed transition record
        self.transition_details.append({
            "from_class": from_display,
            "to_class": to_display,
            "timestamp": timestamp,
            "camera_name": self.camera_name,  # Always this camera
            "from_object_key": from_key,
            "to_object_key": to_key
        })


@dataclass
class SessionMetrics:
    """Top-level session metrics maintaining camera separation.
    
    Provides aggregated view while keeping camera data distinct.
    """
    # Per-camera metrics (no merging)
    camera_metrics: Dict[str, CameraMetrics] = field(default_factory=dict)
    
    # Session-wide configuration and statistics
    configuration: Dict[str, Any] = field(default_factory=dict)
    global_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Session context
    session_id: str = ""
    processing_time_s: float = 0.0
    
    @property
    def total_cameras(self) -> int:
        """Get total number of cameras processed."""
        return len(self.camera_metrics)
    
    @property
    def total_gaze_samples(self) -> int:
        """Get total gaze samples across all cameras."""
        return sum(cm.total_gaze_samples for cm in self.camera_metrics.values())
    
    @property
    def total_samples_on_objects(self) -> int:
        """Get total samples on objects across all cameras."""
        return sum(cm.samples_on_objects for cm in self.camera_metrics.values())
    
    @property
    def overall_hit_rate(self) -> float:
        """Calculate overall hit rate across all cameras."""
        total_samples = self.total_gaze_samples
        if total_samples == 0:
            return 0.0
        return self.total_samples_on_objects / total_samples
    
    def get_camera_names(self) -> List[str]:
        """Get list of camera names."""
        return list(self.camera_metrics.keys())
    
    def get_all_objects(self) -> Dict[ObjectKey, ObjectInteractionMetrics]:
        """Get all objects across cameras using ObjectKey."""
        all_objects = {}
        for camera_metrics in self.camera_metrics.values():
            for metrics in camera_metrics.object_metrics.values():
                all_objects[metrics.object_key] = metrics
        return all_objects