"""Data type definitions for MLGaze Viewer sensors and analytics."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class GazeSample:
    """Single gaze sample with 3D and 2D projection data."""
    timestamp: int  # Nanoseconds
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
class CameraPose:
    """Camera position and orientation in world space."""
    timestamp: int  # Nanoseconds
    frame_id: str
    position: np.ndarray  # 3D position [x, y, z]
    rotation: np.ndarray  # Quaternion [x, y, z, w]
    
    # Camera intrinsics
    focal_length: Optional[np.ndarray] = None  # [fx, fy]
    principal_point: Optional[np.ndarray] = None  # [cx, cy]
    image_width: Optional[int] = None
    image_height: Optional[int] = None


@dataclass
class IMUSample:
    """IMU sensor reading with accelerometer and gyroscope data."""
    timestamp: int  # Nanoseconds
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
class DetectedObject:
    """2D detected object with metadata for object detection."""
    frame_id: str
    timestamp: int
    bbox: BoundingBox  # 2D bounding box in image coordinates
    class_name: str
    class_id: int
    confidence: float
    instance_id: Optional[str] = None  # For tracking across frames
    gaze_hits: int = 0  # Number of gaze samples hitting this object
    first_gaze_time: Optional[int] = None  # Timestamp of first gaze contact
    last_gaze_time: Optional[int] = None  # Timestamp of last gaze contact
    total_dwell_ms: float = 0.0  # Total time gazed at this object
    
    def __post_init__(self):
        """Validate and initialize derived properties."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.bbox.is_3d:  # Detection bboxes should be 2D
            raise ValueError("Detection bounding boxes must be 2D, not 3D")
        if len(self.bbox.bounds) != 4:
            raise ValueError(f"2D bounding box must have 4 elements, got {len(self.bbox.bounds)}")


@dataclass 
class ObjectInteractionMetrics:
    """Metrics for gaze interaction with a specific object."""
    object_id: str  # Unique identifier for the object
    class_name: str
    total_dwell_ms: float
    fixation_count: int
    first_look_timestamp: Optional[int]
    last_look_timestamp: Optional[int]
    entry_count: int  # Number of times gaze entered this object
    exit_count: int  # Number of times gaze left this object
    average_dwell_per_visit_ms: float = 0.0
    gaze_samples: List[int] = None  # Timestamps of all gaze samples
    gaze_state_distribution: Dict[str, int] = None  # Count by gaze state
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.gaze_samples is None:
            self.gaze_samples = []
        if self.gaze_state_distribution is None:
            self.gaze_state_distribution = {}
        
        # Calculate average dwell per visit
        if self.entry_count > 0:
            self.average_dwell_per_visit_ms = self.total_dwell_ms / self.entry_count


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