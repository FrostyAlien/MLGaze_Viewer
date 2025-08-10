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