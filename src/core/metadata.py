"""Session metadata structures for MLGaze Viewer."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class CameraInfo:
    """Information about a camera in the session."""
    name: str
    type: str  # RGB, depth, MR
    resolution: str
    fps: int
    has_gaze_screen_coords: bool
    format: str  # MLCF
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraInfo':
        """Create CameraInfo from dictionary."""
        return cls(
            name=data['name'],
            type=data['type'],
            resolution=data['resolution'],
            fps=data['fps'],
            has_gaze_screen_coords=data.get('hasGazeScreenCoords', False),
            format=data.get('format', 'MLCF')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CameraInfo to dictionary."""
        return {
            'name': self.name,
            'type': self.type,
            'resolution': self.resolution,
            'fps': self.fps,
            'hasGazeScreenCoords': self.has_gaze_screen_coords,
            'format': self.format
        }


@dataclass
class DeviceInfo:
    """Device information from the session."""
    model: str
    os_version: str
    device_id: str
    platform: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceInfo':
        """Create DeviceInfo from dictionary."""
        return cls(
            model=data.get('model', 'Unknown'),
            os_version=data.get('osVersion', 'Unknown'),
            device_id=data.get('deviceId', 'Unknown'),
            platform=data.get('platform', 'Unknown')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DeviceInfo to dictionary."""
        return {
            'model': self.model,
            'osVersion': self.os_version,
            'deviceId': self.device_id,
            'platform': self.platform
        }


@dataclass
class SessionMetadata:
    """Complete metadata for a recording session."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    cameras: List[CameraInfo] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)
    device_info: Optional[DeviceInfo] = None
    app_version: str = "Unknown"
    unity_version: str = "Unknown"
    
    @classmethod
    def from_json_file(cls, json_path: Path) -> 'SessionMetadata':
        """Load SessionMetadata from metadata.json file.
        
        Args:
            json_path: Path to metadata.json file
            
        Returns:
            SessionMetadata object
            
        Raises:
            FileNotFoundError: If metadata.json doesn't exist
            ValueError: If JSON is invalid or missing required fields
        """
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in metadata file: {e}")
        
        # Parse cameras
        cameras = []
        for cam_data in data.get('cameras', []):
            cameras.append(CameraInfo.from_dict(cam_data))
        
        # Parse device info
        device_info = None
        if 'deviceInfo' in data:
            device_info = DeviceInfo.from_dict(data['deviceInfo'])
        
        return cls(
            session_id=data.get('sessionId', 'Unknown'),
            start_time=data.get('startTime', 'Unknown'),
            end_time=data.get('endTime'),
            cameras=cameras,
            sensors=data.get('sensors', []),
            device_info=device_info,
            app_version=data.get('appVersion', 'Unknown'),
            unity_version=data.get('unityVersion', 'Unknown')
        )
    
    @classmethod
    def create_minimal(cls, session_id: str, cameras: List[str]) -> 'SessionMetadata':
        """Create minimal metadata when metadata.json is missing.
        
        Args:
            session_id: Session identifier
            cameras: List of camera names discovered
            
        Returns:
            SessionMetadata with basic information
        """
        camera_infos = []
        for cam_name in cameras:
            camera_infos.append(CameraInfo(
                name=cam_name,
                type="Unknown",
                resolution="Unknown",
                fps=30,  # Default assumption
                has_gaze_screen_coords=True,  # Assume yes
                format="MLCF"
            ))
        
        return cls(
            session_id=session_id,
            start_time="Unknown",
            cameras=camera_infos,
            sensors=["gaze"]  # Minimum expected
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SessionMetadata to dictionary for JSON serialization."""
        return {
            'sessionId': self.session_id,
            'startTime': self.start_time,
            'endTime': self.end_time,
            'cameras': [cam.to_dict() for cam in self.cameras],
            'sensors': self.sensors,
            'deviceInfo': self.device_info.to_dict() if self.device_info else None,
            'appVersion': self.app_version,
            'unityVersion': self.unity_version
        }
    
    def save_to_file(self, json_path: Path) -> None:
        """Save SessionMetadata to JSON file.
        
        Args:
            json_path: Path where to save metadata.json
        """
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_camera_names(self) -> List[str]:
        """Get list of camera names in the session."""
        return [cam.name for cam in self.cameras]
    
    def get_camera_info(self, camera_name: str) -> Optional[CameraInfo]:
        """Get CameraInfo for a specific camera.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            CameraInfo if found, None otherwise
        """
        for cam in self.cameras:
            if cam.name == camera_name:
                return cam
        return None
    
    def has_gaze_screen_coords(self, camera_name: str) -> bool:
        """Check if a camera has gaze screen coordinates.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            True if camera has gaze screen coordinates
        """
        cam_info = self.get_camera_info(camera_name)
        return cam_info.has_gaze_screen_coords if cam_info else False
    
    def summary(self) -> str:
        """Generate a human-readable summary of the session metadata."""
        lines = [
            f"Session: {self.session_id}",
            f"Start Time: {self.start_time}",
            f"Cameras: {len(self.cameras)}",
            f"Sensors: {', '.join(self.sensors)}"
        ]
        
        if self.cameras:
            lines.append("Camera Details:")
            for cam in self.cameras:
                gaze_status = "with gaze" if cam.has_gaze_screen_coords else "no gaze"
                lines.append(f"  - {cam.name} ({cam.type}): {cam.resolution} @ {cam.fps}fps ({gaze_status})")
        
        if self.device_info:
            lines.append(f"Device: {self.device_info.model} ({self.device_info.platform})")
        
        return "\n".join(lines)