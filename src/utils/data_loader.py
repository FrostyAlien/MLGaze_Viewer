"""Data loading utilities for MLGaze Viewer with memory-efficient processing."""

from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from src.core import SessionData, SessionMetadata
from src.utils.mlcf_extractor import MLCFExtractor
from src.utils.logger import logger


class DataLoader:
    """Unified data loader for MLGaze sessions with memory optimization."""
    
    def __init__(self, chunk_size: int = 1000, verbose: bool = True):
        """Initialize the data loader.
        
        Args:
            chunk_size: Number of samples to process at once for large files
            verbose: Whether to print loading progress
        """
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.log = logger.get_logger('DataLoader')
    
    def load_session(self, session_dir: str, config: Dict = None) -> SessionData:
        """Load all sensor data from an organized session directory.
        
        Expected structure:
        session_dir/
        ├── metadata.json                    # Session information
        ├── cameras/
        │   ├── camera_name/                 # One or more cameras
        │   │   ├── camera_frames.mlcf       # Binary frame container
        │   │   ├── frame_metadata.csv       # Frame timestamps & poses
        │   │   └── gaze_screen_coords.csv   # Optional: Gaze in camera coordinates
        │   └── ...
        └── sensors/
            ├── gaze_data.csv                # 3D world gaze data
            └── imu_data.csv                 # Optional: IMU sensor data
        
        Args:
            session_dir: Path to the organized session directory
            
        Returns:
            SessionData object containing all loaded data
            
        Raises:
            FileNotFoundError: If required files or directories are not found
            ValueError: If session structure is invalid
        """
        session_path = Path(session_dir)
        if not session_path.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        self.log.info(f"Loading organized session from: {session_path}")
        
        # Validate session structure
        self._validate_session_structure(session_path)
        
        # Load session metadata
        metadata = self._load_session_metadata(session_path)
        
        # Load all camera data
        cameras_data = self._load_all_cameras(session_path / "cameras")
        
        # Load sensor data
        gaze_3d = self._load_csv(session_path / "sensors" / "gaze_data.csv", "3D gaze data")
        
        # Validate 3D gaze data columns
        required_gaze_3d_cols = ['timestamp', 'gazeOriginX', 'gazeOriginY', 'gazeOriginZ',
                               'gazePositionX', 'gazePositionY', 'gazePositionZ',
                               'isTracking', 'hasHitTarget', 'gazeState']
        self._validate_csv_columns(gaze_3d, required_gaze_3d_cols, "sensors/gaze_data.csv")
        
        imu = self._load_csv_optional(session_path / "sensors" / "imu_data.csv", "IMU data")
        
        # Determine primary camera (first one alphabetically by default)
        primary_camera = sorted(cameras_data['frames'].keys())[0] if cameras_data['frames'] else ""
        
        # Create SessionData
        session = SessionData(
            frames=cameras_data['frames'],
            camera_metadata=cameras_data['metadata'],
            gaze_screen_coords=cameras_data['gaze_coords'],
            gaze=gaze_3d,
            imu=imu,
            metadata=metadata,
            primary_camera=primary_camera,
            session_id=metadata.session_id if metadata else session_path.name,
            input_directory=str(session_path),
            config=config or {}
        )
        
        self.log.info(f"Session loaded successfully:\n{session.summary()}")
        
        return session
    
    def _validate_session_structure(self, session_path: Path) -> None:
        """Validate that the session directory has the required structure.
        
        Args:
            session_path: Path to the session directory
            
        Raises:
            FileNotFoundError: If required directories or files are missing
            ValueError: If session structure is invalid
        """
        # Check required directories
        cameras_dir = session_path / "cameras"
        sensors_dir = session_path / "sensors"
        
        required_dirs = [cameras_dir, sensors_dir]
        missing_dirs = [str(d) for d in required_dirs if not d.exists()]
        
        if missing_dirs:
            raise FileNotFoundError(f"Required directories missing: {missing_dirs}")
        
        # Check for at least one camera directory
        camera_dirs = [d for d in cameras_dir.iterdir() if d.is_dir()]
        if not camera_dirs:
            raise ValueError("No camera directories found in cameras/")
        
        # Check required sensor files
        gaze_data_file = sensors_dir / "gaze_data.csv"
        if not gaze_data_file.exists():
            raise FileNotFoundError(f"Required file missing: {gaze_data_file}")
        
        # Validate each camera directory
        for camera_dir in camera_dirs:
            frame_metadata = camera_dir / "frame_metadata.csv"
            if not frame_metadata.exists():
                raise FileNotFoundError(f"Camera {camera_dir.name} missing frame_metadata.csv")
            
            # Check for frames (either directory or MLCF file)
            frames_dir = camera_dir / "frames"
            mlcf_file = camera_dir / "camera_frames.mlcf"
            
            if not frames_dir.exists() and not mlcf_file.exists():
                raise FileNotFoundError(
                    f"Camera {camera_dir.name} missing both frames/ directory and camera_frames.mlcf"
                )
        
        self.log.info(f"Session structure validated: {len(camera_dirs)} cameras found")
    
    
    def _load_session_metadata(self, session_path: Path) -> Optional[SessionMetadata]:
        """Load session metadata from metadata.json file.
        
        Args:
            session_path: Path to the session directory
            
        Returns:
            SessionMetadata object or None if file doesn't exist
        """
        metadata_file = session_path / "metadata.json"
        
        if metadata_file.exists():
            try:
                return SessionMetadata.from_json_file(metadata_file)
            except (FileNotFoundError, ValueError) as e:
                self.log.warning(f"Failed to load metadata.json: {e}")
                # Fall through to create minimal metadata
        
        # Create minimal metadata if file doesn't exist or is invalid
        self.log.info("Creating minimal metadata from directory structure")
        
        camera_names = []
        cameras_dir = session_path / "cameras"
        if cameras_dir.exists():
            camera_names = [d.name for d in cameras_dir.iterdir() if d.is_dir()]
        
        return SessionMetadata.create_minimal(session_path.name, camera_names)
    
    def _load_all_cameras(self, cameras_dir: Path) -> Dict:
        """Load data for all cameras in the cameras directory.
        
        Args:
            cameras_dir: Path to the cameras directory
            
        Returns:
            Dictionary with 'frames', 'metadata', and 'gaze_coords' keys
        """
        self.log.info(f"Loading data for all cameras...")
        
        all_frames = {}
        all_metadata = {}
        all_gaze_coords = {}
        
        camera_dirs = [d for d in cameras_dir.iterdir() if d.is_dir()]
        
        for camera_dir in sorted(camera_dirs):
            camera_name = camera_dir.name
            self.log.info(f"  Loading camera: {camera_name}")
            
            # Load frame metadata
            metadata_file = camera_dir / "frame_metadata.csv"
            metadata_df = self._load_csv(metadata_file, f"metadata for {camera_name}")
            
            # Validate required columns for frame metadata
            required_metadata_cols = ['frameId', 'timestamp', 'posX', 'posY', 'posZ', 
                                    'rotX', 'rotY', 'rotZ', 'rotW']
            self._validate_csv_columns(metadata_df, required_metadata_cols, 
                                     f"{camera_name}/frame_metadata.csv")
            
            all_metadata[camera_name] = metadata_df
            
            # Load frames (either from directory or extract from MLCF)
            frames = self._load_camera_frames(camera_dir)
            all_frames[camera_name] = frames
            
            # Load gaze screen coordinates if available
            gaze_coords_file = camera_dir / "gaze_screen_coords.csv"
            gaze_coords_df = self._load_csv_optional(gaze_coords_file, f"gaze coords for {camera_name}")
            
            # Validate gaze coordinates if present
            if gaze_coords_df is not None:
                required_gaze_cols = ['timestamp', 'frameId', 'screenPixelX', 'screenPixelY', 
                                    'isTracking', 'gazeState']
                self._validate_csv_columns(gaze_coords_df, required_gaze_cols, 
                                         f"{camera_name}/gaze_screen_coords.csv")
            
            all_gaze_coords[camera_name] = gaze_coords_df
        
        return {
            'frames': all_frames,
            'metadata': all_metadata,
            'gaze_coords': all_gaze_coords
        }
    
    def _load_camera_frames(self, camera_dir: Path) -> Dict[str, bytes]:
        """Load frames for a single camera (from directory or MLCF extraction).
        
        Args:
            camera_dir: Path to the camera directory
            
        Returns:
            Dictionary mapping frame IDs to JPEG bytes
        """
        frames_dir = camera_dir / "frames"
        mlcf_file = camera_dir / "camera_frames.mlcf"
        
        # If frames directory exists, load from it
        if frames_dir.exists():
            return self._load_frames_from_directory(frames_dir)
        
        # If MLCF file exists, extract frames first
        elif mlcf_file.exists():
            self.log.info(f"  Extracting frames from {mlcf_file.name}...")
            
            extractor = MLCFExtractor(verbose=self.verbose)
            success = extractor.extract(mlcf_file, frames_dir)
            
            if not success:
                self.log.error(f"Failed to extract frames from {mlcf_file}")
                return {}
            
            self.log.info(f"  MLCF extraction completed successfully")
            return self._load_frames_from_directory(frames_dir)
        
        else:
            self.log.warning(f"No frames or MLCF file found for camera {camera_dir.name}")
            return {}
    
    def _load_frames_from_directory(self, frames_dir: Path) -> Dict[str, bytes]:
        """Load frame images from a directory as compressed JPEG bytes.
        
        Args:
            frames_dir: Path to directory containing frame_*.jpg files
            
        Returns:
            Dictionary mapping frame IDs to JPEG bytes
        """
        self.log.debug(f"  Loading frames from {frames_dir.name}/ (compressed)")
        
        frames = {}
        # Look for both "frame_*.jpg" and "*frame_*.jpg" patterns to handle camera prefixes
        frame_files = sorted(list(frames_dir.glob("frame_*.jpg")) + list(frames_dir.glob("*frame_*.jpg")))
        
        for i, frame_path in enumerate(frame_files):
            # Use the full filename as frame_id to match metadata CSV format
            # (e.g., "CV_Real_002_frame_000001" matches frameId in CSV)
            frame_id = frame_path.stem
            
            # Load as compressed bytes to save memory
            with open(frame_path, 'rb') as f:
                frames[frame_id] = f.read()
            
            if self.verbose and (i + 1) % 100 == 0:
                self.log.debug(f"    Loaded {i + 1}/{len(frame_files)} frames...")
        
        self.log.debug(f"  Loaded {len(frames)} frames into memory")
        return frames
    
    def _load_csv(self, csv_path: Path, description: str) -> pd.DataFrame:
        """Load CSV data with proper data types and chunking for large files.
        
        Args:
            csv_path: Path to CSV file
            description: Description for logging
            
        Returns:
            DataFrame with timestamp column as int64
        """
        self.log.debug(f"  Loading {description} from {csv_path.name}...")
        
        # Check file size for chunked loading
        file_size = csv_path.stat().st_size / (1024 * 1024)  # Size in MB
        
        if file_size > 100:  # If file is larger than 100MB
            self.log.info(f"    Large file ({file_size:.1f} MB), using chunked loading...")
            
            chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=self.chunk_size * 100):
                if 'timestamp' in chunk.columns:
                    chunk['timestamp'] = chunk['timestamp'].astype(np.int64)
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(csv_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].astype(np.int64)
        
        self.log.debug(f"    Loaded {len(df)} rows")
        return df
    
    def _load_csv_optional(self, csv_path: Path, description: str) -> Optional[pd.DataFrame]:
        """Load optional CSV data.
        
        Args:
            csv_path: Path to CSV file
            description: Description for logging
            
        Returns:
            DataFrame if file exists, None otherwise
        """
        if csv_path.exists():
            return self._load_csv(csv_path, description)
        else:
            self.log.debug(f"  {description} not found (optional): {csv_path.name}")
            return None
    
    def _validate_csv_columns(self, df: pd.DataFrame, required_columns: List[str], 
                             csv_name: str) -> None:
        """Validate that CSV contains required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            csv_name: Name of CSV file for error reporting
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            available_cols = list(df.columns)
            raise ValueError(
                f"Missing required columns in {csv_name}: {missing_columns}\n"
                f"Available columns: {available_cols}\n"
                f"Please check that your data files match the expected format."
            )