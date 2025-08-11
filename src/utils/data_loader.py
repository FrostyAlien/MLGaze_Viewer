"""Data loading utilities for MLGaze Viewer with memory-efficient processing."""

from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from src.core import SessionData
from src.utils.mlcf_extractor import MLCFExtractor


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
    
    def load_session(self, input_dir: str) -> SessionData:
        """Load all sensor data from a session directory.
        
        Args:
            input_dir: Path to the input directory containing data files
            
        Returns:
            SessionData object containing all loaded data
            
        Raises:
            FileNotFoundError: If required files are not found
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if self.verbose:
            print(f"Loading session data from: {input_path}")
        
        # Discover data files
        gaze_csv, metadata_csv, frames_dir, imu_csv = self._discover_files(input_path)
        
        # Load data components
        gaze_df = self._load_gaze_data(gaze_csv)
        metadata_df = self._load_metadata(metadata_csv)
        frames = self._load_frames(frames_dir)
        
        # Extract camera poses from gaze data (which contains camera info)
        camera_poses_df = self._extract_camera_poses(gaze_df)
        
        # Load optional IMU data
        imu_df = None
        if imu_csv:
            imu_df = self._load_imu_data(imu_csv)
        
        # Create session ID from directory name
        session_id = input_path.name
        
        # Create and return SessionData
        session = SessionData(
            gaze=gaze_df,
            frames=frames,
            camera_poses=camera_poses_df,
            metadata=metadata_df,
            imu=imu_df,
            session_id=session_id,
            input_directory=str(input_path)
        )
        
        if self.verbose:
            print(f"\n{session.summary()}")
        
        return session
    
    def _discover_files(self, input_dir: Path) -> Tuple[Path, Path, Path, Optional[Path]]:
        """Automatically discover data files in the input directory.
        
        Searches recursively through subdirectories and handles MLCF extraction if needed.
        
        Returns:
            Tuple of (gaze_csv, metadata_csv, frames_dir, imu_csv)
        """
        # Search recursively for gaze screen coordinates CSV
        gaze_files = list(input_dir.rglob("*gaze_screen_coords*.csv"))
        if not gaze_files:
            raise FileNotFoundError(f"No gaze screen coordinates CSV found in {input_dir}")
        gaze_csv = gaze_files[0]
        
        # Search recursively for frame metadata CSV
        metadata_files = list(input_dir.rglob("frame_metadata.csv"))
        if not metadata_files:
            raise FileNotFoundError(f"No frame_metadata.csv found in {input_dir}")
        metadata_csv = metadata_files[0]
        
        # Determine frames location - prefer same directory as metadata
        metadata_parent = metadata_csv.parent
        frames_dir = metadata_parent / "frames"
        
        # If frames directory doesn't exist, check for MLCF file
        if not frames_dir.exists():
            mlcf_files = list(metadata_parent.glob("*.mlcf"))
            if not mlcf_files:
                # Try searching in parent directories
                mlcf_files = list(input_dir.rglob("*.mlcf"))
            
            if mlcf_files:
                mlcf_file = mlcf_files[0]
                if self.verbose:
                    print(f"No frames directory found, but found MLCF file: {mlcf_file.name}")
                    print(f"Extracting frames from MLCF container...")
                
                # Extract frames from MLCF
                extractor = MLCFExtractor(verbose=self.verbose)
                success = extractor.extract(mlcf_file, frames_dir)
                
                if not success:
                    raise RuntimeError(f"Failed to extract frames from {mlcf_file}")
            else:
                raise FileNotFoundError(f"No frames directory or MLCF file found in {input_dir}")
        
        # Search recursively for IMU log CSV (optional)
        imu_files = list(input_dir.rglob("*imu_log*.csv"))
        imu_csv = imu_files[0] if imu_files else None
        
        if self.verbose:
            print(f"Discovered files:")
            print(f"  Gaze: {gaze_csv.relative_to(input_dir)}")
            print(f"  Metadata: {metadata_csv.relative_to(input_dir)}")
            print(f"  Frames: {frames_dir.relative_to(input_dir)}/")
            if imu_csv:
                print(f"  IMU: {imu_csv.relative_to(input_dir)}")
        
        return gaze_csv, metadata_csv, frames_dir, imu_csv
    
    def _load_gaze_data(self, csv_path: Path) -> pd.DataFrame:
        """Load gaze data from CSV with proper data types."""
        if self.verbose:
            print(f"Loading gaze data from {csv_path.name}...")
        
        # For large files, use chunked reading
        file_size = csv_path.stat().st_size / (1024 * 1024)  # Size in MB
        
        if file_size > 100:  # If file is larger than 100MB
            if self.verbose:
                print(f"  Large file ({file_size:.1f} MB), using chunked loading...")
            
            chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=self.chunk_size * 100):
                chunk['timestamp'] = chunk['timestamp'].astype(np.int64)
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(csv_path)
            df['timestamp'] = df['timestamp'].astype(np.int64)
        
        if self.verbose:
            print(f"  Loaded {len(df)} gaze samples")
        
        return df
    
    def _load_metadata(self, csv_path: Path) -> pd.DataFrame:
        """Load frame metadata from CSV."""
        if self.verbose:
            print(f"Loading frame metadata...")
        
        df = pd.read_csv(csv_path)
        df['timestamp'] = df['timestamp'].astype(np.int64)
        
        if self.verbose:
            print(f"  Loaded metadata for {len(df)} frames")
        
        return df
    
    def _load_frames(self, frames_dir: Path) -> Dict[str, bytes]:
        """Load frame images as compressed JPEG bytes for memory efficiency."""
        if self.verbose:
            print(f"Loading frames from {frames_dir.name}/ (compressed)...")
        
        frames = {}
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        
        for i, frame_path in enumerate(frame_files):
            frame_id = frame_path.stem  # e.g., "frame_000001"
            
            # Load as compressed bytes to save memory
            with open(frame_path, 'rb') as f:
                frames[frame_id] = f.read()
            
            if self.verbose and (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(frame_files)} frames...")
        
        if self.verbose:
            print(f"  Loaded {len(frames)} frames into memory")
        
        return frames
    
    def _load_imu_data(self, csv_path: Path) -> pd.DataFrame:
        """Load IMU sensor data from CSV."""
        if self.verbose:
            print(f"Loading IMU data from {csv_path.name}...")
        
        df = pd.read_csv(csv_path)
        df['timestamp'] = df['timestamp'].astype(np.int64)
        
        # Filter valid data if column exists
        if 'hasValidData' in df.columns:
            valid_df = df[df['hasValidData'] == True].copy()
            if self.verbose:
                print(f"  Loaded {len(df)} IMU samples ({len(valid_df)} valid)")
            return valid_df
        
        if self.verbose:
            print(f"  Loaded {len(df)} IMU samples")
        
        return df
    
    def _extract_camera_poses(self, gaze_df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique camera poses from gaze data."""
        if self.verbose:
            print("Extracting camera poses from gaze data...")
        
        # Camera pose columns in gaze data
        camera_columns = [
            'timestamp', 'frameId',
            'cameraPositionX', 'cameraPositionY', 'cameraPositionZ',
            'cameraRotationX', 'cameraRotationY', 'cameraRotationZ', 'cameraRotationW'
        ]
        
        # Check which columns exist
        existing_columns = [col for col in camera_columns if col in gaze_df.columns]
        
        if len(existing_columns) < 6:  # Need at least position and some rotation
            if self.verbose:
                print("  Warning: Incomplete camera data in gaze file")
            return pd.DataFrame()
        
        # Extract unique camera poses (one per frame)
        camera_df = gaze_df[existing_columns].drop_duplicates(subset=['frameId']).copy()
        camera_df = camera_df.sort_values('timestamp').reset_index(drop=True)
        
        if self.verbose:
            print(f"  Extracted {len(camera_df)} unique camera poses")
        
        return camera_df
    
    def load_frames_lazy(self, frames_dir: Path) -> Dict[str, Path]:
        """Load frame paths only, not the actual image data (for very large datasets).
        
        Returns:
            Dictionary mapping frame IDs to file paths
        """
        if self.verbose:
            print(f"Loading frame paths from {frames_dir.name}/ (lazy mode)...")
        
        frame_paths = {}
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        
        for frame_path in frame_files:
            frame_id = frame_path.stem
            frame_paths[frame_id] = frame_path
        
        if self.verbose:
            print(f"  Found {len(frame_paths)} frame files")
        
        return frame_paths