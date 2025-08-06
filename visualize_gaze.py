#!/usr/bin/env python3
"""
Magic Leap 2 Gaze Visualizer using Rerun
Visualizes eye gaze data in both 3D world space and 2D camera frames
"""

import argparse
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import rerun as rr
from typing import Dict, List, Tuple, Optional

from ui.config_app import run_configuration_tui, VisualizationConfig


def quaternion_rotate_vector(q: List[float], v: List[float]) -> List[float]:
    """Rotate a 3D vector by a quaternion.
    
    Args:
        q: Quaternion in XYZW format [x, y, z, w]
        v: 3D vector to rotate [x, y, z]
        
    Returns:
        Rotated 3D vector [x, y, z]
    """
    qx, qy, qz, qw = q
    vx, vy, vz = v

    # Convert to standard quaternion rotation formula
    # v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)
    q_xyz = np.array([qx, qy, qz])
    v_arr = np.array([vx, vy, vz])

    cross1 = np.cross(q_xyz, v_arr)
    cross2 = np.cross(q_xyz, cross1 + qw * v_arr)
    result = v_arr + 2 * cross2

    return result.tolist()


def quaternion_to_matrix(q: List[float]) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.
    
    Args:
        q: Quaternion in XYZW format [x, y, z, w]
        
    Returns:
        3x3 rotation matrix as numpy array
    """
    x, y, z, w = q

    # Normalize quaternion
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm > 0:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Build rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

    return matrix


def matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    """Convert 3x3 rotation matrix to quaternion.
    
    Args:
        matrix: 3x3 rotation matrix as numpy array
        
    Returns:
        Quaternion in XYZW format [x, y, z, w]
    """
    m = matrix

    # Based on method from Shepperd (1978)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return [float(x), float(y), float(z), float(w)]


def compose_quaternions(q1: List[float], q2: List[float]) -> List[float]:
    """Compose (multiply) two quaternions: result = q1 * q2.
    
    Args:
        q1: First quaternion in XYZW format [x, y, z, w]
        q2: Second quaternion in XYZW format [x, y, z, w]
        
    Returns:
        Composed quaternion in XYZW format [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return [x, y, z, w]


def unity_to_rerun_position(pos: List[float]) -> List[float]:
    """Convert Unity left-handed Y-up to Rerun RDF (Right-Down-Forward).
    Unity: X=right, Y=up, Z=forward (left-handed)
    RDF: X=right, Y=down, Z=forward (right-handed)
    
    Need to:
    1. Flip Y-axis: Y_rdf = -Y_unity (up to down)
    2. Keep Z-axis: Z_rdf = Z_unity (forward stays forward)
    """
    return [pos[0], -pos[1], pos[2]]


def unity_to_rerun_quaternion(q: List[float]) -> List[float]:
    """Convert Unity left-handed Y-up quaternion to Rerun RDF coordinate system.
    Unity: X=right, Y=up, Z=forward (left-handed)
    RDF: X=right, Y=down, Z=forward (right-handed)
    
    This requires:
    1. Converting the rotation from Unity's coordinate system
    2. Handling the change from left-handed to right-handed system
    """
    # Convert quaternion to rotation matrix
    rot_matrix = quaternion_to_matrix(q)

    # Unity to RDF coordinate transformation
    unity_to_rdf = np.array([
        [1, 0, 0],   # X stays right
        [0, -1, 0],  # Y flips (up to down)
        [0, 0, 1]    # Z stays forward
    ])

    # For rotation matrices, the transformation is: R_rdf = T * R_unity * T^(-1)
    # Calculate T^(-1) (which happens to equal T in this case)
    unity_to_rdf_inv = unity_to_rdf.T  # For orthogonal matrices, inverse = transpose

    # Apply the transformation
    transformed_matrix = unity_to_rdf @ rot_matrix @ unity_to_rdf_inv

    # Convert back to quaternion
    result_q = matrix_to_quaternion(transformed_matrix)

    # Ensure proper conversion to list of floats
    return [float(result_q[0]), float(result_q[1]), float(result_q[2]), float(result_q[3])]


def load_gaze_screen_coords(csv_path: Path) -> pd.DataFrame:
    """Load gaze screen coordinates data from CSV file.
    
    Args:
        csv_path: Path to the gaze screen coordinates CSV file
        
    Returns:
        DataFrame with gaze data including timestamps, positions, and screen projections
    """
    print(f"Loading gaze screen coordinates from {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert timestamp to int64 for nanosecond precision
    df['timestamp'] = df['timestamp'].astype(np.int64)

    print(f"Loaded {len(df)} gaze samples")
    return df


def load_frame_metadata(csv_path: Path) -> pd.DataFrame:
    """Load camera frame metadata from CSV file.

    Args:
        csv_path: Path to the frame metadata CSV file

    Returns:
        DataFrame with camera intrinsics and frame information
    """
    print(f"Loading frame metadata from {csv_path}")
    df = pd.read_csv(csv_path)
    df['timestamp'] = df['timestamp'].astype(np.int64)
    print(f"Loaded metadata for {len(df)} frames")
    return df


def load_imu_data(csv_path: Path) -> pd.DataFrame:
    """Load IMU sensor data from CSV file.
    
    Args:
        csv_path: Path to the IMU log CSV file
        
    Returns:
        DataFrame with IMU data including timestamps, accelerometer, and gyroscope readings
    """
    print(f"Loading IMU data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to int64 for nanosecond precision
    df['timestamp'] = df['timestamp'].astype(np.int64)
    
    # Filter out invalid data
    valid_data = df[df['hasValidData'] == True].copy() if 'hasValidData' in df.columns else df.copy()
    
    print(f"Loaded {len(df)} IMU samples ({len(valid_data)} valid)")
    return valid_data


def load_frame_compressed(frame_path: Path) -> bytes:
    """Load a single JPEG frame as compressed bytes for memory efficiency.
    
    Args:
        frame_path: Path to the JPEG frame file
        
    Returns:
        Compressed JPEG bytes
    """
    with open(frame_path, 'rb') as f:
        return f.read()




def load_all_frames(frames_dir: Path) -> Dict[str, bytes]:
    """Load frame images as compressed JPEG bytes for memory-efficient visualization.
    
    Args:
        frames_dir: Directory containing frame_*.jpg files
        
    Returns:
        Dictionary mapping frame IDs to compressed JPEG bytes
    """
    print(f"Loading frames from {frames_dir} (compressed)")
    frames = {}

    # Get all jpg files
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    for i, frame_path in enumerate(frame_files):
        frame_id = frame_path.stem  # e.g., "frame_000001"
        frames[frame_id] = load_frame_compressed(frame_path)

        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1}/{len(frame_files)} frames...")

    print(f"Loaded {len(frames)} frames into memory")
    return frames


def get_gaze_color(gaze_state: str) -> List[int]:
    """Get RGB color for different gaze states.

    Args:
        gaze_state: Gaze state string (e.g., 'Fixation', 'Saccade', 'Pursuit')

    Returns:
        RGB color as [R, G, B] values (0-255)
    """
    colors = {
        'Fixation': [0, 255, 0],  # Green
        'Saccade': [255, 255, 0],  # Yellow
        'Pursuit': [0, 255, 255],  # Cyan
        'WinkRight': [255, 0, 255],  # Magenta
        'WinkLeft': [255, 0, 255],  # Magenta
    }
    return colors.get(gaze_state, [128, 128, 128])  # Gray for unknown


def log_coordinate_indicators(config: VisualizationConfig):
    """Log coordinate system indicators at world origin.
    
    Creates three colored arrows at the world origin to visualize the
    Rerun RDF coordinate system (X=right, Y=down, Z=forward).
    """
    if not config.show_coordinate_indicators:
        return
    
    print("Adding coordinate system indicators at world origin")
    
    # RDF coordinate system: X=right, Y=down, Z=forward
    rr.log(
        "world/coords/origin",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # RGB = XYZ
            radii=0.002,
        ),
        static=True
    )


def apply_imu_axis_styling(entity_path: str, axis: str, color: List[int]) -> None:
    """Apply SeriesLines styling to IMU axis with proper color.
    
    Args:
        entity_path: Rerun entity path for the data
        axis: Axis name (X, Y, Z)
        color: RGB color values [R, G, B]
    """
    try:
        rr.log(entity_path, rr.SeriesLines(colors=color, names=axis), static=True)
    except Exception as e:
        print(f"Warning: Could not apply styling to {entity_path}: {e}")




def log_imu_data(imu_df: pd.DataFrame) -> None:
    """Log IMU sensor data to Rerun using efficient columnar logging.
    
    Args:
        imu_df: DataFrame containing IMU data with timestamp and sensor readings
    """
    if imu_df.empty:
        print("No IMU data to log")
        return
    
    print(f"Logging {len(imu_df)} IMU samples using columnar method...")
    
    # Create time column for efficient logging (convert nanoseconds to seconds)
    times = rr.TimeColumn("timestamp", timestamp=imu_df["timestamp"] * 1e-9)
    
    # Define clear, distinguishable colors for X/Y/Z axes
    axis_colors = {
        'X': [255, 80, 80],   # Bright Red - clear and distinct
        'Y': [80, 200, 80],   # Bright Green - high contrast from red
        'Z': [80, 120, 255]   # Bright Blue - distinct from red/green
    }
    
    # Log accelerometer data with separate axes for proper naming
    rr.send_columns(
        "/sensors/imu/accelerometer/X",
        indexes=[times],
        columns=rr.Scalars.columns(scalars=imu_df["accelX"].values)
    )
    apply_imu_axis_styling("/sensors/imu/accelerometer/X", "X", axis_colors['X'])
    
    rr.send_columns(
        "/sensors/imu/accelerometer/Y",
        indexes=[times],
        columns=rr.Scalars.columns(scalars=imu_df["accelY"].values)
    )
    apply_imu_axis_styling("/sensors/imu/accelerometer/Y", "Y", axis_colors['Y'])
    
    rr.send_columns(
        "/sensors/imu/accelerometer/Z",
        indexes=[times],
        columns=rr.Scalars.columns(scalars=imu_df["accelZ"].values)
    )
    apply_imu_axis_styling("/sensors/imu/accelerometer/Z", "Z", axis_colors['Z'])
    
    # Log gyroscope data with separate axes for proper naming (same colors for consistency)
    rr.send_columns(
        "/sensors/imu/gyroscope/X",
        indexes=[times],
        columns=rr.Scalars.columns(scalars=imu_df["gyroX"].values)
    )
    apply_imu_axis_styling("/sensors/imu/gyroscope/X", "X", axis_colors['X'])
    
    rr.send_columns(
        "/sensors/imu/gyroscope/Y",
        indexes=[times],
        columns=rr.Scalars.columns(scalars=imu_df["gyroY"].values)
    )
    apply_imu_axis_styling("/sensors/imu/gyroscope/Y", "Y", axis_colors['Y'])
    
    rr.send_columns(
        "/sensors/imu/gyroscope/Z",
        indexes=[times],
        columns=rr.Scalars.columns(scalars=imu_df["gyroZ"].values)
    )
    apply_imu_axis_styling("/sensors/imu/gyroscope/Z", "Z", axis_colors['Z'])
    
    print(f"  Accelerometer: {len(imu_df)} samples to /sensors/imu/accelerometer/[X,Y,Z]")
    print(f"  Gyroscope: {len(imu_df)} samples to /sensors/imu/gyroscope/[X,Y,Z]")



def discover_data_files(input_dir: Path) -> Tuple[Path, Path, Path, Optional[Path]]:
    """Automatically discover gaze data files in the input directory.

    Args:
        input_dir: Input directory to search for data files

    Returns:
        Tuple of (gaze_csv_path, metadata_csv_path, frames_dir_path, imu_csv_path)
        imu_csv_path will be None if no IMU data is found

    Raises:
        FileNotFoundError: If required files cannot be found
    """
    input_dir = Path(input_dir)

    # Find gaze screen coordinates CSV (contains "gaze_screen_coords" in name)
    gaze_files = list(input_dir.glob("*gaze_screen_coords*.csv"))
    if not gaze_files:
        raise FileNotFoundError(f"No gaze screen coordinates CSV found in {input_dir}")
    gaze_csv = gaze_files[0]  # Use first match

    # Find frame metadata CSV
    metadata_files = list(input_dir.glob("frame_metadata.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"No frame_metadata.csv found in {input_dir}")
    metadata_csv = metadata_files[0]

    # Find frames directory
    frames_dir = input_dir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"No frames directory found at {frames_dir}")

    # Find IMU log CSV (optional - contains "imu_log" in name)
    imu_files = list(input_dir.glob("*imu_log*.csv"))
    imu_csv = imu_files[0] if imu_files else None

    print(f"Discovered data files:")
    print(f"  Gaze data: {gaze_csv}")
    print(f"  Metadata: {metadata_csv}")
    print(f"  Frames: {frames_dir}")
    if imu_csv:
        print(f"  IMU data: {imu_csv}")
    else:
        print(f"  IMU data: Not found (optional)")

    return gaze_csv, metadata_csv, frames_dir, imu_csv




def visualize_with_config(gaze_df: pd.DataFrame, metadata_df: pd.DataFrame,
                          frames: Dict[str, bytes], imu_df: Optional[pd.DataFrame],
                          config: VisualizationConfig, recording_stream: rr.RecordingStream):
    """Run visualization with the provided configuration using memory-efficient processing.

    Args:
        gaze_df: Complete gaze data
        metadata_df: Frame metadata
        frames: Loaded frame images as compressed JPEG bytes
        imu_df: IMU sensor data (optional)
        config: Visualization configuration
        recording_stream: Rerun recording stream for logging data
    """

    # Log all data chronologically
    print(f"Logging all {len(gaze_df)} gaze samples")

    # Get camera intrinsics from first frame metadata
    first_meta = metadata_df.iloc[0]
    image_width = int(first_meta['width'])
    image_height = int(first_meta['height'])
    focal_length = [first_meta['focalLengthX'], first_meta['focalLengthY']]
    principal_point = [first_meta['principalPointX'], first_meta['principalPointY']]

    print(f"Camera intrinsics: {image_width}x{image_height}, f={focal_length}, c={principal_point}")

    if config.test_y_flip:
        print("Y-flip test enabled: screenPixelY = height - screenPixelY")

    # Log camera intrinsics once as static data for memory efficiency
    rr.log(
        "world/camera/image",
        rr.Pinhole(
            focal_length=focal_length,
            principal_point=principal_point,
            width=image_width,
            height=image_height
        ),
        static=True
    )
    
    # Log IMU data if available
    if imu_df is not None and not imu_df.empty:
        log_imu_data(imu_df)
    
    # Process data chronologically using memory-efficient batch processing
    print(f"Logging data to Rerun using memory-efficient batch processing...")

    # Collect positions for trajectory and point cloud visualization
    all_gaze_positions = []
    all_gaze_colors = []  
    all_gaze_radii = []
    camera_positions = []

    # Collect and batch process all data
    camera_data = {'positions': [], 'rotations': [], 'timestamps': []}
    gaze_data = {'origins': [], 'vectors': [], 'positions': [], 'colors': [], 'timestamps': []}
    screen_data = {'positions': [], 'colors': [], 'timestamps': []}
    
    last_frame_id = None
    
    for idx, (_, row) in enumerate(gaze_df.iterrows()):
        frame_id = row['frameId']
        timestamp_ns = int(row['timestamp'])
        
        # Log camera and frame data when frame changes
        if frame_id != last_frame_id and frame_id in frames:
            # Set timeline for this frame
            rr.set_time("timestamp", timestamp=1e-9 * timestamp_ns)
            
            # Log compressed image
            rr.log("world/camera/image", rr.EncodedImage(contents=frames[frame_id], media_type="image/jpeg"))
            
            # Collect camera transform data
            cam_pos_unity = [row['cameraPositionX'], row['cameraPositionY'], row['cameraPositionZ']]
            cam_pos = unity_to_rerun_position(cam_pos_unity)
            
            cam_rot_unity = [row['cameraRotationX'], row['cameraRotationY'],
                           row['cameraRotationZ'], row['cameraRotationW']]
            cam_rot = unity_to_rerun_quaternion(cam_rot_unity)
            
            
            camera_data['positions'].append(cam_pos)
            camera_data['rotations'].append(cam_rot)
            camera_data['timestamps'].append(timestamp_ns)
            
            last_frame_id = frame_id
        
        # Collect gaze data
        if row['isTracking'] and row['hasHitTarget']:
            origin_unity = [row['gazeOriginX'], row['gazeOriginY'], row['gazeOriginZ']]
            pos_unity = [row['gazePositionX'], row['gazePositionY'], row['gazePositionZ']]
            
            origin = unity_to_rerun_position(origin_unity)
            position = unity_to_rerun_position(pos_unity)
            vector = [position[0] - origin[0], position[1] - origin[1], position[2] - origin[2]]
            color = get_gaze_color(row['gazeState'])
            
            gaze_data['origins'].append(origin)
            gaze_data['vectors'].append(vector)
            gaze_data['positions'].append(position)
            gaze_data['colors'].append(color)
            gaze_data['timestamps'].append(timestamp_ns)
        
        # Collect screen gaze data
        if row['isValidProjection'] and not pd.isna(row['screenPixelX']):
            screen_x = row['screenPixelX']
            screen_y = row['screenPixelY']
            
            if config.test_y_flip:
                screen_y = image_height - screen_y
            
            screen_data['positions'].append([screen_x, screen_y])
            screen_data['colors'].append(get_gaze_color(row['gazeState']))
            screen_data['timestamps'].append(timestamp_ns)
        
        if (idx + 1) % 1000 == 0:
            print(f"Collected {idx + 1}/{len(gaze_df)} samples for batch processing...")
    
    # Log camera transforms in batch
    if camera_data['positions']:
        print(f"Logging {len(camera_data['positions'])} camera transforms in batch...")
        for pos, rot, ts in zip(camera_data['positions'], camera_data['rotations'], camera_data['timestamps']):
            rr.set_time("timestamp", timestamp=1e-9 * ts)
            rr.log(
                "world/camera",
                rr.Transform3D(
                    translation=pos,
                    rotation=rr.Quaternion(xyzw=rot)
                )
            )
        camera_positions.extend(camera_data['positions'])
    
    # Log gaze data in batch
    if gaze_data['origins']:
        print(f"Logging {len(gaze_data['origins'])} gaze rays in batch...")
        for origin, vector, position, color, ts in zip(
            gaze_data['origins'], gaze_data['vectors'], gaze_data['positions'], 
            gaze_data['colors'], gaze_data['timestamps']):
            
            rr.set_time("timestamp", timestamp=1e-9 * ts)
            
            # Log gaze ray
            rr.log(
                "world/gaze_ray",
                rr.Arrows3D(
                    origins=[origin],
                    vectors=[vector],
                    colors=[color],
                    radii=0.002
                )
            )
            
            # Log hit point
            rr.log(
                "world/gaze_hit",
                rr.Points3D(
                    positions=[position],
                    colors=[color],
                    radii=0.01
                )
            )
        
        all_gaze_positions.extend(gaze_data['positions'])
        all_gaze_colors.extend(gaze_data['colors'])
        all_gaze_radii.extend([0.008] * len(gaze_data['positions']))
    
    # Log screen gaze points in batch
    if screen_data['positions']:
        print(f"Logging {len(screen_data['positions'])} screen gaze points in batch...")
        for pos, color, ts in zip(screen_data['positions'], screen_data['colors'], screen_data['timestamps']):
            rr.set_time("timestamp", timestamp=1e-9 * ts)
            rr.log(
                "world/camera/image/gaze_2d",
                rr.Points2D(
                    positions=[pos],
                    colors=[color],
                    radii=10
                )
            )

    # Add static visualizations (trajectories, point clouds)
    _add_static_visualizations(config, all_gaze_positions, all_gaze_colors, 
                              all_gaze_radii, camera_positions, recording_stream)




def _add_static_visualizations(config, all_gaze_positions, all_gaze_colors, all_gaze_radii, camera_positions, recording_stream):
    """Add static visualizations like point clouds and trajectories."""
    if config.show_point_cloud and all_gaze_positions:
        print(f"Creating point clouds")
        rr.log("world/gaze_point_cloud", rr.Points3D(
            positions=all_gaze_positions,
            colors=all_gaze_colors,
            radii=all_gaze_radii,
        ), static=True, )

    if config.show_gaze_trajectory and len(all_gaze_positions) > 1:
        print(f"Creating gaze trajectory...")
        rr.log("world/gaze_trajectory", rr.LineStrips3D(
            strips=[all_gaze_positions],
            colors=[[128, 128, 255]],
            radii=0.001
        ), static=True, )

    if config.show_camera_trajectory and len(camera_positions) > 1:
        print(f"Creating camera trajectory positions...")
        rr.log("world/camera_trajectory", rr.LineStrips3D(
            strips=[camera_positions],
            colors=[[255, 128, 0]],
            radii=0.0002
        ), static=True, )

    # Final flush to ensure all data is sent before exit
    print("Flushing remaining data...")
    flush_start = time.time()
    recording_stream.flush(blocking=True)
    flush_time = time.time() - flush_start
    print(f"  Final flush completed in {flush_time:.3f}s")

    print("Visualization complete! Use the Rerun viewer to explore the data.")


def main():
    """Main function to visualize Magic Leap 2 gaze data using Rerun.
    
    Loads gaze data, camera frames, and metadata, then creates an interactive 
    3D visualization with synchronized 2D camera view overlays.
    """
    parser = argparse.ArgumentParser(description="Visualize Magic Leap 2 gaze data")
    parser.add_argument("--input", type=str, default="input",
                        help="Input directory containing data files")
    parser.add_argument("--no-tui", action="store_true",
                        help="Skip TUI configuration and use defaults")
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Get configuration - either from TUI or defaults
    if args.no_tui:
        print("Using default configuration...")
        config = VisualizationConfig()
        config.input_directory = str(input_dir)  # Use command line input
    else:
        print("Launching configuration interface...")
        config = run_configuration_tui()
        if config is None:
            print("Configuration cancelled. Exiting.")
            return

    # Use input directory from config (TUI can override command line)
    actual_input_dir = Path(config.input_directory)

    print(f"Configuration: trails={'enabled' if config.enable_fade_trail else 'disabled'}")
    print(f"Input directory: {actual_input_dir}")

    # Discover data files
    try:
        gaze_csv, metadata_csv, frames_dir, imu_csv = discover_data_files(actual_input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Initialize Rerun
    print("Initializing Rerun viewer...")
    rec = rr.RecordingStream("ml2_gaze_viewer")
    rec.spawn()

    # Configure gRPC connection with longer flush timeout to prevent data loss
    print("Configuring gRPC connection with 10-second flush timeout...")
    rec.connect_grpc(flush_timeout_sec=10.0)

    # Set as global recording stream
    rr.set_global_data_recording(rec)

    # Set coordinate system - RDF aligns with standard camera coordinates
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    # Log static coordinate indicators at world origin
    log_coordinate_indicators(config)

    # Initial flush after setup
    print("Flushing initial setup...")
    flush_start = time.time()
    rec.flush(blocking=True)
    flush_time = time.time() - flush_start
    print(f"  Initial flush completed in {flush_time:.3f}s")

    # Load data using discovered files
    print("Loading data...")
    gaze_df = load_gaze_screen_coords(gaze_csv)
    metadata_df = load_frame_metadata(metadata_csv)
    frames = load_all_frames(frames_dir)
    
    # Load IMU data if available and enabled
    imu_df = None
    if imu_csv and config.show_imu_data:
        imu_df = load_imu_data(imu_csv)
    elif imu_csv:
        print("IMU data found but disabled in configuration")
    else:
        print("IMU data not available")

    # Run visualization with configuration
    visualize_with_config(gaze_df, metadata_df, frames, imu_df, config, rec)


if __name__ == "__main__":
    main()
