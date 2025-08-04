#!/usr/bin/env python3
"""
Magic Leap 2 Gaze Visualizer using Rerun
Visualizes eye gaze data in both 3D world space and 2D camera frames
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import rerun as rr
from typing import Dict, List, Tuple


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


def unity_to_rerun_position(pos: List[float]) -> List[float]:
    """Convert Unity left-handed Y-up to Rerun RDF (Right-Down-Forward).
    Unity: X=right, Y=up, Z=forward
    RDF: X=right, Y=down, Z=forward
    Need to flip Y-axis: Y_rdf = -Y_unity
    """
    return [pos[0], -pos[1], pos[2]]


def unity_to_rerun_quaternion(q: List[float]) -> List[float]:
    """Convert Unity left-handed Y-up quaternion to Rerun RDF coordinate system.
    Unity: X=right, Y=up, Z=forward (left-handed)
    RDF: X=right, Y=down, Z=forward (right-handed)
    Need to flip Y-axis rotation: negate Y and Z components.
    """
    return [q[0], -q[1], -q[2], q[3]]


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
    df['frameTimestamp'] = df['frameTimestamp'].astype(np.int64)
    
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
    df['frameMLTime'] = df['frameMLTime'].astype(np.int64)
    print(f"Loaded metadata for {len(df)} frames")
    return df


def load_all_frames(frames_dir: Path) -> Dict[str, np.ndarray]:
    """Load all JPEG frame images into memory for fast access.
    
    Args:
        frames_dir: Directory containing frame_*.jpg files
        
    Returns:
        Dictionary mapping frame IDs to RGB image arrays
    """
    print(f"Loading frames from {frames_dir}")
    frames = {}
    
    # Get all jpg files
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    
    for i, frame_path in enumerate(frame_files):
        frame_id = frame_path.stem  # e.g., "frame_000001"
        img = cv2.imread(str(frame_path))
        if img is not None:
            # Convert BGR to RGB for Rerun
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames[frame_id] = img
        
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
        'Fixation': [0, 255, 0],      # Green
        'Saccade': [255, 255, 0],     # Yellow
        'Pursuit': [0, 255, 255],     # Cyan
        'WinkRight': [255, 0, 255],   # Magenta
        'WinkLeft': [255, 0, 255],    # Magenta
    }
    return colors.get(gaze_state, [128, 128, 128])  # Gray for unknown


def main():
    """Main function to visualize Magic Leap 2 gaze data using Rerun.
    
    Loads gaze data, camera frames, and metadata, then creates an interactive 
    3D visualization with synchronized 2D camera view overlays.
    """
    parser = argparse.ArgumentParser(description="Visualize Magic Leap 2 gaze data")
    parser.add_argument("--input", type=str, default="input", 
                        help="Input directory containing data files")
    parser.add_argument("--test-y-flip", action="store_true",
                        help="Test Y-axis flip: display screenPixelY as height - screenPixelY")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    # Initialize Rerun
    print("Initializing Rerun viewer...")
    rr.init("ml2_gaze_viewer", spawn=True)
    
    # Set coordinate system - RDF aligns with standard camera coordinates
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    
    
    # Load data
    gaze_df = load_gaze_screen_coords(input_dir / "gaze_screen_coords_20250801_100123.csv")
    metadata_df = load_frame_metadata(input_dir / "frame_metadata.csv")
    frames = load_all_frames(input_dir / "frames")
    
    # Get camera intrinsics from first frame metadata
    first_meta = metadata_df.iloc[0]
    image_width = int(first_meta['width'])
    image_height = int(first_meta['height'])
    focal_length = [first_meta['focalLengthX'], first_meta['focalLengthY']]
    principal_point = [first_meta['principalPointX'], first_meta['principalPointY']]
    
    print(f"Camera intrinsics: {image_width}x{image_height}, f={focal_length}, c={principal_point}")
    
    if args.test_y_flip:
        print("Y-flip test enabled: screenPixelY = height - screenPixelY")
    
    
    # Process data chronologically
    print("Logging data to Rerun...")
    
    # Collect all valid gaze points for point cloud visualization
    all_gaze_positions = []
    all_gaze_colors = []
    all_gaze_radii = []
    
    # Collect camera positions for trajectory visualization
    camera_positions = []
    
    # Track last logged frame to avoid redundant logging
    last_frame_id = None
    
    for idx, (_, row) in enumerate(gaze_df.iterrows()):
        # Set timeline to nanosecond timestamp
        rr.set_time("timestamp", timestamp=1e-9 * int(row['frameTimestamp']))
        
        frame_id = row['frameId']
        
        # Log camera and frame if changed
        if frame_id != last_frame_id and frame_id in frames:
            # Camera transform (Unity to Rerun coordinates)
            cam_pos_unity = [row['cameraPositionX'], row['cameraPositionY'], row['cameraPositionZ']]
            cam_pos = unity_to_rerun_position(cam_pos_unity)
            
            cam_rot_unity = [row['cameraRotationX'], row['cameraRotationY'], 
                           row['cameraRotationZ'], row['cameraRotationW']]
            cam_rot = unity_to_rerun_quaternion(cam_rot_unity)
            
            
            # Log camera transform
            rr.log(
                "world/camera",
                rr.Transform3D(
                    translation=cam_pos,
                    rotation=rr.Quaternion(xyzw=cam_rot)
                )
            )
            
            # Log camera intrinsics
            rr.log(
                "world/camera/image",
                rr.Pinhole(
                    focal_length=focal_length,
                    principal_point=principal_point,
                    width=image_width,
                    height=image_height
                )
            )
            
            # Log camera image
            rr.log(
                "world/camera/image",
                rr.Image(frames[frame_id])
            )
            
            # Collect camera position for trajectory
            camera_positions.append(cam_pos)
            
            
            last_frame_id = frame_id
        
        # Log gaze ray in 3D (only if tracking and has hit target)
        if row['isTracking'] and row['hasHitTarget']:
            # Gaze origin and position in Unity coordinates
            origin_unity = [row['gazeOriginX'], row['gazeOriginY'], row['gazeOriginZ']]
            pos_unity = [row['gazePositionX'], row['gazePositionY'], row['gazePositionZ']]
            
            # Convert to Rerun coordinates
            origin = unity_to_rerun_position(origin_unity)
            position = unity_to_rerun_position(pos_unity)
            
            # Calculate vector from origin to position
            vector = np.array(position) - np.array(origin)
            
            
            
            # Get color based on gaze state
            color = get_gaze_color(row['gazeState'])
            
            # Log gaze ray as arrow
            rr.log(
                "world/gaze_ray",
                rr.Arrows3D(
                    origins=[origin],
                    vectors=[vector.tolist()],
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
            
            # Collect for point cloud visualization
            all_gaze_positions.append(position)
            all_gaze_colors.append(color)
            all_gaze_radii.append(0.008)  # Slightly smaller for point cloud
        
        # Log 2D gaze point on camera image
        if row['isValidProjection'] and not pd.isna(row['screenPixelX']):
            # Get screen coordinates
            screen_x = row['screenPixelX']
            screen_y = row['screenPixelY']
            
            # Apply Y-flip test if requested
            if args.test_y_flip:
                screen_y = image_height - screen_y
            
            # Log gaze point on 2D image
            rr.log(
                "world/camera/image/gaze_2d",
                rr.Points2D(
                    positions=[[screen_x, screen_y]],
                    colors=[get_gaze_color(row['gazeState'])],
                    radii=5.0
                )
            )
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(gaze_df)} gaze samples...")
    
    # Create gaze point cloud visualization
    if all_gaze_positions:
        print(f"Creating point cloud with {len(all_gaze_positions)} gaze points...")
        rr.log(
            "world/gaze_point_cloud",
            rr.Points3D(
                positions=all_gaze_positions,
                colors=all_gaze_colors,
                radii=all_gaze_radii
            ),
            static=True
        )
    
    # Create gaze trajectory visualization
    if len(all_gaze_positions) > 1:
        print("Creating gaze trajectory...")
        rr.log(
            "world/gaze_trajectory",
            rr.LineStrips3D(
                strips=[all_gaze_positions],
                colors=[[128, 128, 255]],  # Light blue for trajectory
                radii=0.001
            ),
            static=True
        )
    
    # Create camera trajectory visualization
    if len(camera_positions) > 1:
        print("Creating camera trajectory...")
        rr.log(
            "world/camera_trajectory",
            rr.LineStrips3D(
                strips=[camera_positions],
                colors=[[255, 128, 0]],  # Orange for camera path
                radii=0.001
            ),
            static=True
        )
    
    print("Visualization complete! Use the Rerun viewer to explore the data.")


if __name__ == "__main__":
    main()