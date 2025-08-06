#!/usr/bin/env python3
"""
Simple timestamp analysis utility for MLGaze Viewer
Analyzes timing differences between sensors to check synchronization
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_timestamp_sync(gaze_df: pd.DataFrame, metadata_df: pd.DataFrame, imu_df: pd.DataFrame = None, sample_rate: float = 0.3):
    """
    Find timing differences between sensors for same events.
    
    Args:
        gaze_df: DataFrame with gaze data including frameId and timestamp
        metadata_df: DataFrame with frame metadata including frameId and timestamp
        imu_df: Optional DataFrame with IMU data
        sample_rate: Float between 0.0-1.0 for sampling percentage (default: 0.3 = 30%)
    """
    print("\n" + "="*60)
    print("Timestamp Synchronization Analysis")
    print("="*60)
    
    # Analyze frame-gaze timing
    print("\n1. Frame-Gaze Timing Analysis:")
    print("-" * 40)
    
    # Check for required columns
    required_cols = ['frameId', 'timestamp']
    missing_gaze = [col for col in required_cols if col not in gaze_df.columns]
    missing_meta = [col for col in required_cols if col not in metadata_df.columns]
    
    if missing_gaze:
        print(f"ERROR: Missing columns in gaze_df: {missing_gaze}")
        return
    if missing_meta:
        print(f"ERROR: Missing columns in metadata_df: {missing_meta}")
        return
    
    # Create a dict of frame timestamps for quick lookup
    frame_timestamps = dict(zip(metadata_df['frameId'], metadata_df['timestamp']))
    
    # Find gaze samples that have matching frames
    gaze_with_frames = gaze_df[gaze_df['frameId'].isin(frame_timestamps.keys())].copy()
    
    if len(gaze_with_frames) > 0:
        # Calculate time differences in milliseconds
        gaze_with_frames['frame_timestamp'] = gaze_with_frames['frameId'].map(frame_timestamps)
        gaze_with_frames['time_diff_ms'] = (gaze_with_frames['timestamp'] - gaze_with_frames['frame_timestamp']) / 1e6
        
        # Statistics
        mean_diff = gaze_with_frames['time_diff_ms'].mean()
        std_diff = gaze_with_frames['time_diff_ms'].std()
        max_diff = gaze_with_frames['time_diff_ms'].abs().max()
        min_diff = gaze_with_frames['time_diff_ms'].min()
        
        print(f"Samples analyzed: {len(gaze_with_frames)}")
        print(f"Mean difference: {mean_diff:.2f} ms")
        print(f"Std deviation: {std_diff:.2f} ms")
        print(f"Min difference: {min_diff:.2f} ms")
        print(f"Max difference: {max_diff:.2f} ms")
        
        # Check for multiple gaze samples per frame
        samples_per_frame = gaze_with_frames.groupby('frameId').size()
        print(f"\nGaze samples per frame:")
        print(f"  Average: {samples_per_frame.mean():.1f}")
        print(f"  Max: {samples_per_frame.max()}")
        
        # Analyze timing spread within frames
        frames_with_multiple = samples_per_frame[samples_per_frame > 1].index
        if len(frames_with_multiple) > 0:
            max_spreads = []
            # Sample frames using configurable sample rate
            sample_size = max(10, int(len(frames_with_multiple) * sample_rate))
            sample_size = min(sample_size, len(frames_with_multiple))
            sampled_frames = frames_with_multiple[:sample_size]  # Take first N for consistency
            
            for frame_id in sampled_frames:
                frame_gaze = gaze_with_frames[gaze_with_frames['frameId'] == frame_id]
                spread = (frame_gaze['timestamp'].max() - frame_gaze['timestamp'].min()) / 1e6
                max_spreads.append(spread)
            
            if max_spreads:
                print(f"  Max timing spread within frame: {max(max_spreads):.2f} ms")
                print(f"  Avg timing spread within frame: {np.mean(max_spreads):.2f} ms")
    else:
        print("No matching frame-gaze pairs found!")
    
    # Analyze IMU timing if available
    if imu_df is not None and len(imu_df) > 0:
        print("\n2. IMU-Frame Timing Analysis:")
        print("-" * 40)
        
        # For each frame, find nearest IMU sample
        frame_times = metadata_df['timestamp'].values
        imu_times = imu_df['timestamp'].values
        
        # Sample analysis using configurable sample rate
        sample_size = max(10, int(len(frame_times) * sample_rate))  # Minimum 10 samples
        sample_size = min(sample_size, len(frame_times))  # Don't exceed available data
        sample_frames = np.random.choice(len(frame_times), sample_size, replace=False)
        
        time_diffs = []
        for idx in sample_frames:
            frame_time = frame_times[idx]
            nearest_idx = np.argmin(np.abs(imu_times - frame_time))
            time_diff_ms = (imu_times[nearest_idx] - frame_time) / 1e6
            time_diffs.append(time_diff_ms)
        
        print(f"Samples analyzed: {sample_size}")
        print(f"Mean nearest IMU difference: {np.mean(time_diffs):.2f} ms")
        print(f"Std deviation: {np.std(time_diffs):.2f} ms")
        print(f"Max difference: {np.max(np.abs(time_diffs)):.2f} ms")
        
        # IMU sampling rate
        if len(imu_df) > 1:
            imu_intervals = np.diff(imu_df['timestamp'].values) / 1e6  # ms
            print(f"\nIMU sampling:")
            print(f"  Mean interval: {np.mean(imu_intervals):.2f} ms")
            print(f"  Sampling rate: ~{1000/np.mean(imu_intervals):.1f} Hz")
    
    # Summary recommendation
    print("\n" + "="*60)
    print("Summary:")
    if 'mean_diff' in locals():
        # Check for artificial perfect synchronization
        if mean_diff == 0.0 and std_diff == 0.0 and len(gaze_with_frames) > 100:
            print("️   WARNING: Perfect timestamp alignment detected!")
            print("   This suggests post-processed or artificially synchronized data.")
            print("   Gaze timestamps appear to be snapped to frame capture times.")
            
            # Use IMU timing if available, otherwise use frame period
            if imu_df is not None and 'time_diffs' in locals() and len(time_diffs) > 0:
                recommended_tolerance = np.max(np.abs(time_diffs))
                print(f"   Using IMU-Frame timing for tolerance: ±{recommended_tolerance:.1f} ms")
            else:
                # Assume 60 FPS as default
                recommended_tolerance = 16.67  # One frame period at 60 FPS
                print(f"   Using frame period for tolerance: ±{recommended_tolerance:.1f} ms (60 FPS)")
        else:
            # Normal case with actual timing variations
            recommended_tolerance = max(abs(mean_diff) + 2*std_diff, max_diff) if max_diff > 0 else 16.67
            print(f"Recommended sync tolerance: ±{recommended_tolerance:.1f} ms")
    print("="*60 + "\n")


def main():
    """Run standalone timestamp analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze timestamp synchronization")
    parser.add_argument("--input", type=str, default="input",
                        help="Input directory containing data files")
    parser.add_argument("--sample-rate", type=float, default=0.3,
                        help="Sampling rate for analysis (0.0-1.0, default: 0.3 = 30%%)")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    # Load data files
    print("Loading data files...")
    
    # Find and load gaze data
    gaze_files = list(input_dir.glob("*gaze_screen_coords*.csv"))
    if gaze_files:
        gaze_df = pd.read_csv(gaze_files[0])
        gaze_df['timestamp'] = gaze_df['timestamp'].astype(np.int64)
        print(f"Loaded {len(gaze_df)} gaze samples")
    else:
        print("No gaze data found!")
        return
    
    # Load frame metadata
    metadata_file = input_dir / "frame_metadata.csv"
    if metadata_file.exists():
        metadata_df = pd.read_csv(metadata_file)
        metadata_df['timestamp'] = metadata_df['timestamp'].astype(np.int64)
        print(f"Loaded {len(metadata_df)} frame metadata entries")
    else:
        print("No frame metadata found!")
        return
    
    # Load IMU data (optional)
    imu_files = list(input_dir.glob("*imu_log*.csv"))
    imu_df = None
    if imu_files:
        imu_df = pd.read_csv(imu_files[0])
        imu_df['timestamp'] = imu_df['timestamp'].astype(np.int64)
        if 'hasValidData' in imu_df.columns:
            imu_df = imu_df[imu_df['hasValidData'] == True]
        print(f"Loaded {len(imu_df)} IMU samples")
    
    # Validate sample rate
    if not 0.0 <= args.sample_rate <= 1.0:
        print(f"Error: sample-rate must be between 0.0 and 1.0, got {args.sample_rate}")
        return
    
    # Run analysis
    analyze_timestamp_sync(gaze_df, metadata_df, imu_df, args.sample_rate)


if __name__ == "__main__":
    main()