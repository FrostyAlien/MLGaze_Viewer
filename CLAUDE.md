# MLGaze_Viewer System Rulebook

This is the authoritative guide for Magic Leap 2 gaze visualization using Rerun.

## Core Objective
Build a Rerun-based visualizer that displays ML2 eye gaze rays in 3D space and 2D camera frames with synchronized playback.

## Data Schema

### Input Files
- `input/gaze_data_*.csv`: 3D gaze vectors (origin, direction, hit position)
- `input/gaze_screen_coords_*.csv`: 2D projections + camera transforms
- `input/frames/*.jpg`: Extracted camera frames
- `input/camera_data_*.csv`: Camera poses (position, rotation, intrinsics)

### Key Fields
- Timestamps: Nanoseconds (requires int64)
- Camera rotation: Quaternion XYZW
- Gaze ray: origin + direction * distance = position
- Screen pixel: 2D coordinates in camera frame

### Data Flow
1. Load CSVs with pandas
2. Log camera: Transform3D (position, quaternion) + Pinhole (intrinsics)  
3. Log gaze: Arrows3D (origin→position) + Points2D (screenPixel)
4. Log images: Image archetype per frame

### Important Knowledge
- Coordinates system: All data are recorded using Unity, which is  left-handed, Y-up world coordinate system.
- Rerun SDK Docs: https://rerun.io/docs/getting-started/what-is-rerun
- Textual Docs: https://textual.textualize.io/guide/

### Critical Constraints
- Match frameId to timestamps for synchronization
- Color-code by gazeState (Fixation=green, Saccade=yellow, etc.)
- Use timeline with nanosecond precision
- Optimize for 1000+ frames

## Forbidden Actions
- Do NOT modify input data files
- Do NOT implement without Rerun SDK
- DO NOT add emojis unless specified