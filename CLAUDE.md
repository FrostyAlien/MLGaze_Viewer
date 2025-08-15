# MLGaze_Viewer System Rulebook

This is the authoritative guide for Magic Leap 2 gaze visualization using Rerun.

## Core Objective
Build a Rerun-based visualizer that displays ML2 eye gaze rays in 3D space and 2D camera frames with synchronized playback.

## Critical Architecture Rules
- **ALWAYS** implement new features as plugins inheriting from `AnalyticsPlugin` or `BaseSensor` if the feature is sensor or analytics related.
- **NEVER** add analytics code directly to core modules. - use `src/analytics/` or `plugins/`
- **ALWAYS** use `DataLoader` → `SessionData` → `RerunVisualizer` data flow
- **NEVER** bypass SessionData or directly manipulate CSVs
- **NEVER** creating monolithic scripts instead of modular plugins
- **ALWAYS** reuse existing components from `src/` to avoid code duplication. 
- **ALWAYS** notify the user if a reused method should be refactored when it starts breaking the Single Responsibility Principle.
- **ALWAYS** use SessionData.primary_camera for 3D visualization
- **ALWAYS** use organized session structure (cameras/, sensors/ directories)
- **NEVER** mix 2D and 3D camera poses in same entity path

## Multi-Camera Rules
- **PRIMARY CAMERA**: Only one camera renders in 3D space at `/world/camera`
- **ALL CAMERAS**: Get 2D views at `/cameras/{camera_name}/image`
- **SELECTION**: Use Select widget for camera dropdown (NEVER RadioSet - allows multiple selection)
- **VISUALIZATION**: Hide `/cameras` entity in Rerun 3D view to avoid clutter
- **ENTITY PATHS**: `/world/camera` = 3D primary, `/cameras/{name}` = 2D views only

## Data Schema

### Organized Session Structure
```
/session_*/
├── metadata.json         # Session metadata with camera list
├── cameras/
│   ├── {camera_name}/   # Each camera directory
│   │   ├── frame_metadata.csv
│   │   ├── gaze_screen_coords.csv (optional)
│   │   └── frames/ or camera_frames.mlcf
├── sensors/
│   ├── gaze_data.csv    # 3D gaze data
│   └── imu_data.csv     # IMU sensor data (optional)
```

### Key Fields
- Timestamps: Nanoseconds (requires int64)
- Camera rotation: Quaternion XYZW
- Gaze ray: origin + direction * distance = position
- Screen pixel: 2D coordinates in camera frame

## Data Flow Hierarchy
1. Load organized session via `DataLoader` → `SessionData` container
2. SessionData determines primary_camera from config or metadata
3. CameraSensor logs primary to `/world/camera`, all cameras to `/cameras/{name}`
4. Apply analytics plugins for analysis
5. Visualize via `RerunVisualizer` with proper entity path separation

## Important Knowledge
- Coordinates system: All data are recorded using Unity, which is  left-handed, Y-up world coordinate system.
- Rerun SDK Docs: https://rerun.io/docs/getting-started/what-is-rerun
- Textual Docs: https://textual.textualize.io/guide/
- The project package manager is "uv", do not use "python" to avoid conflicts with global Python packages.

### Critical Constraints
- Match frameId to timestamps for synchronization
- Color-code by gazeState (Fixation=green, Saccade=yellow, etc.)
- Use timeline with nanosecond precision
- Optimize for 1000+ frames

## Forbidden Actions
- Do NOT modify input data files
- Do NOT implement without Rerun SDK
- DO NOT add emojis unless specified

## Recommended Practices
- **When planning** ask user for clarification on any ambiguous requirements.

## Documentation Rules
- **NEVER** create documentation files unless explicitly requested
- **Include** docstrings with method, parameters, returns