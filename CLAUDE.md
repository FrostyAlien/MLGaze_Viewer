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

## Data Schema

### Input Files
- `gaze_data_*.csv`: 3D gaze vectors (origin, direction, hit position)
- `gaze_screen_coords_*.csv`: 2D projections + camera transforms
- `frames/*.jpg`: Extracted camera frames
- `camera_data_*.csv`: Camera poses (position, rotation, intrinsics)
- `imu_log_*.csv`: IMU sensor data

### Key Fields
- Timestamps: Nanoseconds (requires int64)
- Camera rotation: Quaternion XYZW
- Gaze ray: origin + direction * distance = position
- Screen pixel: 2D coordinates in camera frame

## Data Flow Hierarchy
1. Load data via `DataLoader` → `SessionData` container
2. Process through sensor handlers (`GazeSensor`, `CameraSensor`, `IMUSensor`)
3. Apply analytics plugins for analysis
4. Visualize via `RerunVisualizer` with proper entity
   paths

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