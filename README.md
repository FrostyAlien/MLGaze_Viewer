# MLGaze Viewer

A Rerun-based visualization tool for Magic Leap 2 eye tracking data, providing synchronized 3D gaze ray visualization and 2D camera frame analysis.

## Overview

MLGaze Viewer transforms raw Magic Leap 2 sensor data into interactive 3D visualizations, enabling researchers and developers to analyze eye tracking behavior, head movements, and sensor data in real-time. Built on a modular plugin architecture, it supports extensible analytics and custom visualizations.

## Key Features

- **3D Gaze Visualization**: Real-time rendering of eye gaze rays in 3D world space
- **Synchronized Playback**: Frame-accurate synchronization between gaze data and camera footage
- **Multi-Sensor Support**: Integrated visualization of gaze, camera, and IMU data
- **Plugin Architecture**: Extensible analytics framework for custom analysis modules
- **Interactive Configuration**: Terminal UI for visualization settings and plugin management
- **AOI Analysis**: Built-in Area of Interest tracking and analysis

## Installation

### Prerequisites

- Python 3.12 or higher
- UV package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/FrostyAlien/MLGaze_Viewer/
cd MLGaze_Viewer
```

2. Install dependencies using UV:
```bash
uv sync
```

## Usage

### Basic Visualization

```bash
uv run scripts/visualize.py --input <data-directory>
```

### With Configuration UI

```bash
uv run scripts/visualize.py
```

## Data Format

The tool expects the following data structure in your input directory:

```
input/
├── gaze_data_*.csv          # 3D gaze vectors
├── gaze_screen_coords_*.csv # 2D screen projections
├── camera_data_*.csv        # Camera poses
├── imu_log_*.csv           # IMU sensor data
└── frames/                 # Extracted camera frames
    └── *.jpg
```

## TODO
- [ ] **Sensor Recorder**: The recoder for Magic Leap 2 sensors is still under development, which will release in the future to use with this viewer.
- [ ] **Object Detection**: YOLO/SAM integration for automatic AOI detection in camera frames
- [ ] **Heatmap Generation**: Gaze density visualization overlays for attention analysis
- [ ] **Scanpath Analysis**: Sequential gaze pattern analysis and visualization
- [ ] **Real-time Streaming**: Live data streaming support for online analysis

## Architecture

The project follows a modular plugin-based architecture:

- **DataLoader**: Handles CSV and frame data ingestion
- **SessionData**: Central data container with unified timestamps
- **Sensors**: Modular handlers for different data types (gaze, camera, IMU)
- **Analytics Plugins**: Extensible analysis modules
- **RerunVisualizer**: Main visualization orchestrator

## Development

### Adding Custom Plugins

Create a new plugin by inheriting from `AnalyticsPlugin`:

```python
from src.analytics.base import AnalyticsPlugin

class CustomAnalyzer(AnalyticsPlugin):
    def process(self, session_data):
        # Your analysis logic here
        pass
```

### Project Structure

```
MLGaze_Viewer/
├── src/
│   ├── core/           # Core data structures
│   ├── sensors/        # Sensor data handlers
│   ├── analytics/      # Analysis plugins
│   ├── visualization/  # Rerun visualization
│   └── ui/            # Terminal UI components
├── plugins/           # Custom plugin modules
└── scripts/          # Entry points
```
