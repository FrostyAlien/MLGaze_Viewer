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

### Optional: Object Detection Support

To enable object detection features:
```bash
# RF-DETR models will be downloaded automatically on first use
# Requires ~500MB for base model
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

The tool supports both modern organized session data and legacy formats for backwards compatibility.

```
input/
└── session_xxxx/           # Session directory
    ├── metadata.json                   # Session metadata with camera list
    ├── cameras/
    │   ├── rgb_camera/                 # Camera-specific directory
    │   │   ├── camera_frames.mlcf      # Binary frame container
    │   │   ├── frame_metadata.csv      # Frame timestamps & poses
    │   │   └── gaze_screen_coords.csv  # Gaze in camera coordinates
    │   └── mr_capture/                 # Multiple cameras supported
    │       ├── camera_frames.mlcf
    │       └── frame_metadata.csv
    ├── sensors/
    │   ├── gaze_data.csv              # 3D world-space gaze data
    │   └── imu_data.csv               # IMU sensor data
    └── logs/
        └── session.log                # Session debug logs
```

## Features

### Completed ✅
- ✅ **Object Detection**: RF-DETR integration with configurable models (nano, small, medium, base)
- ✅ **Plugin Dependency System**: DAG-based automatic execution ordering
- ✅ **Multi-Camera Support**: Synchronized visualization across cameras

### In Progress 🚧
- 🚧 **Gaze-Object Interaction**: Mapping gaze to detected objects
- 🚧 **3D Instance Tracking**: HDBSCAN clustering for object persistence
- 🚧 **Sensor Recorder**: ML2 sensor recording application (in development)

### Planned 📋
- [ ] **Heatmap Generation**: Gaze density visualization overlays for attention analysis
- [ ] **Real-time Streaming**: Live data analysis support
- [ ] **Advanced AOI Analysis**: Enhanced Area of Interest tracking and reporting

## Architecture

The project uses a sophisticated plugin-based architecture with automatic dependency management:

### Plugin Dependency System
- **DAG-based Resolution**: Plugins declare dependencies, system automatically determines execution order
- **Topological Sort**: Ensures plugins run in correct sequence based on dependencies
- **Graceful Degradation**: Optional dependencies enhance functionality when available

### Core Components
- **DataLoader**: Handles organized session data with multi-camera support
- **SessionData**: Central container with plugin results storage
- **Plugin System** (`src/plugin_sys/`): Advanced dependency management and execution
- **Analytics Plugins**: Extensible modules with dependency declarations
- **RerunVisualizer**: Orchestrates sensors and plugins with automatic ordering


## Development

### Adding Custom Plugins

Create a new plugin by inheriting from `AnalyticsPlugin`:

```python
from src.plugin_sys.base import AnalyticsPlugin
from typing import List, Dict, Any

class CustomAnalyzer(AnalyticsPlugin):
    def get_dependencies(self) -> List[str]:
        return ["ObjectDetector"]  # Requires ObjectDetector to run first
    
    def get_optional_dependencies(self) -> List[str]:
        return ["AOIAnalyzer"]  # Enhanced if AOI data available
    
    def process(self, session, config: Dict[str, Any]):
        # Access dependency results
        detections = config["dependencies"]["ObjectDetector"]
        # Your analysis logic here
        return results
```

### Project Structure

```
MLGaze_Viewer/
├── src/
│   ├── core/           # Core data structures and session management
│   ├── plugin_sys/     # Plugin infrastructure with DAG dependency management
│   ├── sensors/        # Sensor data handlers
│   ├── analytics/      # Analytics plugins (ObjectDetector, AOIAnalyzer, etc.)
│   ├── visualization/  # Rerun visualization orchestrator
│   ├── ui/            # Terminal UI components
│   └── utils/         # Utilities (logging, data loading)
├── models/            # ML models for object detection
├── input/             # Session data input directory
├── plugins/           # Custom plugin modules
└── scripts/          # Entry points
```
