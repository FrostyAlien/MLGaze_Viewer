#!/usr/bin/env python3
"""
Main entry point for MLGaze Viewer using the refactored architecture.
Visualizes Magic Leap 2 gaze data in both 3D world space and 2D camera frames.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import VisualizationConfig, BoundingBox
from src.utils import DataLoader
from src.visualization import RerunVisualizer
from src.sensors import GazeSensor, CameraSensor, IMUSensor
from src.analytics import load_plugins
from src.ui.config_app import run_configuration_tui
import numpy as np


def setup_example_aois() -> list:
    """Create example AOI regions for demonstration.
    
    Returns:
        List of BoundingBox objects defining AOIs
    """
    # Example 2D screen-space AOIs
    return [
        BoundingBox(
            name="upper_left",
            bounds=np.array([0, 0, 960, 540]),  # x, y, width, height
            category="screen_quadrant"
        ),
        BoundingBox(
            name="upper_right", 
            bounds=np.array([960, 0, 960, 540]),
            category="screen_quadrant"
        ),
        BoundingBox(
            name="lower_left",
            bounds=np.array([0, 540, 960, 540]),
            category="screen_quadrant"
        ),
        BoundingBox(
            name="lower_right",
            bounds=np.array([960, 540, 960, 540]),
            category="screen_quadrant"
        ),
        BoundingBox(
            name="center",
            bounds=np.array([480, 270, 960, 540]),
            category="screen_center"
        )
    ]


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
    parser.add_argument("--enable-aoi", action="store_true",
                        help="Enable AOI analysis with example regions")
    parser.add_argument("--plugins", nargs="+", default=[],
                        help="List of analytics plugins to enable")
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

    # Initialize data loader
    loader = DataLoader(verbose=True)
    
    try:
        # Load session data
        session = loader.load_session(str(actual_input_dir))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Initialize visualizer with config
    viz = RerunVisualizer(config)
    
    # Add sensors
    viz.add_sensor(GazeSensor())
    viz.add_sensor(CameraSensor())
    if config.show_imu_data:
        viz.add_sensor(IMUSensor())
    
    # Setup analytics plugins
    analytics_results = None
    
    # Add AOI analyzer if requested
    if args.enable_aoi:
        from src.analytics import AOIAnalyzer
        aoi_analyzer = AOIAnalyzer(regions=setup_example_aois())
        viz.add_plugin(aoi_analyzer)
        print("AOI analysis enabled with example screen quadrants")
    
    # Add any additional plugins from command line
    if args.plugins:
        plugins = load_plugins(args.plugins)
        for plugin in plugins:
            viz.add_plugin(plugin)
            print(f"Added plugin: {plugin.name}")
    
    # Or use plugins from config
    elif config.enabled_plugins:
        plugins = load_plugins(config.enabled_plugins)
        for plugin in plugins:
            viz.add_plugin(plugin)
            print(f"Added plugin from config: {plugin.name}")
    
    # Run visualization
    viz.visualize(session, analytics_results)


if __name__ == "__main__":
    main()