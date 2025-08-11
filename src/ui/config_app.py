"""
MLGaze Viewer TUI Configuration App

Textual-based terminal UI for configuring visualization parameters.
"""

from typing import Optional
from pathlib import Path
import sys

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Input, Checkbox, RadioButton, RadioSet, DirectoryTree
)
from textual.screen import ModalScreen

# Import the centralized config from core module
from src.core.config import VisualizationConfig
from src.utils.session_utils import get_session_directories


class DirectoryBrowserScreen(ModalScreen[str]):
    """Modal screen for directory selection."""
    
    CSS = """
    DirectoryBrowserScreen {
        background: $surface;
    }
    
    .directory-container {
        width: 80%;
        height: 80%;
        margin: 2;
        padding: 1;
        border: solid $primary;
    }
    
    .browser-buttons {
        margin-top: 1;
        text-align: center;
    }
    
    .current-path {
        margin-bottom: 1;
        text-style: bold;
        color: $primary;
    }
    """
    
    def __init__(self, initial_path: str = ".") -> None:
        super().__init__()
        self.initial_path = Path(initial_path).resolve()
        self.selected_path = self.initial_path
    
    def compose(self) -> ComposeResult:
        with Container(classes="directory-container"):
            yield Static("Select Directory", classes="section-title")
            
            # Navigation controls
            with Horizontal(classes="nav-controls"):
                yield Button("Parent Folder", id="parent_dir")
                yield Button("Home", id="home_dir")
                yield Static(f"Current: {self.initial_path}", id="current_path_display")
            
            yield Static(f"Selected: {self.selected_path}", classes="current-path", id="current_path")
            yield DirectoryTree(str(self.initial_path), id="directory_tree")
            
            with Container(classes="browser-buttons"):
                yield Button("Select", variant="primary", id="select_dir")
                yield Button("Cancel", variant="default", id="cancel_dir")
    
    @on(DirectoryTree.NodeHighlighted)
    def update_selection(self, event: DirectoryTree.NodeHighlighted) -> None:
        """Update selected path when user navigates."""
        try:
            self.app.log(f"Node highlighted: {event.node}")
            if hasattr(event.node, 'data') and event.node.data and hasattr(event.node.data, 'path'):
                self.selected_path = Path(event.node.data.path)
                self.app.log(f"Selected path updated: {self.selected_path}")
                # Update the display
                self.query_one("#current_path", Static).update(f"Selected: {self.selected_path}")
            else:
                # Fallback: try to get path from tree's current location
                tree = self.query_one(DirectoryTree)
                if hasattr(tree, 'path'):
                    self.selected_path = Path(tree.path)
                    self.query_one("#current_path", Static).update(f"Selected: {self.selected_path}")
                    self.app.log(f"Fallback path: {self.selected_path}")
        except Exception as e:
            self.app.log(f"Selection update error: {e}")

    
    @on(Button.Pressed, "#select_dir")
    def select_directory(self) -> None:
        """Handle Select button press."""
        try:
            path_to_select = str(self.selected_path)
            self.app.log(f"Selecting directory via button: {path_to_select}")
            self.dismiss(path_to_select)
        except Exception as e:
            self.app.log(f"Button selection error: {e}")
            # Fallback: dismiss with None to avoid crash
            self.dismiss(None)
    
    @on(Button.Pressed, "#cancel_dir")
    def cancel_selection(self) -> None:
        """Handle Cancel button press."""
        self.dismiss(None)
    
    @on(Button.Pressed, "#parent_dir")
    def go_to_parent(self) -> None:
        """Navigate to parent directory."""
        try:
            tree = self.query_one(DirectoryTree)
            current_path = Path(tree.path)
            parent_path = current_path.parent
            
            # Check if we can actually go up (not at filesystem root)
            if parent_path != current_path:
                self.refresh_tree(str(parent_path))
            else:
                self.app.log("Already at filesystem root")
        except Exception as e:
            self.app.log(f"Parent navigation error: {e}")
    
    @on(Button.Pressed, "#home_dir")
    def go_to_home(self) -> None:
        """Navigate to home directory."""
        try:
            home_path = Path.home()
            self.refresh_tree(str(home_path))
        except Exception as e:
            self.app.log(f"Home navigation error: {e}")
    
    def refresh_tree(self, new_path: str) -> None:
        """Refresh directory tree with new path."""
        try:
            new_path_obj = Path(new_path).resolve()
            
            # Update the tree's path
            tree = self.query_one(DirectoryTree)
            tree.path = str(new_path_obj)
            tree.reload()
            
            # Update selected path
            self.selected_path = new_path_obj
            
            # Update displays
            self.query_one("#current_path").update(f"Selected: {self.selected_path}")
            self.query_one("#current_path_display").update(f"Current: {new_path_obj}")
            
            self.app.log(f"Navigated to: {new_path_obj}")
        except Exception as e:
            self.app.log(f"Tree refresh error: {e}")


class MLGazeConfigApp(App):
    """Main TUI application for MLGaze configuration."""
    
    CSS_PATH = "styles.tcss"
    
    def __init__(self):
        super().__init__()
        self.config = VisualizationConfig()
    
    def show_error(self, message: str) -> None:
        """Show error notification to user."""
        self.notify(message, severity="error", timeout=5)
    
    def show_success(self, message: str) -> None:
        """Show success notification to user."""
        self.notify(message, severity="information", timeout=3)
    
    def show_warning(self, message: str) -> None:
        """Show warning notification to user."""
        self.notify(message, severity="warning", timeout=4)
    
    def validate_directory(self, path: str) -> bool:
        """Validate if directory contains required MLGaze data files."""
        input_field = self.query_one("#input_directory", Input)
        
        if not path:
            input_field.remove_class("valid-path", "invalid-path", "warning-path")
            return False
            
        path_obj = Path(path)
        
        # Check if path exists
        if not path_obj.exists():
            self.show_error(f"Directory does not exist: {path}")
            input_field.remove_class("valid-path", "warning-path")
            input_field.add_class("invalid-path")
            return False
        
        # Check if it's actually a directory
        if not path_obj.is_dir():
            self.show_error(f"Path is not a directory: {path}")
            input_field.remove_class("valid-path", "warning-path")
            input_field.add_class("invalid-path")
            return False
        
        # Check for required data files using recursive search (matching DataLoader)
        gaze_files = list(path_obj.rglob("*gaze_screen_coords*.csv"))
        metadata_files = list(path_obj.rglob("frame_metadata.csv"))
        
        # Check for missing files
        missing_required = []
        if not gaze_files:
            missing_required.append("gaze_screen_coords*.csv")
        if not metadata_files:
            missing_required.append("frame_metadata.csv")
        
        if missing_required:
            self.show_error(f"Missing required files: {', '.join(missing_required)}")
            input_field.remove_class("valid-path", "warning-path")
            input_field.add_class("invalid-path")
            return False
        
        # Check for multiple session conflicts (matching DataLoader)
        conflict_detected = False
        conflict_messages = []
        
        if len(gaze_files) > 1:
            conflict_detected = True
            conflict_messages.append(f"Found {len(gaze_files)} gaze files")
            
        if len(metadata_files) > 1:
            conflict_detected = True
            conflict_messages.append(f"Found {len(metadata_files)} metadata files")
        
        if conflict_detected:
            # Get main session directories to recommend
            session_dirs = get_session_directories(gaze_files, metadata_files, path_obj)
            
            suggestion = ""
            if session_dirs:
                session_names = [d.relative_to(path_obj) for d in session_dirs]
                suggestion = f"\n\nTry selecting: {', '.join(map(str, session_names))}"
            
            error_msg = f"Multiple sessions detected: {', '.join(conflict_messages)}.{suggestion}"
            self.show_error(error_msg)
            input_field.remove_class("valid-path", "warning-path")
            input_field.add_class("invalid-path")
            return False
        
        # Single session validated - check frames/MLCF status
        metadata_parent = metadata_files[0].parent
        frames_dir = metadata_parent / "frames"
        
        # Check for frames or MLCF
        frames_status = None
        if frames_dir.exists():
            frames_status = "ready"
        else:
            # Look for MLCF files (first in same dir, then recursively)
            mlcf_files = list(metadata_parent.glob("*.mlcf"))
            if not mlcf_files:
                mlcf_files = list(path_obj.rglob("*.mlcf"))
            
            if mlcf_files:
                if len(mlcf_files) > 1:
                    self.show_error(f"Multiple MLCF files found: {len(mlcf_files)} files")
                    input_field.remove_class("valid-path", "warning-path")
                    input_field.add_class("invalid-path")
                    return False
                frames_status = "mlcf"
        
        # Set final status
        if frames_status == "ready":
            self.show_success("Valid MLGaze session with extracted frames")
            input_field.remove_class("invalid-path", "warning-path")
            input_field.add_class("valid-path")
        elif frames_status == "mlcf":
            self.notify(
                "✓ Valid session - MLCF file will be extracted automatically",
                severity="information",
                timeout=5
            )
            input_field.remove_class("invalid-path", "warning-path")
            input_field.add_class("valid-path")
        else:
            self.show_warning("CSV files found, but no frames or MLCF file detected")
            input_field.remove_class("valid-path", "invalid-path")
            input_field.add_class("warning-path")
        
        return True
    
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()
        
        with ScrollableContainer():
            yield Static("MLGaze Viewer Configuration", classes="section-title")
            
            with Container(classes="grid-container"):
                # Left Column - File and Temporal Settings  
                with Container(classes="left-column"):
                    # File Path Section
                    yield Static("Data Files", classes="section-title")
                    with Container(classes="section-content"):
                        yield Static("Input Directory:")
                        with Horizontal():
                            yield Input(
                                placeholder="input", 
                                value="input", 
                                id="input_directory"
                            )
                            yield Button("Browse", id="browse_directory")
                    
                    # Trail Settings Section
                    yield Static("Trail Effects", classes="section-title")
                    with Container(classes="section-content"):
                        yield Checkbox("Enable Trail Fade", value=True, id="fade_trail")
                        
                        yield Static("Fade Duration (seconds):")
                        yield Input(
                            placeholder="5.0", 
                            value="5.0", 
                            id="fade_duration"
                        )
                    
                    # Sliding Window Section
                    yield Static("Sliding Window (3D only)", classes="section-title")
                    with Container(classes="section-content"):
                        yield Checkbox("Enable Sliding Window", value=False, id="sliding_window")
                        
                        yield Static("Window Duration (seconds):")
                        yield Input(
                            placeholder="10.0", 
                            value="10.0", 
                            id="window_duration"
                        )
                        
                        yield Static("Update Rate (seconds):")
                        yield Input(
                            placeholder="0.5", 
                            value="0.5", 
                            id="update_rate"
                        )
                        
                        yield Checkbox("Apply to 3D Gaze Points", value=True, id="window_gaze")
                        yield Checkbox("Apply to 3D Trajectories", value=True, id="window_trajectory")
                        yield Checkbox("Apply to Camera Position", value=True, id="window_camera")
                
                # Right Column - Visualization Settings
                with Container(classes="right-column"):
                    yield Static("Visualization Options", classes="section-title")
                    with Container(classes="section-content"):
                        yield Checkbox("Show Point Cloud", value=True, id="point_cloud")
                        yield Checkbox("Show Gaze Trajectory", value=True, id="gaze_trajectory")
                        yield Checkbox("Show Camera Path", value=True, id="camera_path")
                        yield Checkbox("Color by Gaze State", value=True, id="color_by_state")
                        yield Checkbox("Test Y-Flip", value=False, id="y_flip")
                        yield Checkbox("Show Coordinate Indicators", value=True, id="coord_indicators")
                    
                    # IMU Sensor Data Section
                    yield Static("IMU Sensor Data", classes="section-title")
                    with Container(classes="section-content"):
                        yield Checkbox("Show IMU Data", value=True, id="show_imu")
            
            # Action Buttons (full width, outside grid)
            with Container(classes="button-container"):
                yield Button("Start Visualization", variant="primary", id="start_viz")
                yield Button("Reset to Defaults", variant="default", id="reset")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the form and set focus."""
        self.title = "MLGaze Viewer"
        self.sub_title = "Configure Visualization Parameters"
        
        # Set focus to first input field
        self.query_one("#input_directory", Input).focus()
    
    
    @on(Checkbox.Changed)
    def checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        checkbox_id = event.checkbox.id
        value = event.value
        
        if checkbox_id == "fade_trail":
            self.config.enable_fade_trail = value
        elif checkbox_id == "point_cloud":
            self.config.show_point_cloud = value
        elif checkbox_id == "gaze_trajectory":
            self.config.show_gaze_trajectory = value
        elif checkbox_id == "camera_path":
            self.config.show_camera_trajectory = value
        elif checkbox_id == "color_by_state":
            self.config.color_by_gaze_state = value
        elif checkbox_id == "y_flip":
            self.config.test_y_flip = value
        elif checkbox_id == "coord_indicators":
            self.config.show_coordinate_indicators = value
        elif checkbox_id == "sliding_window":
            self.config.enable_sliding_window = value
            if value:
                self.show_success("Sliding window enabled for 3D data")
            else:
                self.show_success("Sliding window disabled")
        elif checkbox_id == "window_gaze":
            self.config.sliding_window_3d_gaze = value
        elif checkbox_id == "window_trajectory":
            self.config.sliding_window_3d_trajectory = value
        elif checkbox_id == "window_camera":
            self.config.sliding_window_camera = value
        elif checkbox_id == "show_imu":
            self.config.show_imu_data = value
            if value:
                self.show_success("IMU sensor data visualization enabled")
            else:
                self.show_success("IMU sensor data visualization disabled")
    
    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        """Handle input field changes."""
        input_id = event.input.id
        value = event.value
        
        try:
            if input_id == "input_directory":
                self.config.input_directory = value if value else "input"
                # Validate directory path with user feedback
                if value and len(value) > 2:  # Avoid validating every keystroke
                    self.validate_directory(value)
                    
            elif input_id == "fade_duration":
                if value:
                    duration = float(value)
                    if duration < 0.1:
                        self.show_warning("Fade duration must be at least 0.1 seconds")
                    elif duration > 60:
                        self.show_warning("Fade duration is very long")
                    else:
                        self.config.fade_duration = duration
                else:
                    self.config.fade_duration = 5.0
                    
            elif input_id == "window_duration":
                if value:
                    duration = float(value)
                    if duration < 1.0:
                        self.show_warning("Window duration must be at least 1 second")
                    elif duration > 300:
                        self.show_warning("Window duration is very long (>5 minutes)")
                    else:
                        self.config.sliding_window_duration = duration
                else:
                    self.config.sliding_window_duration = 10.0
                    
            elif input_id == "update_rate":
                if value:
                    rate = float(value)
                    if rate < 0.1:
                        self.show_warning("Update rate must be at least 0.1 seconds")
                    elif rate > 10:
                        self.show_warning("Update rate is very slow (>10 seconds)")
                    else:
                        self.config.sliding_window_update_rate = rate
                else:
                    self.config.sliding_window_update_rate = 0.5
                    
        except ValueError as e:
            if "fade_duration" in input_id:
                self.show_error("Value must be a number")
            else:
                self.show_error(f"Invalid input: {e}")
    
    @on(Button.Pressed, "#start_viz")
    def start_visualization(self) -> None:
        """Handle Start Visualization button."""
        self.exit(self.config)
    
    @on(Button.Pressed, "#reset")
    def reset_to_defaults(self) -> None:
        """Handle Reset button."""
        self.config = VisualizationConfig()
        
        # Update UI elements with default values
        self.query_one("#input_directory", Input).value = "input"
        self.query_one("#fade_duration", Input).value = "5.0"
        self.query_one("#window_duration", Input).value = "10.0"
        self.query_one("#update_rate", Input).value = "0.5"
        
        # Reset checkboxes
        self.query_one("#fade_trail", Checkbox).value = True
        self.query_one("#point_cloud", Checkbox).value = True
        self.query_one("#gaze_trajectory", Checkbox).value = True
        self.query_one("#camera_path", Checkbox).value = True
        self.query_one("#color_by_state", Checkbox).value = True
        self.query_one("#y_flip", Checkbox).value = False
        self.query_one("#coord_indicators", Checkbox).value = True
        self.query_one("#sliding_window", Checkbox).value = False
        self.query_one("#window_gaze", Checkbox).value = True
        self.query_one("#window_trajectory", Checkbox).value = True
        self.query_one("#window_camera", Checkbox).value = True
        self.query_one("#show_imu", Checkbox).value = True
    
    @on(Button.Pressed, "#browse_directory")
    def browse_directory(self) -> None:
        """Open directory browser and update input field with selected path."""
        try:
            current_path = self.query_one("#input_directory", Input).value or "."
            browser = DirectoryBrowserScreen(current_path)
            
            # Push screen with callback instead of waiting
            self.push_screen(browser, callback=self.on_directory_selected)
        except Exception as e:
            self.show_error(f"Directory browser error: {e}")
            self.log(f"Directory browser exception: {e}")
    
    def on_directory_selected(self, result: str | None) -> None:
        """Handle directory selection result from browser."""
        if result:  # User selected a directory
            self.log(f"Directory browser returned: {result}")
            try:
                input_field = self.query_one("#input_directory", Input)
                input_field.value = result
                self.config.input_directory = result
                # Validate the selected directory
                self.validate_directory(result)
            except Exception as e:
                self.show_error(f"Error updating directory selection: {e}")
                self.log(f"Directory update exception: {e}")
        else:
            self.log("Directory browser cancelled or returned None")


def run_configuration_tui() -> Optional[VisualizationConfig]:
    """Run the TUI configuration app and return the configuration.
    
    Returns:
        VisualizationConfig if user clicked "Start Visualization", None if cancelled
    """
    app = MLGazeConfigApp()
    result = app.run()
    
    # If result is a VisualizationConfig, user clicked start
    if isinstance(result, VisualizationConfig):
        return result
    else:
        # User cancelled or closed app
        return None


if __name__ == "__main__":
    config = run_configuration_tui()
    if config:
        print("Configuration created:")
        print(f"  Input directory: {config.input_directory}")
        print(f"  Fade trail: {'enabled' if config.enable_fade_trail else 'disabled'}")
        print(f"  Show point cloud: {config.show_point_cloud}")
        print(f"  Show trajectories: {config.show_gaze_trajectory}")
    else:
        print("Configuration cancelled")