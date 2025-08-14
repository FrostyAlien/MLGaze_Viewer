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
    Header, Footer, Static, Button, Input, Checkbox, Select, DirectoryTree
)
from textual.screen import ModalScreen

# Import the centralized config from core module
from src.core.config import VisualizationConfig
from src.core import SessionMetadata


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
        self.cameras = []  # Available cameras from session
        self.session_validated = False  # Track if session directory is valid
    
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
        """Validate if directory contains required organized session structure."""
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
        
        # Validate organized session structure
        try:
            self._validate_session_structure(path_obj)
            # If validation passes, try to load session metadata to get cameras
            self._load_session_info(path_obj)
        except (FileNotFoundError, ValueError) as e:
            self.show_error(f"Invalid session structure: {e}")
            input_field.remove_class("valid-path", "warning-path")
            input_field.add_class("invalid-path")
            return False
        
        # If we get here, validation passed
        input_field.remove_class("invalid-path", "warning-path")
        input_field.add_class("valid-path")
        self.session_validated = True
        self.show_success("Valid session directory")
        # Now show camera selection UI if multiple cameras available
        self._show_camera_selection_ui()
        return True
    
    def _validate_session_structure(self, session_path: Path) -> None:
        """Validate organized session structure with warnings for missing optional files."""
        # Check required directories
        cameras_dir = session_path / "cameras"
        sensors_dir = session_path / "sensors"
        
        missing_dirs = []
        if not cameras_dir.exists():
            missing_dirs.append("cameras/")
        if not sensors_dir.exists():
            missing_dirs.append("sensors/")
        
        if missing_dirs:
            raise FileNotFoundError(f"Required directories missing: {', '.join(missing_dirs)}")
        
        # Check for at least one camera
        camera_dirs = [d for d in cameras_dir.iterdir() if d.is_dir()]
        if not camera_dirs:
            raise ValueError("No camera directories found in cameras/")
        
        # Check required sensor files
        gaze_data_file = sensors_dir / "gaze_data.csv"
        if not gaze_data_file.exists():
            raise FileNotFoundError("Required file missing: sensors/gaze_data.csv")
        
        # Check optional sensor files
        imu_data_file = sensors_dir / "imu_data.csv"
        if not imu_data_file.exists():
            self.show_warning("Optional file missing: sensors/imu_data.csv (IMU visualization disabled)")
        
        # Validate each camera directory with warnings for optional files
        warnings = []
        for camera_dir in camera_dirs:
            camera_name = camera_dir.name
            
            # Required: frame metadata
            frame_metadata = camera_dir / "frame_metadata.csv"
            if not frame_metadata.exists():
                raise FileNotFoundError(f"Camera {camera_name} missing required frame_metadata.csv")
            
            # Required: at least one frame source
            frames_dir = camera_dir / "frames"
            mlcf_file = camera_dir / "camera_frames.mlcf"
            
            if not frames_dir.exists() and not mlcf_file.exists():
                raise FileNotFoundError(
                    f"Camera {camera_name} missing both frames/ directory and camera_frames.mlcf"
                )
            
            # Optional: gaze screen coordinates
            gaze_screen_coords = camera_dir / "gaze_screen_coords.csv"
            if not gaze_screen_coords.exists():
                warnings.append(f"Camera {camera_name} missing gaze_screen_coords.csv (2D gaze overlay disabled)")
        
        # Show warnings for missing optional files
        if warnings:
            for warning in warnings:
                self.show_warning(warning)
    
    def _load_session_info(self, session_path: Path) -> None:
        """Load session information and update UI for camera selection."""
        try:
            # Try to load metadata for camera info
            metadata_file = session_path / "metadata.json"
            if metadata_file.exists():
                metadata = SessionMetadata.from_json_file(metadata_file)
                self.cameras = metadata.get_camera_names()
            else:
                # Get cameras from directory structure
                cameras_dir = session_path / "cameras"
                self.cameras = [d.name for d in cameras_dir.iterdir() if d.is_dir()]
            
            # Update camera selection UI if multiple cameras
            self._show_camera_selection_ui()
            
        except Exception as e:
            # Continue without camera selection - basic validation passed
            self.cameras = []
    
    def _show_camera_selection_ui(self) -> None:
        """Show camera selection UI after successful directory validation."""
        
        try:
            camera_container = self.query_one("#camera_selection")
            if len(self.cameras) <= 1:
                # Hide camera selection if only one or no cameras
                camera_container.display = False
                # Set the single camera as primary if available
                if len(self.cameras) == 1:
                    self.config.primary_camera = self.cameras[0]
                    self.show_success(f"Single camera found: {self.cameras[0]}")
            else:
                # Show camera selection for multiple cameras
                camera_container.display = True
                
                # Update Select dropdown
                try:
                    camera_select = self.query_one("#primary_camera", Select)
                    
                    # Create options for the Select widget (label, value pairs)
                    options = [(camera_name, camera_name) for camera_name in self.cameras]
                    
                    # Set the options
                    camera_select.set_options(options)
                    
                    # Set first camera as default
                    if self.cameras:
                        camera_select.value = self.cameras[0]
                        self.config.primary_camera = self.cameras[0]
                        self.show_success(f"Multiple cameras found. Primary set to: {self.cameras[0]}")
                        
                except Exception as e:
                    self.show_error(f"Failed to populate camera selection: {e}")
                    
        except Exception as e:
            self.show_error(f"Camera selection UI error: {e}")
    
    def _hide_camera_selection_ui(self) -> None:
        """Hide camera selection UI when directory becomes invalid."""
        try:
            camera_container = self.query_one("#camera_selection")
            camera_container.display = False
        except Exception as e:
            pass  # Silently handle UI cleanup errors
        self.cameras = []
        self.config.primary_camera = ""
    
    def _create_camera_selection_section(self):
        """Create the camera selection section (initially hidden)."""
        yield Static("Primary Camera (3D View)", classes="section-title", id="camera_section_title")
        with Container(classes="section-content", id="camera_selection") as container:
            container.display = False  # Initially hidden until session is validated
            yield Static("Select primary camera for 3D visualization:", id="camera_instruction")
            yield Select(
                options=[],  # Empty initially, will be populated dynamically
                prompt="Select a camera",
                id="primary_camera"
            )
    
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()
        
        with ScrollableContainer():
            yield Static("MLGaze Viewer Configuration", classes="section-title")
            
            with Container(classes="grid-container"):
                # Left Column - File and Temporal Settings  
                with Container(classes="left-column"):
                    yield Static("Data Files", classes="section-title")
                    with Container(classes="section-content"):
                        yield Static("Input Directory:")
                        with Horizontal():
                            yield Input(
                                placeholder="Select session directory...", 
                                value="", 
                                id="input_directory"
                            )
                            yield Button("Browse", id="browse_directory")
                    
                    # Camera Selection Section (dynamic based on available cameras)
                    yield from self._create_camera_selection_section()
                    
                    yield Static("Trail Effects", classes="section-title")
                    with Container(classes="section-content"):
                        yield Checkbox("Enable Trail Fade", value=True, id="fade_trail")
                        
                        yield Static("Fade Duration (seconds):")
                        yield Input(
                            placeholder="5.0", 
                            value="5.0", 
                            id="fade_duration"
                        )
                    
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
    
    
    @on(Select.Changed)
    def camera_selection_changed(self, event: Select.Changed) -> None:
        """Handle primary camera selection changes."""
        if event.select.id == "primary_camera":
            selected_camera = event.value
            if selected_camera:  # Ensure a value was selected
                self.config.primary_camera = selected_camera
    
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
        elif checkbox_id == "window_gaze":
            self.config.sliding_window_3d_gaze = value
        elif checkbox_id == "window_trajectory":
            self.config.sliding_window_3d_trajectory = value
        elif checkbox_id == "window_camera":
            self.config.sliding_window_camera = value
        elif checkbox_id == "show_imu":
            self.config.show_imu_data = value
    
    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        """Handle input field changes."""
        input_id = event.input.id
        value = event.value
        
        try:
            if input_id == "input_directory":
                self.config.input_directory = value if value else ""
                # Reset session validation state when directory changes
                if not self.session_validated or value != self.config.input_directory:
                    self.session_validated = False
                    self._hide_camera_selection_ui()
                
                # Validate directory path with user feedback
                if value and len(value) > 3:  # Avoid validating every keystroke
                    self.validate_directory(value)
                elif not value:
                    # Clear validation when input is empty
                    input_field = self.query_one("#input_directory", Input)
                    input_field.remove_class("valid-path", "invalid-path", "warning-path")
                    
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
        if not self.session_validated or not self.config.input_directory:
            self.show_error("Please select a valid session directory first")
            return
        
        self.exit(self.config)
    
    @on(Button.Pressed, "#reset")
    def reset_to_defaults(self) -> None:
        """Handle Reset button."""
        self.config = VisualizationConfig()
        
        # Reset session validation
        self.session_validated = False
        self._hide_camera_selection_ui()
        
        # Update UI elements with default values
        self.query_one("#input_directory", Input).value = ""
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
    
    def on_directory_selected(self, result: str | None) -> None:
        """Handle directory selection result from browser."""
        if result:  # User selected a directory
            try:
                input_field = self.query_one("#input_directory", Input)
                input_field.value = result
                self.config.input_directory = result
                # Validate the selected directory
                self.validate_directory(result)
            except Exception as e:
                self.show_error(f"Error updating directory selection: {e}")


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