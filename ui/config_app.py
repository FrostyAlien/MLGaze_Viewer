"""
MLGaze Viewer TUI Configuration App

Textual-based terminal UI for configuring visualization parameters.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Input, Checkbox, RadioButton, RadioSet, DirectoryTree
)
from textual.screen import ModalScreen


@dataclass
class VisualizationConfig:
    """Configuration settings for gaze visualization."""
    
    # File paths
    input_directory: str = "input"
    gaze_csv_file: str = ""  # Will be auto-detected or selected
    metadata_csv_file: str = ""  # Will be auto-detected or selected
    frames_directory: str = ""  # Will be auto-detected or selected
    
    # Temporal settings
    temporal_mode: str = "all"  # "all", "recent_frames", "recent_seconds"
    recent_frames: int = 500
    recent_seconds: float = 30.0
    enable_fade_trail: bool = True
    fade_duration: float = 5.0
    
    # Visualization toggles
    show_point_cloud: bool = True
    show_gaze_trajectory: bool = True
    show_camera_trajectory: bool = True
    color_by_gaze_state: bool = True
    test_y_flip: bool = False


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
    
    @on(DirectoryTree.DirectorySelected)
    def directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Handle double-click or Enter on directory - auto-select."""
        try:
            selected_path = str(event.path)
            self.app.log(f"Directory selected via double-click: {selected_path}")
            self.dismiss(selected_path)
        except Exception as e:
            self.app.log(f"Directory selection error: {e}")
    
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
        
        # Check for required data files
        required_patterns = [
            "*gaze_screen_coords*.csv",
            "frame_metadata.csv",
            "frames/"
        ]
        
        missing_files = []
        for pattern in required_patterns:
            matches = list(path_obj.glob(pattern))
            if not matches:
                missing_files.append(pattern)
        
        if missing_files:
            self.show_warning(f"Directory selected, but missing expected files: {', '.join(missing_files)}")
            input_field.remove_class("valid-path", "invalid-path")
            input_field.add_class("warning-path")
            # Don't block selection - user might have different file structure
            return True
        
        self.show_success(f"Valid MLGaze directory selected: {path}")
        input_field.remove_class("invalid-path", "warning-path")
        input_field.add_class("valid-path")
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
                    
                    # Temporal Settings Section
                    yield Static("Temporal Display", classes="section-title")
                    with Container(classes="section-content"):
                        yield RadioSet(
                            RadioButton("Show All Points", value=True, id="mode_all"),
                            RadioButton("Recent Frames", id="mode_frames"),
                            RadioButton("Recent Seconds", id="mode_seconds"),
                            id="temporal_mode"
                        )
                        
                        yield Static("Recent Frames:")
                        yield Input(
                            placeholder="500", 
                            value="500", 
                            id="recent_frames"
                        )
                        
                        yield Static("Recent Seconds:")
                        yield Input(
                            placeholder="30.0", 
                            value="30.0", 
                            id="recent_seconds"
                        )
                        
                        yield Checkbox("Enable Trail Fade", value=True, id="fade_trail")
                        
                        yield Static("Fade Duration (seconds):")
                        yield Input(
                            placeholder="5.0", 
                            value="5.0", 
                            id="fade_duration"
                        )
                
                # Right Column - Visualization Settings
                with Container(classes="right-column"):
                    yield Static("Visualization Options", classes="section-title")
                    with Container(classes="section-content"):
                        yield Checkbox("Show Point Cloud", value=True, id="point_cloud")
                        yield Checkbox("Show Gaze Trajectory", value=True, id="gaze_trajectory")
                        yield Checkbox("Show Camera Path", value=True, id="camera_path")
                        yield Checkbox("Color by Gaze State", value=True, id="color_by_state")
                        yield Checkbox("Test Y-Flip", value=False, id="y_flip")
            
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
    
    @on(RadioSet.Changed, "#temporal_mode")
    def temporal_mode_changed(self, event: RadioSet.Changed) -> None:
        """Handle temporal mode selection."""
        if event.pressed.id == "mode_all":
            self.config.temporal_mode = "all"
        elif event.pressed.id == "mode_frames":
            self.config.temporal_mode = "recent_frames"
        elif event.pressed.id == "mode_seconds":
            self.config.temporal_mode = "recent_seconds"
    
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
                    
            elif input_id == "recent_frames":
                if value:
                    frames = int(value)
                    if frames < 1:
                        self.show_warning("Recent frames must be at least 1")
                    elif frames > 10000:
                        self.show_warning("Recent frames value is very high - may impact performance")
                    else:
                        self.config.recent_frames = frames
                else:
                    self.config.recent_frames = 500
                    
            elif input_id == "recent_seconds":
                if value:
                    seconds = float(value)
                    if seconds < 0.1:
                        self.show_warning("Recent seconds must be at least 0.1")
                    elif seconds > 300:
                        self.show_warning("Recent seconds value is very high - may impact performance")
                    else:
                        self.config.recent_seconds = seconds
                else:
                    self.config.recent_seconds = 30.0
                    
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
                    
        except ValueError as e:
            if "recent_frames" in input_id:
                self.show_error("Recent frames must be a whole number")
            elif "recent_seconds" in input_id or "fade_duration" in input_id:
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
        self.query_one("#recent_frames", Input).value = "500" 
        self.query_one("#recent_seconds", Input).value = "30.0"
        self.query_one("#fade_duration", Input).value = "5.0"
        
        # Reset radio buttons
        self.query_one("#mode_all", RadioButton).value = True
        self.query_one("#mode_frames", RadioButton).value = False  
        self.query_one("#mode_seconds", RadioButton).value = False
        
        # Reset checkboxes
        self.query_one("#fade_trail", Checkbox).value = True
        self.query_one("#point_cloud", Checkbox).value = True
        self.query_one("#gaze_trajectory", Checkbox).value = True
        self.query_one("#camera_path", Checkbox).value = True
        self.query_one("#color_by_state", Checkbox).value = True
        self.query_one("#y_flip", Checkbox).value = False
    
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
        print(f"  Temporal mode: {config.temporal_mode}")
        print(f"  Show point cloud: {config.show_point_cloud}")
        print(f"  Show trajectories: {config.show_gaze_trajectory}")
    else:
        print("Configuration cancelled")