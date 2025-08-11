"""Session management utilities for MLGaze Viewer."""

from pathlib import Path
from typing import List


def get_session_directories(gaze_files: List[Path], metadata_files: List[Path], input_dir: Path) -> List[Path]:
    """Get main session directories that should be recommended to users.
    
    This function identifies the primary session directories by analyzing the location
    of gaze and metadata files. It prioritizes gaze file parent directories since
    these are always at the session level, and filters out nested subdirectories
    to provide clean, actionable recommendations.
    
    Args:
        gaze_files: List of gaze CSV file paths found
        metadata_files: List of metadata CSV file paths found  
        input_dir: Input directory path being searched
        
    Returns:
        Sorted list of session directory paths that users should select
        
    Example:
        Given files:
        - input/0806_Bowen_Cafe/gaze_screen_coords.csv
        - input/0806_Bowen_Cafe/test_frames/frame_metadata.csv
        - input/0806_Bowen_Office/gaze_screen_coords.csv
        - input/0806_Bowen_Office/frame_metadata.csv
        
        Returns: [input/0806_Bowen_Cafe, input/0806_Bowen_Office]
        (Not: input/0806_Bowen_Cafe/test_frames)
    """
    session_dirs = set()
    
    # Primary: Use gaze file parent directories (these are always session-level)
    for gaze_file in gaze_files:
        parent = gaze_file.parent
        # Only include direct children of input_dir, not nested subdirectories
        if parent.parent == input_dir:
            session_dirs.add(parent)
    
    # Secondary: Add metadata file parents if they're also direct children
    for metadata_file in metadata_files:
        parent = metadata_file.parent
        # Check if this is a direct child of input_dir
        if parent.parent == input_dir:
            session_dirs.add(parent)
        # If metadata is in a subdirectory, use its parent's parent if it's a direct child
        elif parent.parent.parent == input_dir:
            session_dirs.add(parent.parent)
    
    # Sort and return as list
    return sorted(session_dirs)