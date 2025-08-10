"""
Example Analytics Plugin Template for MLGaze Viewer

This template demonstrates how to create custom analytics plugins
that integrate with the MLGaze Viewer framework.

To use this template:
1. Copy this file and rename it for your plugin
2. Implement the process() method with your analysis logic
3. Optionally implement visualize() for custom Rerun visualizations
4. Register your plugin in src/analytics/__init__.py or load dynamically
"""

import numpy as np
import pandas as pd
import rerun as rr
from typing import Dict, Any, Optional

# Import from parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.base import AnalyticsPlugin
from src.core import SessionData


class ExamplePlugin(AnalyticsPlugin):
    """Example plugin that analyzes gaze velocity and acceleration patterns.
    
    This example plugin demonstrates:
    - Processing gaze data to compute derived metrics
    - Working with different gaze states
    - Creating custom visualizations in Rerun
    - Generating summary statistics
    """
    
    def __init__(self, threshold_velocity: float = 30.0):
        """Initialize the example plugin.
        
        Args:
            threshold_velocity: Velocity threshold in degrees/second
        """
        super().__init__("Example Gaze Velocity Analyzer")
        self.threshold_velocity = threshold_velocity
    
    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict:
        """Analyze gaze velocity and acceleration patterns.
        
        Args:
            session: SessionData containing all sensor data
            config: Optional configuration dictionary
            
        Returns:
            Dictionary containing analysis results
        """
        if session.gaze.empty:
            return {'error': 'No gaze data available'}
        
        results = {
            'velocities': [],
            'accelerations': [],
            'high_velocity_events': [],
            'statistics': {},
            'state_velocities': {}
        }
        
        # Calculate velocities between consecutive samples
        prev_row = None
        velocities_by_state = {}
        
        for idx, row in session.gaze.iterrows():
            if prev_row is not None and row.get('hasHitTarget', False):
                # Calculate time delta in seconds
                dt = (row['timestamp'] - prev_row['timestamp']) / 1e9
                
                if dt > 0:
                    # Calculate angular velocity (simplified)
                    dx = row['gazePositionX'] - prev_row['gazePositionX']
                    dy = row['gazePositionY'] - prev_row['gazePositionY']
                    dz = row['gazePositionZ'] - prev_row['gazePositionZ']
                    
                    # Simple Euclidean distance as proxy for angular change
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # Convert to approximate degrees (this is simplified)
                    # In reality, you'd convert to visual angles properly
                    angular_distance = np.degrees(np.arctan(distance / 2.0))  # Assuming 2m viewing distance
                    velocity = angular_distance / dt
                    
                    results['velocities'].append({
                        'timestamp': row['timestamp'],
                        'velocity': velocity,
                        'state': row.get('gazeState', 'Unknown')
                    })
                    
                    # Track velocities by state
                    state = row.get('gazeState', 'Unknown')
                    if state not in velocities_by_state:
                        velocities_by_state[state] = []
                    velocities_by_state[state].append(velocity)
                    
                    # Detect high velocity events
                    if velocity > self.threshold_velocity:
                        results['high_velocity_events'].append({
                            'timestamp': row['timestamp'],
                            'velocity': velocity,
                            'state': state
                        })
            
            prev_row = row
        
        # Calculate statistics
        if results['velocities']:
            all_velocities = [v['velocity'] for v in results['velocities']]
            results['statistics'] = {
                'mean_velocity': float(np.mean(all_velocities)),
                'std_velocity': float(np.std(all_velocities)),
                'max_velocity': float(np.max(all_velocities)),
                'min_velocity': float(np.min(all_velocities)),
                'median_velocity': float(np.median(all_velocities)),
                'high_velocity_count': len(results['high_velocity_events'])
            }
            
            # Statistics by gaze state
            for state, velocities in velocities_by_state.items():
                results['state_velocities'][state] = {
                    'mean': float(np.mean(velocities)),
                    'std': float(np.std(velocities)),
                    'max': float(np.max(velocities)),
                    'count': len(velocities)
                }
        
        return results
    
    def visualize(self, results: Dict, rr_stream=None) -> None:
        """Visualize velocity analysis results in Rerun.
        
        Args:
            results: Analysis results from process method
            rr_stream: Optional Rerun recording stream
        """
        if 'error' in results:
            return
        
        # Log velocity time series
        if results['velocities']:
            timestamps = [v['timestamp'] for v in results['velocities']]
            velocities = [v['velocity'] for v in results['velocities']]
            
            # Create time column
            times = rr.TimeColumn("timestamp", timestamp=np.array(timestamps) * 1e-9)
            
            # Log velocity scalar time series
            rr.send_columns(
                f"{self.entity_path}/velocity",
                indexes=[times],
                columns=rr.Scalars.columns(scalars=velocities)
            )
            
            # Mark high velocity events
            for event in results['high_velocity_events']:
                rr.set_time("timestamp", timestamp=event['timestamp'] * 1e-9)
                rr.log(
                    f"{self.entity_path}/high_velocity_markers",
                    rr.Points3D(
                        positions=[[0, 0, event['velocity'] / 100]],  # Scale for visibility
                        colors=[[255, 0, 0]],
                        radii=0.02
                    )
                )
        
        # Log summary statistics as text
        if results['statistics']:
            summary_text = self.get_summary(results)
            rr.log(
                f"{self.entity_path}/summary",
                rr.TextDocument(summary_text),
                static=True
            )
    
    def get_summary(self, results: Dict) -> str:
        """Generate a text summary of the velocity analysis.
        
        Args:
            results: Analysis results
            
        Returns:
            Human-readable summary string
        """
        if 'error' in results:
            return f"Error: {results['error']}"
        
        lines = [
            "Gaze Velocity Analysis",
            "=" * 40
        ]
        
        stats = results.get('statistics', {})
        if stats:
            lines.extend([
                f"Mean velocity: {stats.get('mean_velocity', 0):.1f} deg/s",
                f"Max velocity: {stats.get('max_velocity', 0):.1f} deg/s",
                f"High velocity events: {stats.get('high_velocity_count', 0)}",
                f"  (threshold: {self.threshold_velocity} deg/s)",
                ""
            ])
        
        # State-specific statistics
        state_vels = results.get('state_velocities', {})
        if state_vels:
            lines.append("Velocity by gaze state:")
            for state, metrics in state_vels.items():
                lines.append(f"  {state}:")
                lines.append(f"    Mean: {metrics['mean']:.1f} deg/s")
                lines.append(f"    Max: {metrics['max']:.1f} deg/s")
                lines.append(f"    Samples: {metrics['count']}")
        
        return "\n".join(lines)
    
    def validate_data(self, session: SessionData) -> bool:
        """Check if the session data is valid for velocity analysis.
        
        Args:
            session: SessionData to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if session.gaze.empty:
            return False
        
        # Check for required columns
        required_columns = [
            'timestamp', 'gazePositionX', 'gazePositionY', 'gazePositionZ'
        ]
        
        for col in required_columns:
            if col not in session.gaze.columns:
                print(f"Missing required column: {col}")
                return False
        
        return True


# Example of a more complex plugin combining multiple analyses
class AttentionMetricsPlugin(AnalyticsPlugin):
    """Advanced plugin that combines multiple attention metrics.
    
    This demonstrates how to create more sophisticated analytics
    that could integrate with object detection, AOI analysis, etc.
    """
    
    def __init__(self):
        """Initialize the attention metrics plugin."""
        super().__init__("Attention Metrics Analyzer")
    
    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict:
        """Calculate comprehensive attention metrics.
        
        This is a placeholder for more complex analysis that could include:
        - Fixation/saccade ratio
        - Attention distribution entropy
        - Scanpath complexity
        - Cognitive load estimation
        - Task engagement metrics
        """
        results = {
            'metrics': {
                'fixation_saccade_ratio': 0,
                'attention_entropy': 0,
                'scanpath_length': 0,
                'revisits': 0
            },
            'temporal_patterns': [],
            'spatial_distribution': {}
        }
        
        # TODO: Implement sophisticated attention metrics
        # This would involve:
        # 1. Temporal analysis of gaze patterns
        # 2. Spatial distribution analysis
        # 3. Integration with AOI data if available
        # 4. Pattern recognition in scanpaths
        # 5. Cognitive load indicators
        
        return results


if __name__ == "__main__":
    # Example of testing the plugin standalone
    from src.utils import DataLoader
    
    print("Loading test data...")
    loader = DataLoader()
    session = loader.load_session("input/0806_Bowen_Office")
    
    print("Running example plugin...")
    plugin = ExamplePlugin(threshold_velocity=50.0)
    
    if plugin.validate_data(session):
        results = plugin.process(session)
        print("\nResults:")
        print(plugin.get_summary(results))
    else:
        print("Data validation failed")