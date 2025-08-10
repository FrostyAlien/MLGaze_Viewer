"""Area of Interest (AOI) analyzer plugin for MLGaze Viewer."""

import numpy as np
import pandas as pd
import rerun as rr
from typing import Dict, Any, List, Optional
from src.analytics.base import AnalyticsPlugin
from src.core import SessionData, BoundingBox
from src.core.coordinate_utils import unity_to_rerun_position


class AOIAnalyzer(AnalyticsPlugin):
    """Analyze gaze behavior within defined Areas of Interest."""
    
    def __init__(self, regions: Optional[List[BoundingBox]] = None):
        """Initialize AOI analyzer.
        
        Args:
            regions: List of BoundingBox objects defining AOIs
                    Can be 2D (screen space) or 3D (world space)
        """
        super().__init__("AOI Analyzer")
        self.regions = regions or []
    
    def add_region(self, region: BoundingBox) -> None:
        """Add a new AOI region.
        
        Args:
            region: BoundingBox defining the AOI
        """
        self.regions.append(region)
    
    def clear_regions(self) -> None:
        """Clear all AOI regions."""
        self.regions = []
    
    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict:
        """Analyze gaze behavior within AOIs.
        
        Args:
            session: SessionData containing gaze information
            config: Optional configuration with AOI definitions
            
        Returns:
            Dictionary containing AOI analysis results
        """
        if config and 'regions' in config:
            # Load regions from config if provided
            self.regions = [
                BoundingBox(
                    name=r['name'],
                    bounds=np.array(r['bounds']),
                    category=r.get('category', 'default')
                )
                for r in config['regions']
            ]
        
        if not self.regions:
            return {
                'error': 'No AOI regions defined',
                'total_samples': len(session.gaze)
            }
        
        results = {
            'regions': {},
            'transitions': [],
            'summary': {}
        }
        
        # Analyze each region
        for region in self.regions:
            region_results = self._analyze_region(session, region)
            results['regions'][region.name] = region_results
        
        # Analyze transitions between AOIs
        results['transitions'] = self._analyze_transitions(session)
        
        # Generate summary statistics
        results['summary'] = self._generate_summary(results['regions'])
        
        self.results = results
        return results
    
    def _analyze_region(self, session: SessionData, region: BoundingBox) -> Dict:
        """Analyze gaze behavior for a single AOI.
        
        Args:
            session: SessionData containing gaze information
            region: BoundingBox defining the AOI
            
        Returns:
            Dictionary with region-specific metrics
        """
        metrics = {
            'total_dwell_time_ms': 0,
            'fixation_count': 0,
            'fixation_duration_ms': 0,
            'first_fixation_time_ms': None,
            'entry_count': 0,
            'exit_count': 0,
            'samples_in_aoi': 0,
            'percentage_of_time': 0,
            'gaze_state_distribution': {}
        }
        
        # Track if gaze is currently in AOI
        currently_in_aoi = False
        entry_time = None
        
        for idx, row in session.gaze.iterrows():
            # Determine if gaze is in AOI
            if region.is_3d:
                # Use 3D gaze position
                if row.get('hasHitTarget', False):
                    pos_unity = [row['gazePositionX'], row['gazePositionY'], row['gazePositionZ']]
                    pos = unity_to_rerun_position(pos_unity)
                    in_aoi = region.contains_point(np.array(pos))
                else:
                    in_aoi = False
            else:
                # Use 2D screen coordinates
                if row.get('isValidProjection', False) and not pd.isna(row.get('screenPixelX')):
                    screen_point = np.array([row['screenPixelX'], row['screenPixelY']])
                    in_aoi = region.contains_point(screen_point)
                else:
                    in_aoi = False
            
            # Track entries and exits
            if in_aoi and not currently_in_aoi:
                metrics['entry_count'] += 1
                entry_time = row['timestamp']
                currently_in_aoi = True
                
                # Track first fixation
                if metrics['first_fixation_time_ms'] is None and row.get('gazeState') == 'Fixation':
                    metrics['first_fixation_time_ms'] = (row['timestamp'] - session.start_timestamp) / 1e6
            
            elif not in_aoi and currently_in_aoi:
                metrics['exit_count'] += 1
                if entry_time is not None:
                    dwell_time = (row['timestamp'] - entry_time) / 1e6  # Convert to ms
                    metrics['total_dwell_time_ms'] += dwell_time
                currently_in_aoi = False
            
            # Count samples and track gaze states
            if in_aoi:
                metrics['samples_in_aoi'] += 1
                
                # Track gaze state distribution
                state = row.get('gazeState', 'Unknown')
                if state not in metrics['gaze_state_distribution']:
                    metrics['gaze_state_distribution'][state] = 0
                metrics['gaze_state_distribution'][state] += 1
                
                # Count fixations
                if state == 'Fixation':
                    # Simple counting - could be enhanced with fixation grouping
                    if idx == 0 or session.gaze.iloc[idx-1].get('gazeState') != 'Fixation':
                        metrics['fixation_count'] += 1
        
        # Calculate percentages
        total_samples = len(session.gaze)
        if total_samples > 0:
            metrics['percentage_of_time'] = (metrics['samples_in_aoi'] / total_samples) * 100
        
        # Calculate average fixation duration if fixations exist
        if 'Fixation' in metrics['gaze_state_distribution'] and metrics['fixation_count'] > 0:
            # Estimate based on samples (this is simplified)
            fixation_samples = metrics['gaze_state_distribution']['Fixation']
            avg_sample_duration = session.duration * 1000 / total_samples  # ms per sample
            metrics['fixation_duration_ms'] = fixation_samples * avg_sample_duration / metrics['fixation_count']
        
        return metrics
    
    def _analyze_transitions(self, session: SessionData) -> List[Dict]:
        """Analyze transitions between AOIs.
        
        Args:
            session: SessionData containing gaze information
            
        Returns:
            List of transition events
        """
        transitions = []
        current_aoi = None
        
        for idx, row in session.gaze.iterrows():
            # Find which AOI contains current gaze (if any)
            gaze_aoi = None
            
            for region in self.regions:
                if region.is_3d and row.get('hasHitTarget', False):
                    pos_unity = [row['gazePositionX'], row['gazePositionY'], row['gazePositionZ']]
                    pos = unity_to_rerun_position(pos_unity)
                    if region.contains_point(np.array(pos)):
                        gaze_aoi = region.name
                        break
                elif not region.is_3d and row.get('isValidProjection', False):
                    if not pd.isna(row.get('screenPixelX')):
                        screen_point = np.array([row['screenPixelX'], row['screenPixelY']])
                        if region.contains_point(screen_point):
                            gaze_aoi = region.name
                            break
            
            # Track transition
            if gaze_aoi != current_aoi:
                transitions.append({
                    'timestamp': row['timestamp'],
                    'from_aoi': current_aoi,
                    'to_aoi': gaze_aoi,
                    'gaze_state': row.get('gazeState', 'Unknown')
                })
                current_aoi = gaze_aoi
        
        return transitions
    
    def _generate_summary(self, region_results: Dict) -> Dict:
        """Generate summary statistics across all AOIs.
        
        Args:
            region_results: Dictionary of per-region results
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'total_aois': len(region_results),
            'most_viewed_aoi': None,
            'least_viewed_aoi': None,
            'total_fixations': 0,
            'average_dwell_time_ms': 0
        }
        
        if not region_results:
            return summary
        
        # Find most and least viewed AOIs
        max_time = 0
        min_time = float('inf')
        total_dwell = 0
        
        for name, metrics in region_results.items():
            dwell_time = metrics['total_dwell_time_ms']
            total_dwell += dwell_time
            summary['total_fixations'] += metrics['fixation_count']
            
            if dwell_time > max_time:
                max_time = dwell_time
                summary['most_viewed_aoi'] = name
            
            if dwell_time < min_time:
                min_time = dwell_time
                summary['least_viewed_aoi'] = name
        
        # Calculate average
        if len(region_results) > 0:
            summary['average_dwell_time_ms'] = total_dwell / len(region_results)
        
        return summary
    
    def visualize(self, results: Dict, rr_stream=None) -> None:
        """Visualize AOI regions and metrics in Rerun.
        
        Args:
            results: Analysis results from process method
            rr_stream: Optional Rerun recording stream
        """
        if not self.regions:
            return
        
        # Visualize AOI boundaries
        for region in self.regions:
            if region.is_3d:
                # TODO: Visualize 3D bounding boxes
                pass
            else:
                # Visualize 2D regions on camera image
                rr.log(
                    f"{self.entity_path}/regions/{region.name}",
                    rr.Boxes2D(
                        centers=[region.center],
                        sizes=[region.bounds[2:]],  # width, height
                        labels=[region.name],
                        colors=[[255, 255, 0, 128]]  # Semi-transparent yellow
                    ),
                    static=True
                )
        
        # Log metrics as text
        if 'summary' in results:
            summary_text = self.get_summary(results)
            rr.log(
                f"{self.entity_path}/summary",
                rr.TextDocument(summary_text),
                static=True
            )
    
    def get_summary(self, results: Dict) -> str:
        """Generate text summary of AOI analysis.
        
        Args:
            results: Analysis results
            
        Returns:
            Human-readable summary
        """
        lines = ["AOI Analysis Summary", "=" * 40]
        
        if 'error' in results:
            lines.append(f"Error: {results['error']}")
            return "\n".join(lines)
        
        # Overall summary
        summary = results.get('summary', {})
        lines.append(f"Total AOIs analyzed: {summary.get('total_aois', 0)}")
        lines.append(f"Most viewed: {summary.get('most_viewed_aoi', 'N/A')}")
        lines.append(f"Total fixations: {summary.get('total_fixations', 0)}")
        lines.append("")
        
        # Per-region details
        for name, metrics in results.get('regions', {}).items():
            lines.append(f"{name}:")
            lines.append(f"  Time in AOI: {metrics['percentage_of_time']:.1f}%")
            lines.append(f"  Dwell time: {metrics['total_dwell_time_ms']:.0f} ms")
            lines.append(f"  Fixations: {metrics['fixation_count']}")
            lines.append(f"  Entries: {metrics['entry_count']}")
            lines.append("")
        
        return "\n".join(lines)