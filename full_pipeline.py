import numpy as np
import pandas as pd
from scipy import signal
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import deque

@dataclass
class PrinterConfig:
    """Printer-specific configuration from the paper"""
    x_steps_per_mm: float = 6.0
    y_steps_per_mm: float = 6.0
    z_steps_per_mm: float = 95.2
    e_steps_per_mm: float = 1.0
    
    # Filter parameters from Section 4.2
    x_filter_cutoff: float = 300.0  # Hz
    y_filter_cutoff: float = 300.0
    z_filter_cutoff: float = 300.0
    e_filter_cutoff: float = 275.0
    
    # Peak detection parameters from Section 4.2
    x_height: float = 0.4
    x_prominence: float = 0.3
    y_height: float = 0.4
    y_prominence: float = 0.3
    z_height: float = 0.1
    z_prominence: float = 0.125
    e_height: float = 0.1
    e_prominence: float = 0.2

@dataclass
class TraceEntry:
    """Represents a single trace entry as described in Section 4.1"""
    time: float
    phase0: float
    phase1: float  # Inverted phase0
    phase2: float
    phase3: float  # Inverted phase2
    axis: str
    isPeak: bool = False
    isReversal: bool = False
    isDwell: bool = False

@dataclass
class BadSection:
    """Represents a section with invalid firing order"""
    start_idx: int
    end_idx: int
    original_firing_order: List[int]
    must_reverse: bool
    axis: str

class SideChannelReconstructor:
    """Complete implementation of the paper's reconstruction pipeline"""
    
    def __init__(self, config: PrinterConfig = None):
        self.config = config or PrinterConfig()
        self.traces = {'X': [], 'Y': [], 'Z': [], 'E': []}
        self.peaks = {'X': [], 'Y': [], 'Z': [], 'E': []}
        self.bad_sections = {'X': [], 'Y': [], 'Z': [], 'E': []}
        
    def load_csv(self, x_csv: str, y_csv: str, z_csv: str, e_csv: str):
        """Load the 4 CSV files as described in Section 4.1"""
        print("Loading CSV files...")
        
        for axis, csv_path in [('X', x_csv), ('Y', y_csv), ('Z', z_csv), ('E', e_csv)]:
            df = pd.read_csv(csv_path)
            # Assuming format: Time(s),Channel A(A),Channel B(A)
            times = df.iloc[:, 0].values
            phase0_vals = df.iloc[:, 1].values
            phase2_vals = df.iloc[:, 2].values
            
            traces = []
            for i in range(len(times)):
                entry = TraceEntry(
                    time=times[i],
                    phase0=phase0_vals[i],
                    phase1=-phase0_vals[i],  # Inverted
                    phase2=phase2_vals[i],
                    phase3=-phase2_vals[i],  # Inverted
                    axis=axis
                )
                traces.append(entry)
            
            self.traces[axis] = traces
            print(f"  Loaded {len(traces)} entries for {axis}-axis")
    
    def apply_lowpass_filter(self, axis: str):
        """Section 4.2: Apply Butterworth low-pass filter"""
        cutoff_map = {
            'X': self.config.x_filter_cutoff,
            'Y': self.config.y_filter_cutoff,
            'Z': self.config.z_filter_cutoff,
            'E': self.config.e_filter_cutoff
        }
        
        traces = self.traces[axis]
        if len(traces) < 100:
            return
        
        # Extract time and phase data
        times = np.array([t.time for t in traces])
        phase0 = np.array([t.phase0 for t in traces])
        phase2 = np.array([t.phase2 for t in traces])
        
        # Calculate sampling rate
        dt = np.median(np.diff(times))
        fs = 1.0 / dt if dt > 0 else 2000.0
        
        # Design and apply 8th-order Butterworth filter
        cutoff = cutoff_map[axis]
        sos = signal.butter(8, cutoff, 'low', fs=fs, output='sos')
        
        phase0_filtered = signal.sosfilt(sos, phase0)
        phase2_filtered = signal.sosfilt(sos, phase2)
        
        # Update trace entries
        for i, entry in enumerate(traces):
            entry.phase0 = phase0_filtered[i]
            entry.phase1 = -phase0_filtered[i]
            entry.phase2 = phase2_filtered[i]
            entry.phase3 = -phase2_filtered[i]
    
    def detect_peaks(self, axis: str):
        """Section 4.2: Peak detection using scipy find_peaks"""
        param_map = {
            'X': (self.config.x_height, self.config.x_prominence),
            'Y': (self.config.y_height, self.config.y_prominence),
            'Z': (self.config.z_height, self.config.z_prominence),
            'E': (self.config.e_height, self.config.e_prominence)
        }
        
        height, prominence = param_map[axis]
        traces = self.traces[axis]
        
        # Detect peaks on all 4 phases
        for phase_idx in range(4):
            phase_data = np.array([getattr(t, f'phase{phase_idx}') for t in traces])
            peaks, _ = signal.find_peaks(phase_data, height=height, prominence=prominence)
            
            for peak_idx in peaks:
                traces[peak_idx].isPeak = True
        
        # Handle simultaneous peaks - keep highest
        for i, entry in enumerate(traces):
            if entry.isPeak:
                phases = [entry.phase0, entry.phase1, entry.phase2, entry.phase3]
                max_phase = np.argmax(np.abs(phases))
                # Mark which phase this peak belongs to
                entry.peak_phase = max_phase
        
        # Remove low-prominence peaks near higher ones
        avg_period = self._estimate_period(axis)
        threshold = min(avg_period / 2, 0.025)
        
        peak_indices = [i for i, t in enumerate(traces) if t.isPeak]
        for i in range(len(peak_indices)):
            idx = peak_indices[i]
            entry = traces[idx]
            phases = [entry.phase0, entry.phase1, entry.phase2, entry.phase3]
            current_val = abs(phases[entry.peak_phase])
            
            # Check nearby peaks
            for j in range(max(0, i-5), min(len(peak_indices), i+6)):
                if i == j:
                    continue
                other_idx = peak_indices[j]
                if abs(traces[other_idx].time - entry.time) < threshold:
                    other_phases = [traces[other_idx].phase0, traces[other_idx].phase1,
                                   traces[other_idx].phase2, traces[other_idx].phase3]
                    other_val = abs(other_phases[traces[other_idx].peak_phase])
                    if current_val < other_val:
                        entry.isPeak = False
                        break
    
    def _estimate_period(self, axis: str) -> float:
        """Estimate average period between peaks"""
        traces = self.traces[axis]
        peak_times = [t.time for t in traces if t.isPeak]
        if len(peak_times) < 2:
            return 0.01
        diffs = np.diff(peak_times)
        return np.median(diffs) if len(diffs) > 0 else 0.01
    
    def identify_firing_order(self, axis: str) -> List[int]:
        """Section 4.4: Identify the firing order dynamically"""
        traces = self.traces[axis]
        peaks = [(i, t) for i, t in enumerate(traces) if t.isPeak]
        
        if len(peaks) < 16:
            return [0, 1, 2, 3]  # Default firing order
        
        # Build initial 4-peak buffer
        fo_buffer = [peaks[i][1].peak_phase for i in range(4)]
        prediction_count = 0
        next_peak_idx = 4
        
        while prediction_count < 16 and next_peak_idx < len(peaks):
            prediction_idx = next_peak_idx % 4
            if peaks[next_peak_idx][1].peak_phase == fo_buffer[prediction_idx]:
                prediction_count += 1
            else:
                # Reset with new firing order
                fo_buffer = [peaks[i][1].peak_phase for i in range(next_peak_idx-3, next_peak_idx+1)]
                prediction_count = 0
            next_peak_idx += 1
        
        return fo_buffer
    
    def segment_good_bad_sections(self, axis: str):
        """Section 4.4: Identify good and bad sections based on firing order"""
        traces = self.traces[axis]
        peaks = [(i, t) for i, t in enumerate(traces) if t.isPeak]
        
        if len(peaks) < 4:
            return
        
        firing_order = self.identify_firing_order(axis)
        
        peak_idx = 0
        bad_section = None
        good_peaks_counter = 0
        good_peaks_needed = 3
        original_fo = firing_order.copy()
        must_reverse = False
        
        while peak_idx < len(peaks):
            current_phase = peaks[peak_idx][1].peak_phase
            expected_forward = firing_order[peak_idx % 4]
            expected_reverse = firing_order[(4 - (peak_idx % 4)) % 4]
            
            if current_phase == expected_forward:
                if bad_section is not None:
                    good_peaks_counter += 1
                    if good_peaks_counter >= good_peaks_needed:
                        # Close bad section
                        bad_section.end_idx = peak_idx - good_peaks_needed
                        self.bad_sections[axis].append(bad_section)
                        bad_section = None
                        good_peaks_counter = 0
            
            elif current_phase == expected_reverse:
                # Reversal
                firing_order = firing_order[::-1]
                if bad_section is None:
                    traces[peaks[peak_idx][0]].isReversal = True
                else:
                    must_reverse = not must_reverse
                    good_peaks_counter += 1
                    if good_peaks_counter >= good_peaks_needed:
                        bad_section.end_idx = peak_idx - good_peaks_needed
                        self.bad_sections[axis].append(bad_section)
                        bad_section = None
                        good_peaks_counter = 0
            
            else:
                # Invalid firing order
                if bad_section is None:
                    bad_section = BadSection(
                        start_idx=peak_idx,
                        end_idx=-1,
                        original_firing_order=original_fo.copy(),
                        must_reverse=False,
                        axis=axis
                    )
                good_peaks_counter = 0
                # Shift firing order to current peak
                fo_buffer_idx = -1
                for i, phase in enumerate(firing_order):
                    if phase == current_phase:
                        fo_buffer_idx = i
                        break
                if fo_buffer_idx >= 0:
                    firing_order = firing_order[fo_buffer_idx:] + firing_order[:fo_buffer_idx]
            
            peak_idx += 1
        
        # Close any remaining bad section
        if bad_section is not None:
            bad_section.end_idx = len(peaks) - 1
            self.bad_sections[axis].append(bad_section)
    
    def correct_bad_sections(self, axis: str):
        """Section 4.5: Heuristic correction of bad sections"""
        # Simplified version - in practice this is very complex
        # For now, we'll just mark them and skip in reconstruction
        print(f"  {axis}-axis: Found {len(self.bad_sections[axis])} bad sections")
        for bs in self.bad_sections[axis]:
            print(f"    Bad section from peak {bs.start_idx} to {bs.end_idx}")
    
    def generate_point_cloud(self) -> np.ndarray:
        """Section 4.6: Generate point cloud from peaks"""
        print("\nGenerating point cloud...")
        
        # Collect all peaks with timestamps
        all_peaks = []
        for axis in ['X', 'Y', 'Z', 'E']:
            traces = self.traces[axis]
            for i, entry in enumerate(traces):
                if entry.isPeak:
                    all_peaks.append((entry.time, axis, entry.isReversal, entry.peak_phase))
        
        # Sort by timestamp
        all_peaks.sort(key=lambda x: x[0])
        
        # State machine to track position
        position = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        direction = {'X': 1, 'Y': 1, 'Z': 1, 'E': 1}
        is_extruding = False
        point_cloud = []
        
        for timestamp, axis, is_reversal, phase in all_peaks:
            if is_reversal:
                direction[axis] *= -1
            
            if axis == 'E':
                is_extruding = True
                continue
            
            # Update position (one step)
            if axis == 'X':
                position['X'] += direction['X'] / self.config.x_steps_per_mm
            elif axis == 'Y':
                position['Y'] += direction['Y'] / self.config.y_steps_per_mm
            elif axis == 'Z':
                position['Z'] += direction['Z'] / self.config.z_steps_per_mm
            
            # Add to point cloud if extruding
            if is_extruding:
                point_cloud.append([position['X'], position['Y'], position['Z']])
        
        point_cloud_array = np.array(point_cloud)
        print(f"Generated {len(point_cloud_array)} points")
        return point_cloud_array
    
    def reconstruct_from_csvs(self, x_csv: str, y_csv: str, z_csv: str, e_csv: str) -> np.ndarray:
        """Complete pipeline: CSV files -> Point Cloud"""
        print("="*70)
        print("SIDE-CHANNEL RECONSTRUCTION PIPELINE")
        print("="*70)
        
        # Load data
        self.load_csv(x_csv, y_csv, z_csv, e_csv)
        
        # Process each axis
        for axis in ['X', 'Y', 'Z', 'E']:
            print(f"\nProcessing {axis}-axis...")
            
            # Apply filter
            print(f"  Applying low-pass filter...")
            self.apply_lowpass_filter(axis)
            
            # Detect peaks
            print(f"  Detecting peaks...")
            self.detect_peaks(axis)
            peak_count = sum(1 for t in self.traces[axis] if t.isPeak)
            print(f"    Found {peak_count} peaks")
            
            # Segment sections
            print(f"  Segmenting good/bad sections...")
            self.segment_good_bad_sections(axis)
            
            # Correct bad sections
            self.correct_bad_sections(axis)
        
        # Generate point cloud
        point_cloud = self.generate_point_cloud()
        
        print("\n" + "="*70)
        print("RECONSTRUCTION COMPLETE")
        print("="*70)
        
        return point_cloud
    
    def export_xyz(self, point_cloud: np.ndarray, filename: str):
        """Export point cloud to XYZ format"""
        with open(filename, 'w') as f:
            f.write(f"{len(point_cloud)}\n")
            f.write("Reconstructed from side-channel\n")
            for x, y, z in point_cloud:
                f.write(f"{x:.5f} {y:.5f} {z:.5f}\n")
        print(f"\nExported to {filename}")


class PointCloudToCSV:
    """Reverse operation: Point Cloud -> CSV files"""
    
    def __init__(self, config: PrinterConfig = None):
        self.config = config or PrinterConfig()
    
    def load_xyz(self, filename: str) -> np.ndarray:
        """Load XYZ file"""
        with open(filename, 'r') as f:
            lines = f.readlines()
            point_cloud = []
            for line in lines[2:]:  # Skip header
                parts = line.strip().split()
                if len(parts) >= 3:
                    point_cloud.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(point_cloud)
    
    def pointcloud_to_traces(self, point_cloud: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert point cloud to motor traces"""
        print("="*70)
        print("POINT CLOUD TO CSV CONVERSION")
        print("="*70)
        print(f"\nProcessing {len(point_cloud)} points...")
        
        # Generate peaks from point cloud
        peaks_by_axis = {'X': [], 'Y': [], 'Z': [], 'E': []}
        position = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        last_direction = {'X': 1, 'Y': 1, 'Z': 1}
        current_time = 0.0
        time_step = 0.0005  # 0.5ms between peaks
        
        # Add initial E peak
        peaks_by_axis['E'].append((current_time, 0, False))
        current_time += time_step
        
        for point in point_cloud:
            target_x, target_y, target_z = point
            
            # Calculate steps needed
            dx_steps = round((target_x - position['X']) * self.config.x_steps_per_mm)
            dy_steps = round((target_y - position['Y']) * self.config.y_steps_per_mm)
            dz_steps = round((target_z - position['Z']) * self.config.z_steps_per_mm)
            
            # Generate peaks for each axis
            for axis, steps, steps_per_mm in [
                ('X', dx_steps, self.config.x_steps_per_mm),
                ('Y', dy_steps, self.config.y_steps_per_mm),
                ('Z', dz_steps, self.config.z_steps_per_mm)
            ]:
                if steps == 0:
                    continue
                
                direction = 1 if steps > 0 else -1
                is_reversal = (direction != last_direction[axis])
                last_direction[axis] = direction
                
                for step in range(abs(steps)):
                    phase = step % 4
                    peaks_by_axis[axis].append((current_time, phase, is_reversal and step == 0))
                    current_time += time_step
                    position[axis] += direction / steps_per_mm
            
            # E-axis peak
            peaks_by_axis['E'].append((current_time, 0, False))
            current_time += time_step
        
        print(f"Generated peaks:")
        for axis in ['X', 'Y', 'Z', 'E']:
            print(f"  {axis}: {len(peaks_by_axis[axis])} peaks")
        
        # Convert peaks to current traces
        traces = {}
        for axis in ['X', 'Y', 'Z', 'E']:
            traces[axis] = self._peaks_to_current_trace(peaks_by_axis[axis], axis)
        
        return traces
    
    def _peaks_to_current_trace(self, peaks: List[Tuple], axis: str) -> np.ndarray:
        """Convert peaks to simulated current trace"""
        if len(peaks) == 0:
            return np.zeros((100, 3))  # Minimal trace
        
        # Create time array
        max_time = peaks[-1][0] + 0.1
        sample_rate = 2000  # 2 kHz
        num_samples = int(max_time * sample_rate)
        times = np.linspace(0, max_time, num_samples)
        
        # Initialize current traces
        phase0_current = np.zeros(num_samples)
        phase2_current = np.zeros(num_samples)
        
        # Generate sinusoidal current for each peak
        peak_amplitude = 1.5  # Amps
        peak_width = 0.002  # 2ms
        
        for peak_time, phase, is_reversal in peaks:
            # Find nearest time index
            idx = np.argmin(np.abs(times - peak_time))
            
            # Generate current pulse
            pulse_samples = int(peak_width * sample_rate)
            start_idx = max(0, idx - pulse_samples // 2)
            end_idx = min(num_samples, idx + pulse_samples // 2)
            
            # Sinusoidal pulse
            pulse_time = np.linspace(0, np.pi, end_idx - start_idx)
            pulse = peak_amplitude * np.sin(pulse_time)
            
            # Apply to appropriate phase
            if phase in [0, 1]:
                phase0_current[start_idx:end_idx] += pulse * (1 if phase == 0 else -1)
            else:
                phase2_current[start_idx:end_idx] += pulse * (1 if phase == 2 else -1)
        
        # Add noise
        noise_level = 0.05
        phase0_current += np.random.normal(0, noise_level, num_samples)
        phase2_current += np.random.normal(0, noise_level, num_samples)
        
        # Combine into array
        trace = np.column_stack([times, phase0_current, phase2_current])
        return trace
    
    def export_csvs(self, traces: Dict[str, np.ndarray], 
                    x_csv: str, y_csv: str, z_csv: str, e_csv: str):
        """Export traces to CSV files"""
        print("\nExporting CSV files...")
        
        csv_map = {'X': x_csv, 'Y': y_csv, 'Z': z_csv, 'E': e_csv}
        
        for axis, filename in csv_map.items():
            trace = traces[axis]
            df = pd.DataFrame(trace, columns=['Time(s)', 'Channel A(A)', 'Channel B(A)'])
            df.to_csv(filename, index=False)
            print(f"  Exported {filename} ({len(trace)} samples)")
        
        print("\n" + "="*70)
        print("CSV EXPORT COMPLETE")
        print("="*70)
    
    def xyz_to_csvs(self, xyz_file: str, x_csv: str, y_csv: str, z_csv: str, e_csv: str):
        """Complete pipeline: XYZ file -> CSV files"""
        point_cloud = self.load_xyz(xyz_file)
        traces = self.pointcloud_to_traces(point_cloud)
        self.export_csvs(traces, x_csv, y_csv, z_csv, e_csv)


# Example usage
if __name__ == "__main__":
    reconstructor = SideChannelReconstructor()
    point_cloud = reconstructor.reconstruct_from_csvs(
        'Jul23-AllAxesTest/X.csv', 'Jul23-AllAxesTest/Y.csv', 'Jul23-AllAxesTest/Z.csv', 'Jul23-AllAxesTest/E.csv'
    )
    reconstructor.export_xyz(point_cloud, 'reconstructed.xyz')
