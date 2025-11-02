import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Peak:
    """Represents a peak in the motor trace"""
    timestamp: float
    axis: str  # 'X', 'Y', 'Z', or 'E'
    phase: int  # 0-3
    is_reversal: bool = False
    
@dataclass
class PrinterConfig:
    """Printer-specific configuration"""
    x_steps_per_mm: float = 6.0
    y_steps_per_mm: float = 6.0
    z_steps_per_mm: float = 95.2
    e_steps_per_mm: float = 1.0  # Not used for position, only extrusion tracking

class PointCloudGenerator:
    """Converts annotated peaks to point cloud"""
    
    def __init__(self, config: PrinterConfig = None):
        self.config = config or PrinterConfig()
        self.position = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        self.e_position = 0.0
        self.is_extruding = False
        self.point_cloud = []
        
        # Direction tracking for each axis
        self.direction = {'X': 1, 'Y': 1, 'Z': 1, 'E': 1}
        
    def process_peaks(self, peaks: List[Peak]) -> np.ndarray:
        """
        Convert peaks to point cloud
        
        Args:
            peaks: List of Peak objects sorted by timestamp
            
        Returns:
            numpy array of shape (N, 3) containing X, Y, Z coordinates
        """
        # Sort peaks by timestamp to ensure proper ordering
        sorted_peaks = sorted(peaks, key=lambda p: p.timestamp)
        
        for peak in sorted_peaks:
            # Handle reversals
            if peak.is_reversal:
                self.direction[peak.axis] *= -1
            
            # Update position based on axis and direction
            if peak.axis == 'X':
                self.position['X'] += self.direction['X'] / self.config.x_steps_per_mm
            elif peak.axis == 'Y':
                self.position['Y'] += self.direction['Y'] / self.config.y_steps_per_mm
            elif peak.axis == 'Z':
                self.position['Z'] += self.direction['Z'] / self.config.z_steps_per_mm
            elif peak.axis == 'E':
                self.e_position += self.direction['E']
                # E-axis activity indicates extrusion
                # In the paper, dwell annotations mark extrusion periods
                # Here we assume any E movement means extrusion
                self.is_extruding = True
                continue  # Don't add E-axis peaks to point cloud
            
            # Only add point if extruding (material being deposited)
            if self.is_extruding:
                self.point_cloud.append([
                    self.position['X'],
                    self.position['Y'],
                    self.position['Z']
                ])
        
        return np.array(self.point_cloud)
    
    def mark_extrusion_state(self, is_extruding: bool):
        """Manually set extrusion state (for dwell commands)"""
        self.is_extruding = is_extruding
    
    def export_xyz(self, filename: str):
        """Export point cloud to XYZ file format"""
        with open(filename, 'w') as f:
            f.write(f"{len(self.point_cloud)}\n")
            f.write("Point cloud from side-channel reconstruction\n")
            for x, y, z in self.point_cloud:
                f.write(f"{x:.5f} {y:.5f} {z:.5f}\n")


class PeakExtractor:
    """Converts point cloud back to peaks/motor movements"""
    
    def __init__(self, config: PrinterConfig = None):
        self.config = config or PrinterConfig()
        
    def pointcloud_to_peaks(self, point_cloud: np.ndarray) -> List[Peak]:
        """
        Convert point cloud back to peaks
        
        Args:
            point_cloud: numpy array of shape (N, 3) with X, Y, Z coordinates
            
        Returns:
            List of Peak objects
        """
        if len(point_cloud) == 0:
            return []
        
        peaks = []
        current_time = 0.0
        time_increment = 0.001  # Assume 1ms between peaks (arbitrary)
        
        # Track position and direction for each axis
        position = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        last_direction = {'X': 1, 'Y': 1, 'Z': 1}
        
        # Add E-axis peak to start extrusion
        peaks.append(Peak(
            timestamp=current_time,
            axis='E',
            phase=0,
            is_reversal=False
        ))
        current_time += time_increment
        
        for point in point_cloud:
            # Calculate movement in steps for each axis
            target_x, target_y, target_z = point
            
            dx_steps = round((target_x - position['X']) * self.config.x_steps_per_mm)
            dy_steps = round((target_y - position['Y']) * self.config.y_steps_per_mm)
            dz_steps = round((target_z - position['Z']) * self.config.z_steps_per_mm)
            
            # Generate peaks for each axis that needs to move
            movements = [
                ('X', dx_steps, self.config.x_steps_per_mm),
                ('Y', dy_steps, self.config.y_steps_per_mm),
                ('Z', dz_steps, self.config.z_steps_per_mm)
            ]
            
            for axis, steps, steps_per_mm in movements:
                if steps == 0:
                    continue
                    
                # Determine direction
                direction = 1 if steps > 0 else -1
                
                # Check if direction changed (reversal)
                is_reversal = (direction != last_direction[axis])
                last_direction[axis] = direction
                
                # Generate peaks for each step
                for step in range(abs(steps)):
                    # Cycle through phases 0-3
                    phase = step % 4
                    
                    peaks.append(Peak(
                        timestamp=current_time,
                        axis=axis,
                        phase=phase,
                        is_reversal=(is_reversal and step == 0)
                    ))
                    
                    current_time += time_increment
                    
                    # Update position
                    position[axis] += direction / steps_per_mm
            
            # Add E-axis peak for extrusion
            peaks.append(Peak(
                timestamp=current_time,
                axis='E',
                phase=0,
                is_reversal=False
            ))
            current_time += time_increment
        
        return peaks
    
    def load_xyz(self, filename: str) -> np.ndarray:
        """Load point cloud from XYZ file format"""
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            # Skip first two header lines
            point_cloud = []
            for line in lines[2:]:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        point_cloud.append([x, y, z])
        
        return np.array(point_cloud)


# Example usage
if __name__ == "__main__":
    # Convert xyz file into peaks
    extractor = PeakExtractor()
    loaded_cloud = extractor.load_xyz("cube_10mm.xyz")
    reconstructed_peaks = extractor.pointcloud_to_peaks(loaded_cloud)
    print(reconstructed_peaks)

    # Convert peaks back into xyz file
    generator = PointCloudGenerator()
    point_cloud = generator.process_peaks(reconstructed_peaks)
    generator.export_xyz("reconstructed.xyz")
