import numpy as np
import pandas as pd

# Steps per mm from paper
STEPS_PER_MM = {'x': 6.0, 'y': 6.0, 'z': 95.2}

def point_cloud_generation(x_peaks, y_peaks, z_peaks, e_peaks, output_path="reconstruction.xyz"):
    """
    Generate a 3D point cloud from corrected motor peaks.
    Each peak array should contain dictionaries with:
      - 'timestamp': float
      - 'direction': +1 or -1
      - 'reversal': bool
      - 'dwell': bool (E-axis only)
    """
    # Add a axis column indicating which type of peak
    axis_col = np.full((x_peaks.shape[0], 1), 'x')
    x_peaks = np.hstack((axis_col, x_peaks))
    axis_col = np.full((y_peaks.shape[0], 1), 'y')
    x_peaks = np.hstack((axis_col, y_peaks))
    axis_col = np.full((z_peaks.shape[0], 1), 'z')
    x_peaks = np.hstack((axis_col, z_peaks))
    axis_col = np.full((e_peaks.shape[0], 1), 'e')
    x_peaks = np.hstack((axis_col, e_peaks))

    # Combine and sort by timestamp
    events = np.concatenate((x_peaks, y_peaks, z_peaks, e_peaks))
    # sorting step to still figure out...

    # Initialize state
    pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    extruding = False
    points = []

    # Process peaks
    for event in events:
        axis = event[0]

        if axis == 'e':
            # Toggle extrusion based on dwell markers
            if event.get('dwell_start', False):
                extruding = True
            elif event.get('dwell_end', False):
                extruding = False
            continue

        # Update position
        step_size = 1.0 / STEPS_PER_MM[axis]
        pos[axis] += event['direction'] * step_size

        # If extruding, add current position to point cloud
        if extruding:
            points.append((pos['x'], pos['y'], pos['z']))

    # Save as XYZ
    np.savetxt(output_path, np.array(points), fmt="%.6f %.6f %.6f")
    print(f"Saved {len(points)} points to {output_path}")
    return np.array(points)

