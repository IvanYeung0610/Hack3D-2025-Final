import numpy as np
import pandas as pd

STEPS_PER_MM = {'x': 6.0, 'y': 6.0, 'z': 95.2}

# ================================================================
# 1️⃣  PEAKS → POINT CLOUD
# ================================================================
def peaks_to_pointcloud(x_peaks, y_peaks, z_peaks, e_peaks, output_path="reconstruction.xyz"):
    events = []
    for axis, data in zip(['x','y','z','e'], [x_peaks, y_peaks, z_peaks, e_peaks]):
        for d in data:
            events.append({'axis': axis, **d})
    events.sort(key=lambda e: e['timestamp'])

    pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    extruding = False
    points = []

    for event in events:
        axis = event['axis']
        if axis == 'e':
            if event.get('dwell_start', False):
                extruding = True
            elif event.get('dwell_end', False):
                extruding = False
            continue

        step_size = 1.0 / STEPS_PER_MM[axis]
        pos[axis] += event['direction'] * step_size
        if extruding:
            points.append((pos['x'], pos['y'], pos['z']))

    np.savetxt(output_path, np.array(points), fmt="%.6f %.6f %.6f")
    print(f"✅ Saved {len(points)} points to {output_path}")
    return np.array(points)


# ================================================================
# 2️⃣  POINT CLOUD → PEAKS
# ================================================================
def pointcloud_to_peaks(xyz_path, start_timestamp=0.0, time_step=0.01):
    """
    Reconstruct approximate peaks from an XYZ point cloud.
    Returns per-axis peak event lists.
    """
    pts = np.loadtxt(xyz_path)
    diffs = np.diff(pts, axis=0)

    # Initialize results
    peaks = {'x': [], 'y': [], 'z': [], 'e': []}
    t = start_timestamp

    for d in diffs:
        # Convert movement (in mm) to number of steps
        step_counts = {
            'x': int(round(abs(d[0]) * STEPS_PER_MM['x'])),
            'y': int(round(abs(d[1]) * STEPS_PER_MM['y'])),
            'z': int(round(abs(d[2]) * STEPS_PER_MM['z']))
        }

        # For each axis, create a series of step events
        for axis in ['x', 'y', 'z']:
            direction = np.sign(d[['x','y','z'].index(axis)]) or 0
            for _ in range(step_counts[axis]):
                t += time_step
                peaks[axis].append({
                    'timestamp': t,
                    'direction': int(direction),
                    'reversal': False
                })

        # If there was movement, assume extrusion active
        if np.linalg.norm(d) > 0:
            peaks['e'].append({'timestamp': t, 'dwell_start': True})
        else:
            peaks['e'].append({'timestamp': t, 'dwell_end': True})

    return peaks


# ================================================================
# 3️⃣  TEST HARNESS
# ================================================================
def synthetic_test():
    """
    Generates fake peaks for a cube path, converts to point cloud,
    then reconstructs peaks back from that point cloud.
    """
    # Simple square motion in XY plane
    step_time = 0.01
    n_steps = 10

    def make_axis(axis, direction, start_t):
        return [{'timestamp': start_t + i*step_time, 'direction': direction, 'reversal': False}
                for i in range(n_steps)]

    x_peaks = make_axis('x', +1, 0)
    y_peaks = make_axis('y', +1, n_steps*step_time)
    x_peaks += make_axis('x', -1, 2*n_steps*step_time)
    y_peaks += make_axis('y', -1, 3*n_steps*step_time)

    # Z moves up once
    z_peaks = [{'timestamp': 4*n_steps*step_time, 'direction': +1, 'reversal': False}]
    # Extruder active throughout
    e_peaks = [{'timestamp': 0, 'dwell_start': True}, {'timestamp': 4*n_steps*step_time, 'dwell_end': True}]

    # Run forward
    pts = peaks_to_pointcloud(x_peaks, y_peaks, z_peaks, e_peaks, "synthetic.xyz")

    # Reverse (approximate)
    peaks_back = pointcloud_to_peaks("synthetic.xyz")

    print(f"\nReconstructed peaks summary:")
    for k in ['x','y','z']:
        print(f"Axis {k.upper()}: {len(peaks_back[k])} peaks")

    return pts, peaks_back


# Run test harness
if __name__ == "__main__":
    pts, peaks_back = synthetic_test()
