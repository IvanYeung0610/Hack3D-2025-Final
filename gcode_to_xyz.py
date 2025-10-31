import re
import sys

def gcode_to_xyz(gcode_path, xyz_path):
    # Regular expression to extract X, Y, Z coordinates
    coord_pattern = re.compile(r'([XYZ])([-+]?\d*\.?\d+)')
    
    # Initialize position memory
    current_pos = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
    
    xyz_points = []
    
    with open(gcode_path, 'r') as gfile:
        for line in gfile:
            line = line.strip()
            
            # Only process motion commands (G0, G1)
            if line.startswith(('G0', 'G1')):
                matches = coord_pattern.findall(line)
                for axis, value in matches:
                    current_pos[axis] = float(value)
                
                xyz_points.append((current_pos['X'], current_pos['Y'], current_pos['Z']))
    
    # Write to XYZ file
    with open(xyz_path, 'w') as outfile:
        outfile.write(f"{len(xyz_points)}\n")
        outfile.write("Converted from G-code\n")
        for x, y, z in xyz_points:
            outfile.write(f"{x:.5f} {y:.5f} {z:.5f}\n")
    
    print(f"✅ Converted {len(xyz_points)} points from {gcode_path} → {xyz_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gcode_to_xyz.py input.gcode output.xyz")
    else:
        gcode_to_xyz(sys.argv[1], sys.argv[2])
