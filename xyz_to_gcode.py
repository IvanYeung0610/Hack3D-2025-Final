import sys

def xyz_to_gcode(xyz_path, gcode_path):
    xyz_points = []
    
    with open(xyz_path, 'r') as infile:
        lines = infile.readlines()
        
        # Skip header lines (first two lines: count and comment)
        for line in lines[2:]:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    xyz_points.append((x, y, z))
    
    # Write to G-code file
    with open(gcode_path, 'w') as outfile:
        # Write G-code header
        outfile.write("; Converted from XYZ format\n")
        outfile.write("G21 ; Set units to millimeters\n")
        outfile.write("G90 ; Use absolute positioning\n")
        outfile.write("M82 ; Use absolute extrusion mode\n")
        outfile.write("G92 E0 ; Reset extruder position\n")
        outfile.write("\n")
        
        # Track extrusion amount
        e_pos = 0.0
        prev_point = None
        
        # Write motion commands
        for i, (x, y, z) in enumerate(xyz_points):
            if i == 0:
                # First point: rapid move without extrusion
                outfile.write(f"G0 X{x:.5f} Y{y:.5f} Z{z:.5f}\n")
            else:
                # Calculate distance and add extrusion
                if prev_point:
                    dx = x - prev_point[0]
                    dy = y - prev_point[1]
                    dz = z - prev_point[2]
                    distance = (dx**2 + dy**2 + dz**2)**0.5
                    
                    # Add extrusion proportional to distance (typical value ~0.04mm per mm traveled)
                    e_pos += distance * 0.04
                
                outfile.write(f"G1 X{x:.5f} Y{y:.5f} Z{z:.5f} E{e_pos:.5f}\n")
            
            prev_point = (x, y, z)
        
        # Footer
        outfile.write("\n")
        outfile.write("G92 E0 ; Reset extruder\n")
        outfile.write("M84 ; Disable motors\n")
    
    print(f"Converted {len(xyz_points)} points from {xyz_path} → {gcode_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python xyz_to_gcode.py input.xyz output.gcode")
    else:
        xyz_to_gcode(sys.argv[1], sys.argv[2])
