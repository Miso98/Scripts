import numpy as np

def translate_points(matrix, angles_deg, lengths):
    # Check that angles and lengths have the same number of elements as the matrix
    if len(angles_deg) != len(matrix) or len(lengths) != len(matrix):
        raise ValueError("The number of angles and lengths must match the number of points in the matrix.")
    
    # Initialize a list to store translated points
    translated_matrix = []
    
    for point, angle_deg, length in zip(matrix, angles_deg, lengths):
        # Convert angle to radians for each point
        angle_rad = np.radians(angle_deg)
        
        # Calculate x and z offsets for each point
        x_offset = -length * np.cos(angle_rad)
       
        z_offset = -length * np.sin(angle_rad) #for face 2
        
        # Apply transformation to the point
        x, y, z = point
        new_x = x + x_offset
        new_z = z + z_offset
        
        # Append the translated point to the matrix
        translated_matrix.append([new_x, y, new_z])
    
    return np.array(translated_matrix)

# Example usage with different angles and lengths for each point
# points_matrix = np.array([
#     [-29.17, 0, 46.25],
#     [-10.981, -29, 56.75],
#     [3, 25, 65]
# ]) # face 3 at 60 deg

points_matrix = np.array([ #face 2 at 70 deg 
    [-13.32, 32.5, -46.9],
    [-30.321, 12.5, -42.5],
    [8.369, -30, -52.6]
])

# angles = [-60, -60, -60]  # angles in degrees for each point
angles = [70, 70, 70]  # angles in degrees for each point in face 2 
lengths = [7, 32, 27]  # lengths in mm for each point in face 2 
# lengths = [32, 7, 17]  # lengths in mm for each point

translated_points = translate_points(points_matrix, angles, lengths)
print("Translated Points:\n", translated_points)
