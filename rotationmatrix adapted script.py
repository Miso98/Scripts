import numpy as np

def transform_points(points, angle_deg, rotation_axis, translation_vector):
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle_deg)
    
    # rotation matrix based on the  rotation axis
    if rotation_axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif rotation_axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif rotation_axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid rotation axis. Choose 'x', 'y', or 'z'.")

    # Transform each point
    transformed_points = []
    for point in points:
        # Convert the local point to a 3D vector in the local coordinate system
        local_point = np.array([point[0], point[1], 0])  # Assuming z=0 in the local plane

        # Apply rotation
        rotated_point = rotation_matrix @ local_point

        # Apply translation
        global_point = rotated_point + translation_vector

        # Append to the list of transformed points
        transformed_points.append(global_point)

    return transformed_points

#
# Points in the local coordinate system of the face 
local_points = [
    #[-21.5, 32.5], #point 4
    [-17.5, 0], #point 7
    #[-12.5, -12.5], # point 5 
    [7.5, -27.5], # point 8
    #[15,-30]   #  for point 6
    [20, 25]
    
]

# Rotation angle in degrees (angle of the face's normal relative to the y-axis)
angle_deg = -30  

# rotation axis ('x', 'y', or 'z')
rotation_axis = 'y'  # Set to 'y' for rotation around the y-axis

# Define the translation values
x_translation = -14  #  x translation value
y_translation = 0   #  y translation value
z_translation = 62  # z translation value 55+7

# Combine into a translation vector
translation_vector = np.array([x_translation, y_translation, z_translation])

# Calculate transformed points in global coordinates
transformed_points = transform_points(local_points, angle_deg, rotation_axis, translation_vector)

# Print 
for i, point in enumerate(transformed_points):
    print(f"Global coordinates of point {i+1}: {point}")
