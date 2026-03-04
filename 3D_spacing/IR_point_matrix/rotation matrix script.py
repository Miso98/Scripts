import numpy as np

def transform_points(points, angle_deg, translation_vector):
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle_deg)
    
    # Define rotation matrix around the y-axis
    rotation_matrix_y = np.array([
        [np.cos(angle_rad), 0, -np.sin(angle_rad)],
        [0, 1, 0],
        [np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    
    # Transform each point
    transformed_points = []
    for point in points:
        # Convert the local point to a 3D vector in the local coordinate system
        local_point = np.array([point[0], point[1], 0])  # Assuming z=0 in the local plane

        # Apply rotation
        rotated_point = rotation_matrix_y @ local_point

        # Apply translation
        global_point = rotated_point + translation_vector

        # Append to the list of transformed points
        transformed_points.append(global_point)

    return transformed_points

# Example usage
# Points in the local coordinate system of the face (e.g., coordinates of the holes on the face)
local_points = [
    [-2.5, 32.5],  # Point 1
    #[42, -10],  # Point 2
    # Add more points as needed
]

# Rotation angle in degrees (angle of the face's normal relative to the x-axis)
angle_deg = 70  

# Translation vector (assuming the face's origin in local coordinates aligns with this position in global space)
translation_vector = np.array([0, 0, -47])

# Calculate transformed points in global coordinates
transformed_points = transform_points(local_points, angle_deg, translation_vector)

# Print results
for i, point in enumerate(transformed_points):
    print(f"Global coordinates of point {i+1}: {point}")
