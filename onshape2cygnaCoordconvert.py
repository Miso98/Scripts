import numpy as np

def transform_points(matrix):
    # Define the transformation function
    def transform_point(point):
        x, y, z = point
        # Apply the transformations
        new_x = -y
        new_y = z
        new_z = -x
        return [new_x, new_y, new_z]
    
    # Apply the transformation to each point in the matrix
    transformed_matrix = np.array([transform_point(point) for point in matrix])
    return transformed_matrix

# Example matrix of points (each row is a point with [x, y, z])
input_matrix = np.array([
    [27.5, 40, 15], #point 1 
    [-30, 40, 32.5], #point 2 
    [0, 40, -27.5], #point 3 
    [46.9, 13.382, 32.5], #point 4
    [42.5, 30.321, -12.5], #point 5
    [52.563, -8.396, -30 ], #point 6
    [-46.25, 29.175, 0], #point 7
    [-56.75, 10.988, -29], #point 8
    [-65, -3, 25] #point 9
])

# Transform the matrix
output_matrix = transform_points(input_matrix)
#(+7 mm to account for standoff and radius of sphere)
# Display the result
print("Original Matrix:")
print(input_matrix)
print("\nTransformed Matrix:")
print(output_matrix)

