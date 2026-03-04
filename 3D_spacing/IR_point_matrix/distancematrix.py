import numpy as np
# Define 9x3 matrix with x, y, z coordinates for 9 marker points in custom Lyra tool
points = np.array([
    [-5, 0, 35],
    [-15, 40, -30],
    [-10, -30, -15],
    [15, 55, 40],
    [35, 55, 5],
    [35, 55, -40.5],
    [10, -40, 30],
    [30, -55, 5],
    [10, -50, -30]
])
#matrix currently used in prototype that adheres to NDI standard 
prototype_matrix = np.array([
    [-5, 0, 35],
    [-15, 40, -30],
    [-30, -20, -5],
    [10, 55, 42.5],
    [42.5, 75, 10],
    [35, 55, -40],
    [10, -80, 40],
    [42.5, -55, 10],
    [15, -60, -40]
]) 

# Initialize a 9x9 distance matrix filled with zeros
distance_matrix = np.zeros((9, 9))

# Calculate pairwise distances
for i in range(len(points)):
    for j in range(i+1, len(points)):
        # Calculate Euclidean distance
        distance = np.linalg.norm(points[i] - points[j])
        # Populate the symmetric entries in the distance matrix
        #this creates a hollow matrix with zeroes across the diagonal and reflected values 
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

# Print the distance matrix
print("Initial Distance Matrix:")
print(distance_matrix)

# Threshold distance
min_distance = 50 #minimum marker distance of 50mm 
max_iterations = 1000
# Function to adjust coordinates to ensure minimum distance
def adjust_coordinates(points, dist_matrix, min_dist, max_iter):
    n_points = points.shape[0]
    adjusted = True
    iteration = 0
    # Face constraints
    face_1 = [0, 1, 2]  # x < 0.5; -55 < y < 55
    face_2 = [3, 4, 5]  # 55 < y < 80; 0 < x < 55
    face_3 = [6, 7, 8]  # -80 < y < -55; 0 < x < 55
    z_min, z_max = -45, 45  # z constraint for all faces

    edge_min = 5 # minimum distance to point to edge of face 

    while adjusted and iteration < max_iter:
        adjusted = False
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if dist_matrix[i, j] < min_dist:
                    # Find the direction vector and normalize it
                    direction = points[i] - points[j]
                    norm = np.linalg.norm(direction)
                    
                    if norm != 0:  # Avoid division by zero
                        direction /= norm
                    
                        # Calculate the required adjustment distance
                        adjustment_distance = (min_dist - dist_matrix[i, j]) / 2
                        
                        # Adjust both points in opposite directions
                        new_points_i = points[i] + direction * adjustment_distance
                        new_points_j = points[j] - direction * adjustment_distance
                        

                        # Apply face constraints
                        if i in face_1:
                            new_points_i[0] = min(new_points_i[0], 0.5) # x < .5
                            new_points_i[1] = max(-55, min(new_points_i[1], 42)) #  -55 < y < 55
                        elif i in face_2:
                            new_points_i[1] = max(55, min(new_points_i[1], 80)) # 55 < y < 80
                            new_points_i[0] = max(0, min(new_points_i[0], 42)) # 0 < x < 55
                        elif i in face_3:
                            new_points_i[1] = max(-80, min(new_points_i[1], -42)) # -80 < y < -55
                            new_points_i[0] = max(0, min(new_points_i[0], 42)) # 0 < x < 55
                        
                        #Z constraint for all points
                        new_points_i[2] = max(z_min, min(new_points_i[2], z_max))


                        if j in face_1:
                            new_points_j[0] = min(new_points_j[0], 0.5)  # x < 0.5
                            new_points_j[1] = max(-55, min(new_points_j[1], 42))  # y between -55 and 55
                        elif j in face_2:
                            new_points_j[1] = max(55, min(new_points_j[1], 80))  # y between 55 and 80
                            new_points_j[0] = max(0, min(new_points_j[0], 42))  # x between 0 and 55
                        elif j in face_3:
                            new_points_j[1] = max(-80, min(new_points_j[1], -42))  # y between -80 and -55
                            new_points_j[0] = max(0, min(new_points_j[0], 42))  # x between 0 and 55
                        
                        # Apply z constraint for all points
                        new_points_j[2] = max(z_min, min(new_points_j[2], z_max))


                        # Update the distance matrix for this pair
                        dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
                        dist_matrix[j, i] = dist_matrix[i, j]
                        points[i] = new_points_i
                        points[j] = new_points_j
                        adjusted = True
    
        iteration += 1
    return points, iteration

# Adjust the coordinates
new_coordinates, iterations_used = adjust_coordinates(points, distance_matrix, min_distance, max_iterations)

# Print the adjusted coordinates and the updated distance matrix
print("Adjusted Coordinates:\n", new_coordinates)
print("\nUpdated Distance Matrix:\n", distance_matrix)
print(f"\nTotal Iterations: {iterations_used}")
