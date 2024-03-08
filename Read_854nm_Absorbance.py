import os

# Function to extract value after 854.439 from a line
def extract_value(line):
    # Split the line by spaces
    parts = line.split()
    # Check if the line contains the pattern "854.439"
    if "854.439" in parts:
        # Find the index of "854.439"
        index = parts.index("854.439")
        # Check if there is a value after "854.439"
        if index + 1 < len(parts):
            # Get the value after "854.439"
            value = parts[index + 1]
            return value
    return None  # "854.439" not found or no value after it

# Directory containing text files
directory = "C:/Users/mso/Documents"

# List to store extracted values along with their respective filenames
results = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        # Open the file
        with open(filepath, 'r') as file:
            # Read each line
            for line in file:
                # Extract the value after 854.439
                value = extract_value(line)
                if value is not None:
                    # Append (filename, value) pair to results list
                    results.append((filename[:-4], value))  # Remove the ".txt" extension from the filename

# Write the extracted values to a new results.txt file
with open("results.txt", 'w') as output_file:
    for filename, value in results:
        output_file.write(f"{filename}: {value}\n")
