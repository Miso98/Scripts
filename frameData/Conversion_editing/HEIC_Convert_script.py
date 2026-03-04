import os
from PIL import Image
import pyheif

input_folder = '/home/mitchell/Documents/repos/Tool_Recognition/src/HEIC_Images/'
output_folder = '/home/mitchell/Documents/repos/Tool_Recognition/src/Converted_Images/'

# make sure folder is real
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Print files in the input folder
print("Files in input folder:", os.listdir(input_folder))

def convert_heic_to_jpg(input_path, output_path):
    heif_file = pyheif.read(input_path)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    image.save(output_path, format="JPEG")

# convert and process files
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".heic"):  # Case-insensitive check
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jpg")
        print(f"Converting {input_path} to {output_path}")
        
        try:
            convert_heic_to_jpg(input_path, output_path)
            print(f"Converted: {filename} -> {os.path.basename(output_path)}")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
