import os
import pickle
import numpy as np

def normalize_images(input_dir, output_dir):
    """
    Normalizes all images in a directory to the range [0, 1] and saves them to an output directory.

    Args:
        input_dir (str): Path to the directory containing input .pkl files.
        output_dir (str): Path to the directory to save normalized .pkl files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all .pkl files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".pkl"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            try:
                # Load the image
                with open(input_path, 'rb') as pkl_file:
                    image_data = pickle.load(pkl_file)

                # Normalize the image to the range [0, 1]
                min_val = np.min(image_data)
                max_val = np.max(image_data)
                if max_val > min_val:  # Avoid division by zero
                    normalized_image = (image_data - min_val) / (max_val - min_val)
                else:
                    normalized_image = image_data  # If min == max, image is constant

                # Save the normalized image to a new .pkl file
                with open(output_path, 'wb') as out_file:
                    pickle.dump(normalized_image, out_file)

                print(f"Successfully normalized and saved: {output_path}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

# Example usage
input_dir = "Release_pkl/Resized_imagesTr"  # Replace with the path to your input directory containing .pkl files
output_dir = "Release_pkl/Resized_normalized_imagesTr"  # Replace with the path to your output directory for normalized .pkl files
normalize_images(input_dir, output_dir)
