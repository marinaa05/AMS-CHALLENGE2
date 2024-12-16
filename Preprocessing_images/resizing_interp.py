# Datoteka za resizing
import nibabel as nib
import pickle
import os
import numpy as np
from skimage.transform import downscale_local_mean  # pip install scikit-image
from skimage.transform import resize
import matplotlib.pyplot as plt

def resize_pkl_images_in_directory(input_dir, output_dir, new_shape):
    """
    Resizes all 3D medical images stored in .pkl format in a directory and saves them to an output directory.

    Args:
        input_dir (str): Path to the directory containing .pkl files.
        output_dir (str): Path to the directory to save resized .pkl files.
        new_shape (tuple): Desired shape for the resized images (e.g., (128, 128, 128)).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all .pkl files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".pkl"):
            input_pkl_path = os.path.join(input_dir, file_name)
            output_pkl_path = os.path.join(output_dir, file_name)

            try:
                # Load the .pkl file
                with open(input_pkl_path, 'rb') as pkl_file:
                    image_data = pickle.load(pkl_file)

                # Resize the image using interpolation
                resized_image = resize(image_data, new_shape, mode='reflect', anti_aliasing=True, preserve_range=True)

                # Save the resized image to a new .pkl file
                with open(output_pkl_path, 'wb') as pkl_file:
                    pickle.dump(resized_image, pkl_file)

                print(f"Successfully resized image and saved to {output_pkl_path}")
                print(f"New size of the image: {resized_image.shape}")
            except FileNotFoundError as e:
                print(f"Error: {e}. Please check if the input .pkl file path is correct.")
            except Exception as e:
                print(f"An unexpected error occurred for file {file_name}: {e}")

# Example usage
input_dir = "Release_pkl/imagesTr"  # Replace with the path to your input directory containing .pkl files
output_dir = "Release_pkl/Resized_imagesTr"  # Replace with the path to your output directory for resized .pkl files
new_shape = (256 // 1.5, 192 // 1.5, 192 // 1.5)  # Example new shape

resize_pkl_images_in_directory(input_dir, output_dir, new_shape)

