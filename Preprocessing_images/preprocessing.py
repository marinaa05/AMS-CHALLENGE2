import os
import pickle
import nibabel as nib
import numpy as np
from skimage.transform import resize

class Preprocessing:
    def __init__(self, input_dir, output_dir, new_shape):
        """
        Initializes the Preprocessing class with input and output directories and the desired shape for resizing.

        Args:
            input_dir (str): Path to the directory containing .nii.gz files.
            output_dir (str): Path to the directory to save processed .pkl files.
            new_shape (tuple): Desired shape for resized images (e.g., (128, 128, 128)).
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.new_shape = new_shape

    def nii_to_pkl(self, input_path, output_path):
        """
        Converts a .nii.gz file to a .pkl file.

        Args:
            input_path (str): Path to the input .nii.gz file.
            output_path (str): Path to the output .pkl file.
        """
        nii_image = nib.load(input_path)
        image_data = nii_image.get_fdata()
        with open(output_path, 'wb') as pkl_file:
            pickle.dump(image_data, pkl_file)

    def resize_image(self, image_data):
        """
        Resizes a 3D image to the specified shape.

        Args:
            image_data (numpy array): The 3D image data to resize.

        Returns:
            numpy array: The resized image.
        """
        return resize(image_data, self.new_shape, mode='reflect', anti_aliasing=True, preserve_range=True)

    def normalize_image(self, image_data):
        """
        Normalizes a 3D image to the range [0, 1].

        Args:
            image_data (numpy array): The 3D image data to normalize.

        Returns:
            numpy array: The normalized image.
        """
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        if max_val > min_val:  # Avoid division by zero
            return (image_data - min_val) / (max_val - min_val)
        return image_data

    def process_images(self):
        """
        Processes all .nii.gz files in the input directory: converts to .pkl, resizes, and normalizes them.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        for file_name in os.listdir(self.input_dir):
            if file_name.endswith(".nii.gz"):
                input_path = os.path.join(self.input_dir, file_name)
                output_file_name = file_name.replace(".nii.gz", ".pkl")
                output_path = os.path.join(self.output_dir, output_file_name)

                try:
                    print(f"Processing file: {file_name}")
                    # Convert to .pkl
                    self.nii_to_pkl(input_path, output_path)

                    # Load the .pkl file
                    with open(output_path, 'rb') as pkl_file:
                        image_data = pickle.load(pkl_file)

                    # Resize the image
                    resized_image = self.resize_image(image_data)

                    # Normalize the image
                    normalized_image = self.normalize_image(resized_image)

                    # Save the final processed image
                    with open(output_path, 'wb') as pkl_file:
                        pickle.dump(normalized_image, pkl_file)

                    print(f"Successfully processed and saved: {output_path}")

                except Exception as e:
                    print(f"An error occurred while processing {file_name}: {e}")

if __name__ == "__main__":
    input_dir = "path_to_input_directory"
    output_dir = "path_to_output_directory"
    new_shape = (256 // 1.5, 192 // 1.5, 192 // 1.5)  # Example shape

    preprocessing = Preprocessing(input_dir, output_dir, new_shape)
    preprocessing.process_images()
