import matplotlib.pyplot as plt
import pickle

def display_slices(image_data, title="Image Slices"):
    """
    Displays three orthogonal slices (axial, sagittal, coronal) of a 3D image.

    Args:
        image_data (numpy.ndarray): The 3D image data.
        title (str): Title of the plot.
    """
    try:
        axial_slice = image_data[image_data.shape[0] // 2, :, :]
        sagittal_slice = image_data[:, image_data.shape[1] // 2, :]
        coronal_slice = image_data[:, :, image_data.shape[2] // 2]

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(axial_slice, cmap="gray")
        plt.title("Axial Slice")

        plt.subplot(1, 3, 2)
        plt.imshow(sagittal_slice, cmap="gray")
        plt.title("Sagittal Slice")

        plt.subplot(1, 3, 3)
        plt.imshow(coronal_slice, cmap="gray")
        plt.title("Coronal Slice")

        plt.suptitle(title)
        plt.show()
    except Exception as e:
        print(f"An error occurred while displaying slices: {e}")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def display_and_save_slices(image_data, save_dir, base_filename="slices"):
    """
    Displays and saves three orthogonal slices (axial, sagittal, coronal) of a 3D image as a single figure.

    Args:
        image_data (numpy.ndarray): The 3D image data.
        save_dir (str): Directory where the combined figure will be saved.
        base_filename (str): Base name for the saved slice files.
    """
    try:
        # Ustvari mapo, če še ne obstaja
        os.makedirs(save_dir, exist_ok=True)

        # Izračunaj reze
        axial_slice = image_data[image_data.shape[0] // 2, :, :]
        sagittal_slice = image_data[:, image_data.shape[1] // 2, :]
        coronal_slice = image_data[:, :, image_data.shape[2] // 2]

        # Prikaz slik na subplotih
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(axial_slice, cmap="gray")
        axes[0].set_title("Axial Slice")
        axes[0].axis("off")

        axes[1].imshow(sagittal_slice, cmap="gray")
        axes[1].set_title("Sagittal Slice")
        axes[1].axis("off")

        axes[2].imshow(coronal_slice, cmap="gray")
        axes[2].set_title("Coronal Slice")
        axes[2].axis("off")

        # Prilagoditev in shranjevanje figure z unikatnim imenom
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"{base_filename}_{timestamp}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Figure saved at: {save_path}")
    except Exception as e:
        print(f"An error occurred while displaying and saving slices: {e}")

def load_pkl_image(path):
    """
    Loads 3D image data from a .pkl file.

    Args:
        path (str): Path to the .pkl file.

    Returns:
        numpy.ndarray: The 3D image data.
    """
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, tuple):
            return data[0]
        return data
    except Exception as e:
        print(f"An error occurred while loading the .pkl file: {e}")
        return None
    

original_pkl_image = "Release_pkl/Resized_imagesTr/ThoraxCBCT_0000_0000.pkl"
resized_pkl_image = "Release_pkl/Resized_merged_imagesTr/Pre_therapy/Train/merged_patient_0000_pre.pkl"

# Load and display the original .pkl image
with open(original_pkl_image, 'rb') as pkl_file:
    original_image = pickle.load(pkl_file)
    display_slices(original_image, title="Original Image")

# # Load and display the resized .pkl image
# with open(resized_pkl_image, 'rb') as pkl_file:
#     resized_image = pickle.load(pkl_file)
#     display_slices(resized_image, title="Resized Image")

save_dir = "Release_pkl/Resized_merged_imagesTr/data_viewed"
# processed_data = "processed_ThoraxCBCT/patient_0000_CBCT_post.pkl"
# # Naloži sliko
image_data = load_pkl_image(resized_pkl_image)

# # # Če je slika uspešno naložena, jo prikaži in shrani
if image_data is not None:
    display_and_save_slices(image_data, save_dir, base_filename="ThoraxCBCT_slices.png")


