import pickle
import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

# path = "./dataset/LPBA_data/Train/subject_13.pkl"
# with open(path, "rb") as file:
#    res= pickle.load(file)
#print(type(res))
# print(len(res))
#print(res[0])
# print(res[0].shape)
#print(res[1])
#print(res[1].shape)
#plt.imshow(res[0])
# path = "Thorax_pairs/Train_Resized/Thorax_pair_000.pkl"
# with open(path, "rb") as file:
#    res= pickle.load(file)
# #print(type(res))
# print(len(res))
# #print(res[0])
# print(res[0].shape)
# # print(res[1])
# print(res[1].shape)
# plt.imshow(res[0])


def nii_to_pkl(input_folder, output_folder):
    """
    Converts all .nii.gz files in the input folder to .pkl format
    and saves them in the output folder.

    Parameters:
    - input_folder (str): Path to the folder containing .nii.gz files.
    - output_folder (str): Path to the folder to save .pkl files.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all .nii.gz files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, filename)
            
            # Load the .nii.gz file
            nii_data = nib.load(input_path)
            data_array = nii_data.get_fdata()  # Extract the image data as a numpy array
            
            # Save as .pkl
            output_filename = filename.replace('.nii.gz', '.pkl')
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, 'wb') as pkl_file:
                pickle.dump(data_array, pkl_file)
            
            print(f"Converted {filename} to {output_filename}")

# Example usage
#input_folder = "Release_06_12_23/masksTr"  # Replace with the path to your .nii.gz folder
#output_folder = "/workspace/modetv2/Release_pkl/masksTr"  # Replace with the path to your desired .pkl output folder
#nii_to_pkl(input_folder, output_folder)

import os
import pickle
import matplotlib.pyplot as plt

def load_and_save_slice(pkl_file, output_folder):
    """
    Loads a .pkl file and saves a slice of the data as an image.

    Parameters:
    - pkl_file (str): Path to the .pkl file to load.
    - output_folder (str): Path to the folder where images will be saved.
    """
    # Load the .pkl file
    with open(pkl_file, 'rb') as file:
        data_array = pickle.load(file)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the middle slice (assuming 3D data)
    if len(data_array.shape) == 3:
        middle_slice = data_array.shape[2] // 2  # Middle slice along Z-axis
        
        # Generate a unique filename based on the input file name
        base_name = os.path.splitext(os.path.basename(pkl_file))[0]  # Extract base name without extension
        output_file = os.path.join(output_folder, f"{base_name}_middle_slice.png")
        
        plt.imshow(data_array[:, :, middle_slice], cmap='gray')
        plt.title(f"Middle Slice of {base_name}")
        plt.colorbar()
        plt.savefig(output_file)  # Save as PNG file
        plt.close()  # Close the figure to prevent overlap
        print(f"Image saved to {output_file}")
    else:
        print("Data is not 3D, cannot visualize slices.")

# Example usage
# pkl_file_path = "Release_pkl/masksTr/ThoraxCBCT_0000_0002.pkl"  # Replace with your .pkl file path
# output_folder = "Release_pkl/masksTr_showed"  # Replace with your desired output folder
# load_and_save_slice(pkl_file_path, output_folder)

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np  # Ensure NumPy is imported

def load_and_save_slice2(pkl_file, output_folder):
    """
    Loads a .pkl file and saves a slice of the data as an image.

    Parameters:
    - pkl_file (str): Path to the .pkl file to load.
    - output_folder (str): Path to the folder where images will be saved.
    """
    # Load the .pkl file
    with open(pkl_file, 'rb') as file:
        data_array = pickle.load(file)
    
    # Check if the data is already a NumPy array, or try to convert it
    if isinstance(data_array, np.ndarray):
        print(f"Data loaded successfully from {pkl_file} as a NumPy array.")
    elif isinstance(data_array, (list, tuple)):
        # Try to convert list or tuple to a NumPy array
        try:
            data_array = np.array(data_array)
            print(f"Data loaded from {pkl_file} and converted to a NumPy array.")
        except Exception as e:
            print(f"Error converting data from {pkl_file} to NumPy array: {e}")
            return
    else:
        print(f"Error: Data in {pkl_file} is not a NumPy array, list, or tuple.")
        return

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Check if data is 4D (e.g., (2, 160, 192, 160))
    if data_array.ndim == 4:
        # Select the first slice (index 0) for visualization
        data_3d = data_array[0, :, :, :]  # Select the first slice (if you want the second slice, use data_array[1, ...])
        print(f"Selected 3D data with shape: {data_3d.shape}")
        
        # Now extract the middle slice along Z-axis (index 2)
        middle_slice = data_3d.shape[2] // 2  # Middle slice along Z-axis
        
        # Generate a unique filename based on the input file name
        base_name = os.path.splitext(os.path.basename(pkl_file))[0]  # Extract base name without extension
        output_file = os.path.join(output_folder, f"{base_name}_middle_slice.png")
        
        # Display and save the middle slice
        plt.imshow(data_3d[:, :, middle_slice], cmap='gray')
        plt.title(f"Middle Slice of {base_name}")
        plt.colorbar()
        plt.savefig(output_file)  # Save as PNG file
        plt.close()  # Close the figure to prevent overlap
        print(f"Image saved to {output_file}")
    else:
        print(f"Error: Data in {pkl_file} is not 4D (shape: {data_array.shape}).")


# Example usage
# pkl_file_path = "Thorax_pairs/Train_Resized2/Thorax_pair_000.pkl"  # Replace with your .pkl file path
# output_folder = "Thorax_pairs/Train2_showed"  # Replace with your desired output folder
# load_and_save_slice2(pkl_file_path, output_folder)


def load_and_show_slice(pkl_file):
    """
    Loads a .pkl file containing 3D data and displays a middle slice.
    
    Parameters:
    - pkl_file (str): Path to the .pkl file containing 3D data.
    """
    # Load the .pkl file
    with open(pkl_file, 'rb') as file:
        data_array = pickle.load(file)
    
    # Check if the data is 3D (assuming the MRI data is 3D)
    if len(data_array.shape) == 3:
        # Extract the middle slice
        middle_slice = data_array.shape[2] // 2
        
        # Display the middle slice using matplotlib
        plt.imshow(data_array[:, :, middle_slice], cmap='gray')
        plt.title(f"Middle Slice of {pkl_file}")
        plt.colorbar()
        plt.show()
    else:
        print(f"Data is not 3D (shape: {data_array.shape}), cannot visualize.")

# Example usage
# pkl_file_path = 'dataset/LPBA_data/Train/subject_11.pkl'  # Replace with the actual file path
# load_and_show_slice(pkl_file_path)

# import pickle
# import numpy as np

# with open('Release_pkl/labelsTr/ThoraxCBCT_0011_0000.pkl', 'rb') as f:
#     data = pickle.load(f)

# label_image = data[0]['label']  # Adjust key if needed
# unique_labels = np.unique(label_image)
# print(unique_labels)  # Use these values in `self.seg_table`

import pickle

# Pot do datoteke
# path_to_file = 'Release_pkl/labelsTr/ThoraxCBCT_0012_0000.pkl'

# # Odpiranje in branje datoteke
# with open(path_to_file, 'rb') as f:
#     data = pickle.load(f)

# import numpy as np

# # label_image je že vaš naložen numpy array
# label_image = data  # Celotna struktura podatkov je array

# # Poiščite unikatne vrednosti
# unique_labels = np.unique(label_image)
# print("Unique labels:", unique_labels)

# import pandas as pd

# # Pot do datoteke
# file_path = 'Release_06_12_23/keypoints01Tr/ThoraxCBCT_0000_0000.csv'

# # Preberi CSV datoteko
# data = pd.read_csv(file_path)
# print(data.head())  # Prikaže prvih nekaj vrstic


import pickle

# file_path = "Thorax_pairs/Val_Resized/Thorax_pair_001.pkl"  # zamenjaj z dejansko potjo do .pkl datoteke

# with open(file_path, 'rb') as f:
#     data = pickle.load(f)

# print(f"Type of data: {type(data)}")
# if isinstance(data, (list, tuple)):
#     print(f"Length of data: {len(data)}")
#     for i, item in enumerate(data):
#         print(f"Item {i}: Type: {type(item)}, Shape: {getattr(item, 'shape', 'N/A')}")
# else:
#     print("Data format is unexpected.")

import glob

# Converting into tuple:
# file_paths = glob.glob("Thorax_pairs/Val_Resized/*.pkl")  # Prilagodi pot

# for path in file_paths:
#     with open(path, 'rb') as f:
#         data = pickle.load(f)

#     # Pretvori v tuple in ponovno shrani
#     data = tuple(data)
#     with open(path, 'wb') as f:
#         pickle.dump(data, f)

# print("Vse datoteke so posodobljene.")

# Preveri vse .pkl datoteke v datasetu
import pickle
import glob

train_files = glob.glob("Thorax_pairs/Resized3_Train/*.pkl")
for file in train_files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
        #print(f"File: {file}, Data length: {len(data)}")
        print(f"Fixed image size: {data[0].shape}, Moving image size: {data[1].shape}")

    


