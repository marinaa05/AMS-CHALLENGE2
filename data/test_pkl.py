import pickle
import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

path = "./dataset/LPBA_data/Train/subject_13.pkl"
with open(path, "rb") as file:
   res= pickle.load(file)
#print(type(res))
print(len(res))
#print(res[0])
print(res[0].shape)
#print(res[1])
print(res[1].shape)
#plt.imshow(res[0])
path = "Preprocessing/imagesTr_to_pkl/ThoraxCBCT_0000_0000.pkl"
with open(path, "rb") as file:
   res= pickle.load(file)
#print(type(res))
print(len(res))
#print(res[0])
print(res[0].shape)
#print(res[1])
# print(res[1].shape)
#plt.imshow(res[0])


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

# def load_and_save_slice2(pkl_file, output_folder):
    # """
    # Loads a .pkl file and saves a slice of the data as an image.

    # Parameters:
    # - pkl_file (str): Path to the .pkl file to load.
    # - output_folder (str): Path to the folder where images will be saved.
    # """
    # # Load the .pkl file
    # with open(pkl_file, 'rb') as file:
    #     data_array = pickle.load(file)
    
    # # Check if the data is already a NumPy array, or try to convert it
    # if isinstance(data_array, np.ndarray):
    #     print(f"Data loaded successfully from {pkl_file} as a NumPy array.")
    # elif isinstance(data_array, (list, tuple)):
    #     # Try to convert list or tuple to a NumPy array
    #     try:
    #         data_array = np.array(data_array)
    #         print(f"Data loaded from {pkl_file} and converted to a NumPy array.")
    #     except Exception as e:
    #         print(f"Error converting data from {pkl_file} to NumPy array: {e}")
    #         return
    # else:
    #     print(f"Error: Data in {pkl_file} is not a NumPy array, list, or tuple.")
    #     return

    # # Ensure output folder exists
    # os.makedirs(output_folder, exist_ok=True)

    # # Check if data is 4D (e.g., (2, 160, 192, 160))
    # if data_array.ndim == 4:
    #     # Select the first slice (index 0) for visualization
    #     data_3d = data_array[0, :, :, :]  # Select the first slice (if you want the second slice, use data_array[1, ...])
    #     print(f"Selected 3D data with shape: {data_3d.shape}")
        
    #     # Now extract the middle slice along Z-axis (index 2)
    #     middle_slice = data_3d.shape[2] // 2  # Middle slice along Z-axis
        
    #     # Generate a unique filename based on the input file name
    #     base_name = os.path.splitext(os.path.basename(pkl_file))[0]  # Extract base name without extension
    #     output_file = os.path.join(output_folder, f"{base_name}_middle_slice.png")
        
    #     # Display and save the middle slice
    #     plt.imshow(data_3d[:, :, middle_slice], cmap='gray')
    #     plt.title(f"Middle Slice of {base_name}")
    #     plt.colorbar()
    #     plt.savefig(output_file)  # Save as PNG file
    #     plt.close()  # Close the figure to prevent overlap
    #     print(f"Image saved to {output_file}")
    # else:
    #     print(f"Error: Data in {pkl_file} is not 4D (shape: {data_array.shape}).")


# Example usage
# pkl_file_path = "dataset/LPBA_data/Train/subject_11.pkl"  # Replace with your .pkl file path
# output_folder = "dataset/LPBA_data/Train_images"  # Replace with your desired output folder
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
pkl_file_path = 'dataset/LPBA_data/Train/subject_11.pkl'  # Replace with the actual file path
load_and_show_slice(pkl_file_path)
