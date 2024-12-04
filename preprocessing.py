import pickle
import nibabel as nib
import os
import numpy as np

# def load_nifti(file_path):
#     img = nib.load(file_path)
#     return img.get_fdata()

def load_nifti(file_path):
    """Load NIfTI file and normalize its values to a range of [0, 1]."""
    img = nib.load(file_path)
    data = img.get_fdata()
    # Normalize to range [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max > data_min:  # Avoid division by zero
        data = (data - data_min) / (data_max - data_min)
    else:
        data = np.zeros_like(data)  # Handle edge case where max == min
    return data + 0.001

def save_as_pkl(fixed_image, moving_image, save_path):
    data_tuple = (fixed_image, moving_image)
    with open(save_path, 'wb') as f:
        pickle.dump(data_tuple, f)


# file_name = "ThoraxCBCT_00{}_000{}.nii.gz"
# # Example for one patient
# patient_folder = "Dataset/Patient01"
# fixed_image = load_nifti(os.path.join(patient_folder, file_name))
# moving_image = load_nifti(os.path.join(patient_folder, "CBCT1.nii.gz"))  # Use CBCT1

# # Save as a single .pkl file
os.makedirs("ProcessedData/Train", exist_ok=True)
os.makedirs("ProcessedData/Val", exist_ok=True)
# save_as_pkl(fixed_image, moving_image, "ProcessedData/Train/subject_01.pkl")

image_dir_path = "Release_06_12_23/imagesTr"

file_name = "ThoraxCBCT_00{}_000{}.nii.gz"
for patient_num in range(0,14):

    patient_num = str(patient_num)
    if len(patient_num) == 1:
        patient_num = "0" + patient_num
    
    fixed_image_path = os.path.join(image_dir_path, file_name.format(patient_num, 0))
    moving_image_path = os.path.join(image_dir_path, file_name.format(patient_num, 2))

    fixed_image = load_nifti(fixed_image_path)
    moving_image = load_nifti(moving_image_path)

    patient_num_int= int(patient_num)

    if patient_num_int >= 0 and patient_num_int <= 10:
        processed_dir = "ProcessedData/Train"
    elif patient_num_int >=11 and patient_num_int <=13:
        processed_dir = "ProcessedData/Val"

    pkl_file_name = "ThoraxCBCT_00{}.pkl".format(patient_num)
    processed_dir = os.path.join(processed_dir, pkl_file_name)


    save_as_pkl(fixed_image, moving_image, processed_dir)


# print(fixed_image)



    