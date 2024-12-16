import os
import pickle
import numpy as np

def merge_and_save_patient_images(input_dir, output_dir):
    """
    Merges FBCT images with CBCT (pre-therapy and post-therapy) for each patient and saves them in pairs as tuples.

    Args:
        input_dir (str): Path to the directory containing .pkl files.
        output_dir (str): Path to the directory to save merged pairs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all patient files sorted by name
    patient_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pkl")])

    # Process files based on their naming pattern
    for no_patient in range(len(patient_files) // 3):
        patient_str = f"{no_patient:04d}"
        try:
            fbct_file = os.path.join(input_dir, f"ThoraxCBCT_{patient_str}_0000.pkl")
            cbct_pre_file = os.path.join(input_dir, f"ThoraxCBCT_{patient_str}_0001.pkl")
            cbct_post_file = os.path.join(input_dir, f"ThoraxCBCT_{patient_str}_0002.pkl")

            # Load FBCT and CBCT images
            with open(fbct_file, 'rb') as fbct:
                fbct_data = pickle.load(fbct)
            with open(cbct_pre_file, 'rb') as cbct_pre:
                cbct_pre_data = pickle.load(cbct_pre)
            with open(cbct_post_file, 'rb') as cbct_post:
                cbct_post_data = pickle.load(cbct_post)

            # Ensure the images have the expected size
            if fbct_data.shape != (170, 128, 128) or cbct_pre_data.shape != (170, 128, 128) or cbct_post_data.shape != (170, 128, 128):
                print(f"Error: Unexpected image shape for patient {no_patient}")
                continue

            # Merge FBCT and CBCT (pre-therapy) as a tuple
            merged_pre = (fbct_data, cbct_pre_data)
            pre_output_path = os.path.join(output_dir, f"merged_patient_{patient_str}_pre.pkl")
            with open(pre_output_path, 'wb') as out_file:
                pickle.dump(merged_pre, out_file)

            # Merge FBCT and CBCT (post-therapy) as a tuple
            merged_post = (fbct_data, cbct_post_data)
            post_output_path = os.path.join(output_dir, f"merged_patient_{patient_str}_post.pkl")
            with open(post_output_path, 'wb') as out_file:
                pickle.dump(merged_post, out_file)

            print(f"Successfully saved merged files for patient {patient_str}")
        except Exception as e:
            print(f"Error processing patient {patient_str}: {e}")

# Example usage
input_dir = "Release_pkl/Resized_normalized_imagesTr"  # Replace with the directory containing .pkl files
output_dir = "Release_pkl/Resized_merged_imagesTr"  # Replace with the directory to save merged files
merge_and_save_patient_images(input_dir, output_dir)
