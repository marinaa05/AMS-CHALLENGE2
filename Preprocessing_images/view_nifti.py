import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

def display_and_save_nii_gz(file_path, output_dir):
    # Naloži .nii.gz datoteko
    img = nib.load(file_path)
    data = img.get_fdata()

    # Preveri dimenzije podatkov
    if data.ndim == 4:
        print("Podatki so 4D. Obdelujem posamezne kanale četrte dimenzije.")
    elif data.ndim != 3:
        raise ValueError(f"Podatki morajo biti 3D ali 4D, vendar so {data.ndim}D.")

    # Preveri, če mapa za shranjevanje obstaja, če ne, jo ustvari
    os.makedirs(output_dir, exist_ok=True)

    # Obdelava za vsak kanal četrte dimenzije (če obstaja)
    channels = range(data.shape[3]) if data.ndim == 4 else [None]

    for channel in channels:
        if channel is not None:
            print(f"Obdelujem kanal {channel}.")
            channel_data = data[..., channel]
        else:
            channel_data = data

        # Izberi srednjo rezino za vsako os
        slice_axial = channel_data[:, :, channel_data.shape[2] // 2]
        slice_coronal = channel_data[:, channel_data.shape[1] // 2, :]
        slice_sagittal = channel_data[channel_data.shape[0] // 2, :, :]

        # Shranjevanje rezin kot slike
        channel_suffix = f"_channel_{channel}" if channel is not None else ""
        plt.imsave(os.path.join(output_dir, f"axial_slice{channel_suffix}.png"), slice_axial.T, cmap="gray", origin="lower")
        plt.imsave(os.path.join(output_dir, f"coronal_slice{channel_suffix}.png"), slice_coronal.T, cmap="gray", origin="lower")
        plt.imsave(os.path.join(output_dir, f"sagittal_slice{channel_suffix}.png"), slice_sagittal.T, cmap="gray", origin="lower")

    print(f"Rezine so bile shranjene v: {output_dir}")

def check_nii_gz_size(file_path):
    # Naloži .nii.gz datoteko
    img = nib.load(file_path)
    data = img.get_fdata()

    # Prikaže osnovne informacije
    print("Velikost .nii.gz datoteke (v točkah):", data.shape)
    print("Skupno število točk:", np.prod(data.shape))

# Uporaba funkcije
data_file = "dof_final/patient_0_flow.nii.gz"
output_directory = "dof_final/viewed_images"
display_and_save_nii_gz(data_file, output_directory)

# Preverjanje velikosti
data_file = "dof_final/patient_0_flow.nii.gz"
check_nii_gz_size(data_file)
