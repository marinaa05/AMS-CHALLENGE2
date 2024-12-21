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

        # Priprava subplots za prikaz vseh treh dimenzij
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(slice_axial.T, cmap="gray", origin="lower")
        axes[0].set_title("Axial")
        axes[0].axis("off")

        axes[1].imshow(slice_coronal.T, cmap="gray", origin="lower")
        axes[1].set_title("Coronal")
        axes[1].axis("off")

        axes[2].imshow(slice_sagittal.T, cmap="gray", origin="lower")
        axes[2].set_title("Sagittal")
        axes[2].axis("off")

        # Shranjevanje slike z unikatnim imenom
        channel_suffix = f"_channel_{channel}" if channel is not None else ""
        base_name = os.path.basename(file_path).replace('.nii.gz', '')
        output_file = os.path.join(output_dir, f"{base_name}{channel_suffix}_combined.png")
        plt.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

        print(f"Slika za kanal {channel if channel is not None else 0} shranjena kot {output_file}.")

def check_nii_gz_size(file_path):
    # Naloži .nii.gz datoteko
    img = nib.load(file_path)
    data = img.get_fdata()

    # Preveri dimenzije podatkov
    if data.ndim == 4:
        print("Podatki so 4D. Prikazujem velikost po kanalih.")
        for channel in range(data.shape[3]):
            print(f"Kanal {channel}: {data[..., channel].shape}")
    elif data.ndim != 3:
        raise ValueError(f"Podatki morajo biti 3D ali 4D, vendar so {data.ndim}D.")

    # Prikaže osnovne informacije
    print("Velikost .nii.gz datoteke (v točkah):", data.shape)
    print("Skupno število točk:", np.prod(data.shape))

# Uporaba funkcije
data_file = "Release_06_12_23/imagesTr/ThoraxCBCT_0000_0002.nii.gz"
output_directory = "Release_06_12_23/imagesTr_showed"
display_and_save_nii_gz(data_file, output_directory)

# Preverjanje velikosti
# data_file = "pot_do_datoteke/ime_datoteke.nii.gz"
# check_nii_gz_size(data_file)
