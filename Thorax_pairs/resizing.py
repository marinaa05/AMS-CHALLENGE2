import torch
import pickle
import os
from pathlib import Path
import torch.nn.functional as F

# Funkcija za spremembo velikosti slike
def resize_images(input_path, output_path, target_size=(160, 192, 160)):
    # Preveri, če izhodna mapa obstaja
    os.makedirs(output_path, exist_ok=True)

    # Poišči vse datoteke .pkl v mapi
    file_paths = list(Path(input_path).glob("*.pkl"))

    for file_path in file_paths:
        # Naloži .pkl datoteko
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Pridobi slike (fixed in moving)
        fixed_image = torch.tensor(data[0])  # [256, 192, 192]
        moving_image = torch.tensor(data[1])  # [256, 192, 192]

        # Preoblikuj v 3D format (N, C, D, H, W)
        fixed_image = fixed_image.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 192, 192]
        moving_image = moving_image.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 192, 192]

        # Interpolacija na novo velikost
        fixed_resized = F.interpolate(fixed_image, size=target_size, mode='trilinear', align_corners=False)
        moving_resized = F.interpolate(moving_image, size=target_size, mode='trilinear', align_corners=False)

        # Odstrani dodatne dimenzije
        fixed_resized = fixed_resized.squeeze(0).squeeze(0)  # [160, 192, 160]
        moving_resized = moving_resized.squeeze(0).squeeze(0)  # [160, 192, 160]

        # Shranjevanje novih slik v datoteko
        output_file = Path(output_path) / file_path.name
        with open(output_file, 'wb') as f:
            pickle.dump([fixed_resized.numpy(), moving_resized.numpy()], f)

        print(f"Processed and saved: {output_file}")

# Definiraj poti
input_dir = "Thorax_pairs/Val"  # Kje so originalne slike
output_dir = "Thorax_pairs/Val_Resized"  # Kam shraniti spremenjene slike

# Pokliči funkcijo za spremembo velikosti
resize_images(input_dir, output_dir, target_size=(160, 192, 160))
