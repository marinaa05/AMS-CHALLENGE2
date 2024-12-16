# import torch
# import pickle
# import os
# from pathlib import Path
# import torch.nn.functional as F

# # Funkcija za spremembo velikosti slike
# def resize_images(input_path, output_path, target_size=(160, 192, 160)):
#     # Preveri, če izhodna mapa obstaja
#     os.makedirs(output_path, exist_ok=True)

#     # Poišči vse datoteke .pkl v mapi
#     file_paths = list(Path(input_path).glob("*.pkl"))

#     for file_path in file_paths:
#         # Naloži .pkl datoteko
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)

#         # Pridobi slike (fixed in moving)
#         fixed_image = torch.tensor(data[0])  # [256, 192, 192]
#         moving_image = torch.tensor(data[1])  # [256, 192, 192]

#         # Preoblikuj v 3D format (N, C, D, H, W)
#         fixed_image = fixed_image.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 192, 192]
#         moving_image = moving_image.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 192, 192]

#         # Interpolacija na novo velikost  -- deli vsako dimenzijo
#         fixed_resized = F.interpolate(fixed_image, size=target_size, mode='trilinear', align_corners=False)
#         moving_resized = F.interpolate(moving_image, size=target_size, mode='trilinear', align_corners=False)

#         # Odstrani dodatne dimenzije
#         fixed_resized = fixed_resized.squeeze(0).squeeze(0)  # [160, 192, 160]
#         moving_resized = moving_resized.squeeze(0).squeeze(0)  # [160, 192, 160]

#         # Shranjevanje novih slik v datoteko
#         output_file = Path(output_path) / file_path.name
#         with open(output_file, 'wb') as f:
#             pickle.dump([fixed_resized.numpy(), moving_resized.numpy()], f)

#         print(f"Processed and saved: {output_file}")

# # Definiraj poti
# input_dir = "Thorax_pairs/Val"  # Kje so originalne slike
# output_dir = "Thorax_pairs/Val_Resized"  # Kam shraniti spremenjene slike

# # Pokliči funkcijo za spremembo velikosti
# resize_images(input_dir, output_dir, target_size=(160, 192, 160))

from PIL import Image
import numpy as np
import pickle
import os

def resize_image(image, target_shape):
    """
    Prilagodi dimenzije slike z upoštevanjem linearne interpolacije.

    Parameters:
        image (numpy.ndarray): Vhodna slika dimenzij (H, W, D).
        target_shape (tuple): Ciljne prostorske dimenzije (H, W, D).
    
    Returns:
        numpy.ndarray: Slika z ohranjenimi prostorskimi dimenzijami.
    """
    assert image.ndim == 3, "Slika mora imeti dimenzije (H, W, D)."
    assert len(target_shape) == 3, "Ciljna dimenzija mora biti (H, W, D)."

    target_H, target_W, target_D = target_shape

    # Spremeni sliko v PIL Image format za lažje skaliranje
    pil_image = Image.fromarray(image.astype(np.uint8))

    # Prilagoditev velikosti slike
    resized_image = pil_image.resize((target_W, target_H), Image.BILINEAR)

    # Ustvari novo sliko z željenimi prostorskimi dimenzijami
    resized_image = np.array(resized_image)  # pretvori nazaj v numpy array
    resized_image = np.stack([resized_image] * target_D, axis=-1)

    return resized_image

# Funkcija za procesiranje in shranjevanje obdelanih .pkl datotek
def process_and_save_resized_pkl(input_folder, output_folder, target_shape):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Ustvari mapo, če še ne obstaja

    for filename in os.listdir(input_folder):
        if filename.endswith(".pkl"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Naloži sliko iz .pkl datoteke
            with open(input_path, 'rb') as f:
                data = pickle.load(f)  # Predpostavka: Slika je shranjena kot numpy array

            # Preverjanje dimenzij pred obdelavo
            print(f"Originalna velikost fiksne slike: {data[0].shape}, premikajoče slike: {data[1].shape}")

            # Resizanje vsake slike posebej z linearno interpolacijo
            resized_fixed = resize_image(data[0], target_shape)
            resized_moving = resize_image(data[1], target_shape)
            resized_data = np.stack([resized_fixed, resized_moving], axis=0)  # Združi nazaj v (2, H, W, D)

            # Preverjanje dimenzij po obdelavi
            print(f"Nova velikost fiksne slike: {resized_data[0].shape}, premikajoče slike: {resized_data[1].shape}")

            # Shrani obdelano sliko nazaj v .pkl datoteko
            with open(output_path, 'wb') as f:
                pickle.dump(resized_data, f)

# Primer klica funkcije
input_folder = 'Thorax_pairs/Train'
output_folder = 'Thorax_pairs/Resized3_Train'
target_shape = (128, 96, 96)  # Spremeni velikost

process_and_save_resized_pkl(input_folder, output_folder, target_shape)
