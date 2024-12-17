import os
import glob
import numpy as np
import SimpleITK as sitk
from natsort import natsorted
import pickle
import csv
import matplotlib.pyplot as plt

# Funkcije iz tvoje kode
def pksave(img, label, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump((img, label), f)

def nii2arr(nii_img):
    return sitk.GetArrayFromImage(sitk.ReadImage(nii_img))

def center(arr):
    c = np.sort(np.nonzero(arr))[:, [0, -1]]
    return np.mean(c, axis=-1).astype('int16')

def cropByCenter(image, center, final_shape=(256, 192, 192)):
    """
    Izreže 3D sliko okoli določenega centra na podano velikost.
    """
    crop = np.array([s // 2 for s in final_shape])  # Polovica velikosti za obrezovanje
    crop_ranges = []

    for dim in range(3):  # Za vsako dimenzijo preverimo meje
        cropmin, cropmax = center[dim] - crop[dim], center[dim] + crop[dim]
        
        # Popravimo meje, če segajo izven slike
        if cropmin < 0:
            cropmin, cropmax = 0, final_shape[dim]
        if cropmax > image.shape[dim]:
            cropmax = image.shape[dim]
            cropmin = image.shape[dim] - final_shape[dim]
        
        crop_ranges.append(slice(cropmin, cropmax))
    
    # Preveri obrezane dimenzije
    cropped_image = image[tuple(crop_ranges)]
    print("Center:", center, "Crop ranges:", crop_ranges, "Cropped shape:", cropped_image.shape)
    
    # Preverimo, ali je velikost ustrezna
    if cropped_image.shape != final_shape:
        print("WARNING: Cropped image shape does not match final_shape")
    return cropped_image

def minmax(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# Branje keypoints iz CSV
def load_keypoints(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        keypoints = [list(map(float, row)) for row in reader]
    return np.array(keypoints, dtype=np.int16)

# Pot do podatkov
path_to_images = 'Release_06_12_23/imagesTr/'
path_to_labels = 'Release_06_12_23/labelsTr/'
path_to_keypoints01 = 'Release_06_12_23/keypoints01Tr/'
path_to_keypoints02 = 'Release_06_12_23/landmarks02Tr/'

save_path = 'pklDataset_Poskus2/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Seznam slik in keypoints
images = natsorted(glob.glob(os.path.join(path_to_images, '*.nii.gz')))
labels = natsorted(glob.glob(os.path.join(path_to_labels, '*.nii.gz')))

print("Number of images:", len(images))
print("Number of labels:", len(labels))

# Loop čez vse paciente
for patient_idx in range(14):
    for img_idx in range(3):  # FBCT (0), CBCT1 (1), CBCT2 (2)
        img_path = f'ThoraxCBCT_{patient_idx:04d}_{img_idx:04d}.nii.gz'
        img_full_path = os.path.join(path_to_images, img_path)
        
        if not os.path.exists(img_full_path):
            print(f"Image not found: {img_full_path}")
            continue
        
        print(f"\nProcessing {img_full_path}")
        img = nii2arr(img_full_path)
        print(f"Original image shape: {img.shape}")
        
        # Preveri keypoints za obrezovanje
        if img_idx == 0:
            keypoints_path = os.path.join(path_to_keypoints01, f'ThoraxCBCT_{patient_idx:04d}_{img_idx:04d}.csv')
        elif img_idx == 1:
            keypoints_path = os.path.join(path_to_keypoints01, f'ThoraxCBCT_{patient_idx:04d}_{img_idx:04d}.csv')
        else:
            keypoints_path = os.path.join(path_to_keypoints02, f'ThoraxCBCT_{patient_idx:04d}_{img_idx:04d}.csv')
        
        if os.path.exists(keypoints_path):
            keypoints = load_keypoints(keypoints_path)
            center_point = np.mean(keypoints, axis=0).astype('int16')
            print(f"Loaded keypoints, calculated center: {center_point}")
        else:
            center_point = np.array(img.shape) // 2  # Privzeti center slike
            print(f"No keypoints found, using default center: {center_point}")
        
        # Obrezovanje in normalizacija
        img_cropped = cropByCenter(img, center_point, final_shape=(170, 128, 128))
        img_normalized = minmax(img_cropped).astype('float32')
        print(f"Final image shape: {img_normalized.shape}, Range: [{img_normalized.min()}, {img_normalized.max()}]")
        
        # Branje pripadajočega labela (samo za paciente 11, 12, 13)
        label_cropped = None
        if patient_idx >= 11:
            label_path = os.path.join(path_to_labels, f'ThoraxCBCT_{patient_idx:04d}_{img_idx:04d}.nii.gz')
            if os.path.exists(label_path):
                label = nii2arr(label_path)
                label_cropped = cropByCenter(label, center_point, final_shape=(170, 128, 128)).astype('uint16')
                print(f"Label loaded and cropped: {label_cropped.shape}")
            else:
                print(f"Label not found for {label_path}")
        
        # Shranjevanje obdelanih podatkov
        save_file = os.path.join(save_path, f'patient_{patient_idx:02d}_{img_idx:02d}.pkl')
        pksave(img_normalized, label_cropped, save_file)
        print(f"Saved to {save_file}")


def display_three_views(image, save_dir, filename_prefix):
    """
    Prikaže pravilno orientirane sagitalne, koronalne in aksialne rezine v eni sliki.

    Args:
        image (numpy array): 3D slika.
        save_dir (str): Ciljna mapa za shranjevanje.
        filename_prefix (str): Predpona za ime shranjene slike.
    """
    # Preverjanje in ustvarjanje ciljne mape
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Obrat osi za pravilno orientacijo
    sagital_slice = np.rot90(image[image.shape[0] // 2, :, :])   # Sagitalni pogled (x-os)
    coronal_slice = np.rot90(image[:, image.shape[1] // 2, :])   # Koronalni pogled (y-os)
    axial_slice = np.rot90(image[:, :, image.shape[2] // 2])     # Aksialni pogled (z-os)

    # Ustvarimo subplot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Sagitalni pogled
    axes[0].imshow(sagital_slice, cmap='gray', aspect='auto')
    axes[0].set_title('Sagitalni pogled')
    axes[0].axis('off')

    # Koronalni pogled
    axes[1].imshow(coronal_slice, cmap='gray', aspect='auto')
    axes[1].set_title('Koronalni pogled')
    axes[1].axis('off')

    # Aksialni pogled
    axes[2].imshow(axial_slice, cmap='gray', aspect='auto')
    axes[2].set_title('Aksialni pogled')
    axes[2].axis('off')

    # Shranjevanje slike
    save_path = os.path.join(save_dir, f"{filename_prefix}.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {save_path}")

# Primer uporabe
path_to_processed_data = 'pklDataset_Poskus2/'
save_visualizations_dir = 'Three_Views_NEW_Corrected/'

if not os.path.exists(save_visualizations_dir):
    os.makedirs(save_visualizations_dir)

processed_files = sorted([f for f in os.listdir(path_to_processed_data) if f.endswith('.pkl')])

# Obdelava vseh .pkl datotek
for pkl_file in processed_files:
    pkl_path = os.path.join(path_to_processed_data, pkl_file)
    
    # Preberi sliko iz .pkl datoteke
    with open(pkl_path, 'rb') as f:
        img, _ = pickle.load(f)
    
    print(f"Processing {pkl_file}, Image shape: {img.shape}")
    
    # Prikaz treh pogledov
    display_three_views(img, save_dir=save_visualizations_dir, filename_prefix=pkl_file.replace('.pkl', ''))


