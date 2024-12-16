import pickle
import SimpleITK as sitk
import numpy as np
import glob
from natsort import natsorted
import os

def pksave(img, label, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump((img, label), f)

def nii2arr(nii_img):
    return sitk.GetArrayFromImage(sitk.ReadImage(nii_img))

def center(arr):
    c = np.sort(np.nonzero(arr))[:,[0,-1]]
    return np.mean(c, axis=-1).astype('int16')

def minmax(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def cropByCenter(image,center,final_shape=(160,192,160)):
    c = center
    crop = np.array([s // 2 for s in final_shape])
    # 0 axis
    cropmin, cropmax = c[0] - crop[0], c[0] + crop[0]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[0]
    if cropmax > image.shape[0]:
        cropmax = image.shape[0]
        cropmin = image.shape[0] - final_shape[0]
    image = image[cropmin:cropmax, :, :]
    # 1 axis
    cropmin, cropmax = c[1] - crop[1], c[1] + crop[1]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[1]
    if cropmax > image.shape[1]:
        cropmax = image.shape[1]
        cropmin = image.shape[1] - final_shape[1]
    image = image[:, cropmin:cropmax, :]

    # 2 axis
    cropmin, cropmax = c[2] - crop[2], c[2] + crop[2]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[2]
    if cropmax > image.shape[2]:
        cropmax = image.shape[2]
        cropmin = image.shape[2] - final_shape[2]
    image = image[:, :, cropmin:cropmax]
    return image

# path_to_LPBA='/data/LPBA40/' # the path of the original dataset
# img_niis = natsorted(glob.glob(path_to_LPBA+'*/*/*skullstripped.img.gz'))
# label_niis = natsorted(glob.glob(path_to_LPBA+'*/*/*label.img.gz'))
# print(img_niis, label_niis)

# save_path = 'LPBA_data/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# for i, nii in enumerate(zip(img_niis, label_niis)):
#     print(nii)
#     img_nii, label_nii = nii
#     img, label = nii2arr(img_nii), nii2arr(label_nii)
#     print(img.shape, label.shape)
    
#     # crop by center
#     c = center(img)
#     img = cropByCenter(img, c)
#     label = cropByCenter(label, c)

#     #norm
#     img = minmax(img).astype('float32')
#     label = label.astype('uint16')
#     print(img.shape,np.unique(img),label.dtype, label.shape,np.unique(label),label.dtype)
#     print(save_path+'subject_%02d.pkl'%(i+1))
#     pksave(img,label, save_path=save_path+'subject_%02d.pkl'%(i+1))

# Za nov dataset:
path_to_dataset = '/workspace/modetv2/Release_06_12_23/imagesTr'  # Poskrbi, da ima pravilno pot

# Preberi vse datoteke
files_in_dir = os.listdir(path_to_dataset)
files_in_dir = [f for f in files_in_dir if f.endswith('.nii.gz')]  # Filtriraj samo .nii.gz datoteke

# Inicializiraj strukturo za organizacijo datotek
data = {}

for file in files_in_dir:
    # Razčleni ime datoteke
    filename = os.path.basename(file)
    patient_id = filename.split('_')[1]  # ID pacienta, npr. '0001'
    img_type = filename.split('_')[2].split('.')[0]  # Tip slike, npr. '0000', '0001', '0002'

    # Dodaj datoteko v strukturo
    if patient_id not in data:
        data[patient_id] = {'FBCT': None, 'CBCT_pre': None, 'CBCT_post': None}
    if img_type == '0000':
        data[patient_id]['FBCT'] = os.path.join(path_to_dataset, file)
    elif img_type == '0001':
        data[patient_id]['CBCT_pre'] = os.path.join(path_to_dataset, file)
    elif img_type == '0002':
        data[patient_id]['CBCT_post'] = os.path.join(path_to_dataset, file)

# Izpiši razvrščene datoteke
print("Organized data structure:", data)

save_path = 'processed_ThoraxCBCT/'  # Pot za shranjevanje obdelanih datotek
if not os.path.exists(save_path):
    os.makedirs(save_path)

for patient_id, files in data.items():
    print(f"Processing patient {patient_id}...")
    
    for img_type, img_path in files.items():
        if img_path is None:
            print(f"Missing {img_type} for patient {patient_id}")
            continue  # Preskoči, če slika manjka
        
        print(f"Processing {img_type} for patient {patient_id}: {img_path}")
        
        try:
            # Preberi sliko
            img = nii2arr(img_path)
            print(f"Image shape: {img.shape}")
            
            # Izračunaj center in obreži
            c = center(img)
            img = cropByCenter(img, c, final_shape=(128, 128, 170))
            print(f"Cropped shape: {img.shape}")
            
            # Normalizacija
            img = minmax(img).astype('float32')
            print(f"Normalized shape: {img.shape}, dtype: {img.dtype}")
            
            # Shranjevanje kot .pkl
            save_file = os.path.join(save_path, f"patient_{patient_id}_{img_type}.pkl")
            pksave(img, None, save_path=save_file)
            print(f"Saved to {save_file}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
