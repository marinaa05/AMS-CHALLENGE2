from data import datasets
import os, utils
from torch.utils.data import DataLoader
import numpy as np
from data import datasets, trans
from torchvision import transforms
import nibabel as nib

file_path = "new_dof_post_final_55/def_polja_new/disp_0011_0002_0011_0000.nii.gz"

# Naloži datoteko
img = nib.load(file_path)
data = img.get_fdata()

# Dimenzije in število točk
print(f"Dimenzije podatkov: {data.shape}")
print(f"Skupno število točk: {np.prod(data.shape)}")
print(f"Velikost podatkov (približno v MB): {data.nbytes / (1024 ** 2):.2f} MB")




