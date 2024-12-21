from data import datasets
import os, utils
from torch.utils.data import DataLoader
import numpy as np
from data import datasets, trans
from torchvision import transforms
import nibabel as nib
import matplotlib.pyplot as plt
import pickle

# Pot do mape s podatki
data_dir = "Release_pkl/Resized_normalized_imagesTr/Train"

import os

# Inicializiraj razred
dataset = datasets.ThoraxDatasetSequentialPre(data_dir)

# Preveri, ali so poti pravilno prebrane in sortirane
print("Prebrane poti:")
for path in dataset.image_paths:
    print(path)








