import os, glob
import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle
import numpy as np
from skimage.io import imread

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class ThoraxDatasetSequentialPre(Dataset):
    def __init__(self, data_dir, transforms=None):
        """
        Args:
            data_dir (str): Pot do mape z vsemi slikami.
            transforms (callable, optional): Transformacije za slike.
        """
        # Preveri vhodni argument
        if not isinstance(data_dir, (str, os.PathLike)):
            raise TypeError("data_dir must be a string or PathLike object, got {}".format(type(data_dir)))
        
        self.data_dir = data_dir
        self.transforms = transforms

        # Branje vseh poti do slik in sortiranje po vrstnem redu
        self.image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])

        # Preveri, da imamo slike
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in directory: {data_dir}")

    def __len__(self):
        return len(self.image_paths) // 3 # Število pacientov

    def __getitem__(self, index):
        # Preberi FBCT in CBCT1 za trenutnega pacienta
        fbct_path = self.image_paths[3*index]
        cbct_path = self.image_paths[3*index+1]

        with open(fbct_path, 'rb') as f:
            fbct = pickle.load(f)

        with open(cbct_path, 'rb') as f:
            cbct = pickle.load(f)

        # Normalizacija na [0, 1]
        # fbct = (fbct - fbct.min()) / (fbct.max() - fbct.min())
        # cbct1 = (cbct1 - cbct1.min()) / (cbct1.max() - cbct1.min())

        # Dodaj dimenzijo kanala
        fbct = fbct[None, ...]
        cbct = cbct[None, ...]

        # Uporabi transformacije, če so definirane (FBCT je moving, CBCT je fixed)
        if self.transforms:
            fbct, cbct = self.transforms((fbct, cbct))

        # Pretvori v PyTorch tenzor
        fbct = torch.from_numpy(fbct)
        cbct = torch.from_numpy(cbct)

        return fbct, cbct

class ThoraxDatasetSequentialPost(Dataset):
    def __init__(self, data_dir, transforms=None):
        """
        Args:
            data_dir (str): Pot do mape z vsemi slikami.
            transforms (callable, optional): Transformacije za slike.
        """
        # Preveri vhodni argument
        if not isinstance(data_dir, (str, os.PathLike)):
            raise TypeError("data_dir must be a string or PathLike object, got {}".format(type(data_dir)))
        
        self.data_dir = data_dir
        self.transforms = transforms

        # Branje vseh poti do slik in sortiranje po vrstnem redu
        self.image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])

        # Preveri, da imamo slike
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in directory: {data_dir}")

    def __len__(self):
        return len(self.image_paths) // 3 # Število pacientov

    def __getitem__(self, index):
        # Preberi FBCT in CBCT2 za trenutnega pacienta
        fbct_path = self.image_paths[3*index]
        cbct_path = self.image_paths[3*index+2]

        with open(fbct_path, 'rb') as f:
            fbct = pickle.load(f)

        with open(cbct_path, 'rb') as f:
            cbct = pickle.load(f)

        # Normalizacija na [0, 1]
        # fbct = (fbct - fbct.min()) / (fbct.max() - fbct.min())
        # cbct = (cbct - cbct2.min()) / (cbct.max() - cbct.min())

        # Dodaj dimenzijo kanala
        fbct = fbct[None, ...]
        cbct = cbct[None, ...]

        # Uporabi transformacije, če so definirane
        if self.transforms:
            fbct, cbct = self.transforms([fbct, cbct])

        # Pretvori v PyTorch tenzor
        fbct = torch.from_numpy(fbct)
        cbct = torch.from_numpy(cbct)

        return fbct, cbct