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

class LPBABrainDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)


class LPBABrainInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index//(len(self.paths)-1)
        s = index%(len(self.paths)-1)
        y_index = s+1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x) # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)


class LPBABrainHalfDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    def half_pair(self,pair):
        return pair[0][::2,::2,::2], pair[1][::2,::2,::2]

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = self.half_pair(pkload(path_x))
        y, y_seg = self.half_pair(pkload(path_y))
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)


class LPBABrainHalfInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    def half_pair(self,pair):
        return pair[0][::2,::2,::2], pair[1][::2,::2,::2]
    def __getitem__(self, index):
        x_index = index//(len(self.paths)-1)
        s = index%(len(self.paths)-1)
        y_index = s+1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = self.half_pair(pkload(path_x))
        y, y_seg = self.half_pair(pkload(path_y))
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)

class ThoraxDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        # Pridobivanje indeksov za par podatkov (fixed, moving)
        # x_index = index // (len(self.paths) - 1)
        # s = index % (len(self.paths) - 1)
        # y_index = s + 1 if s >= x_index else s
        x_index = index

        # Pridobitev poti do datotek
        path_x = self.paths[x_index]
        # path_y = self.paths[y_index]

        # Naložitev podatkov
        # x, x_seg = pkload(path_x)
        # y, y_seg = pkload(path_y)
        x, y = pkload(path_x)

        # Dodajanje dimenzije kanala
        x, y = x[None, ...], y[None, ...]

        # Transformacije (če so podane)
        if self.transforms:
            x, y = self.transforms([x, y])

        # Pretvorba v contiguous array
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        # Pretvorba v PyTorch tensorje
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        return x, y

    def __len__(self):
        # Izračun dolžine dataset-a glede na vse možne pare
        # return len(self.paths) * (len(self.paths) - 1)
        return len(self.paths)
    
class ThoraxInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms=None):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        # Pridobivanje indeksov za par podatkov (fixed, moving)
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s

        # Naložitev podatkov
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        # x, x_seg = pkload(path_x)
        x, y = pkload(path_x)
        # y, y_seg = pkload(path_y)

        # Dodajanje dimenzije kanala
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        # Transformacije (če so podane)
        if self.transforms:
            x, x_seg = self.transforms([x, x_seg])
            y, y_seg = self.transforms([y, y_seg])

        # Pretvorba v contiguous array
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        # Pretvorba v PyTorch tensorje
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x_seg, y_seg = torch.from_numpy(x_seg), torch.from_numpy(y_seg)

        return x, y, x_seg, y_seg

    def __len__(self):
        # Izračun dolžine dataset-a glede na vse možne pare
        return len(self.paths) * (len(self.paths) - 1)

class ThoraxDatasetPairwise(Dataset):
    def __init__(self, data_paths, transforms=None):
        """
        Inicializacija dataset razreda.
        Args:
            data_paths (list): Seznam poti do `.pkl` datotek.
            transforms (callable, optional): Funkcije za transformacije podatkov.
        """
        self.paths = sorted(data_paths)  # Uredi, da so slike pravilno poravnane
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Pridobi FBCT in CBCT par za določenega pacienta.
        Args:
            index (int): Indeks pacienta.
        Returns:
            torch.Tensor: FBCT slika.
            torch.Tensor: CBCT slika.
        """
        path = self.paths[index]
        data = pkload(path)  # Naloži .pkl datoteko
        fbct, cbct = data[0], data[1]  # Razčleni na FBCT in CBCT

        fbct, cbct = fbct.astype(np.float32), cbct.astype(np.float32)

        # Dodaj dimenzijo kanala (če ni prisotna)
        fbct, cbct = fbct[None, ...], cbct[None, ...]

        # Uporabi transformacije (če so definirane)
        if self.transforms:
            fbct, cbct = self.transforms([fbct, cbct])

        # Pretvori v PyTorch tenzor
        fbct = torch.from_numpy(np.ascontiguousarray(fbct))
        cbct = torch.from_numpy(np.ascontiguousarray(cbct))

        return fbct, cbct

    def __len__(self):
        """Vrne število pacientov v datasetu."""
        return len(self.paths)
    
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

        # Razvrsti slike na FBCT in CBCT1
        # self.fbct_paths = self.image_paths[::3]  # FBCT slike
        # self.cbct1_paths = self.image_paths[1::3]  # CBCT1 slike

    def __len__(self):
        return len(self.image_paths) // 3 # Število pacientov

    def __getitem__(self, index):
        # Preberi FBCT in CBCT1 za trenutnega pacienta
        fbct_path = self.image_paths[3*index]
        cbct1_path = self.image_paths[3*index+1]

        with open(fbct_path, 'rb') as f:
            fbct = pickle.load(f)

        with open(cbct1_path, 'rb') as f:
            cbct1 = pickle.load(f)

        # Normalizacija na [0, 1]
        fbct = (fbct - fbct.min()) / (fbct.max() - fbct.min())
        cbct1 = (cbct1 - cbct1.min()) / (cbct1.max() - cbct1.min())

        # Dodaj dimenzijo kanala
        fbct = fbct[None, ...]
        cbct1 = cbct1[None, ...]

        # Uporabi transformacije, če so definirane
        if self.transforms:
            fbct, cbct1 = self.transforms([fbct, cbct1])

        # Pretvori v PyTorch tenzor
        fbct = torch.from_numpy(fbct)
        cbct1 = torch.from_numpy(cbct1)

        return fbct, cbct1

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
        cbct2_path = self.image_paths[3*index+2]

        with open(fbct_path, 'rb') as f:
            fbct = pickle.load(f)

        with open(cbct2_path, 'rb') as f:
            cbct2 = pickle.load(f)

        # Normalizacija na [0, 1]
        fbct = (fbct - fbct.min()) / (fbct.max() - fbct.min())
        cbct2 = (cbct2 - cbct2.min()) / (cbct2.max() - cbct2.min())

        # Dodaj dimenzijo kanala
        fbct = fbct[None, ...]
        cbct2 = cbct2[None, ...]

        # Uporabi transformacije, če so definirane
        if self.transforms:
            fbct, cbct2 = self.transforms([fbct, cbct2])

        # Pretvori v PyTorch tenzor
        fbct = torch.from_numpy(fbct)
        cbct2 = torch.from_numpy(cbct2)

        return fbct, cbct2

class ThoraxDatasetSequentialPre2(Dataset):
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
        # Preberi FBCT in CBCT za trenutnega pacienta
        fbct_path = self.image_paths[3*index]
        cbct_path = self.image_paths[3*index+1]

        with open(fbct_path, 'rb') as f:
            fbct = pickle.load(f)

        with open(cbct_path, 'rb') as f:
            cbct = pickle.load(f)

        # Normalizacija na [0, 1]
        fbct = (fbct - fbct.min()) / (fbct.max() - fbct.min())
        cbct = (cbct - cbct.min()) / (cbct.max() - cbct.min())

        # Dodaj dimenzijo kanala
        fbct = fbct[None, ...]
        cbct = cbct[None, ...]

        # Uporabi transformacije, če so definirane
        if self.transforms:
            cbct, fbct = self.transforms([cbct, fbct])

        # Pretvori v PyTorch tenzor
        fbct = torch.from_numpy(fbct)
        cbct = torch.from_numpy(cbct)

        return cbct, fbct

class ThoraxDatasetSequentialPost2(Dataset):
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
        cbct2_path = self.image_paths[3*index+2]

        with open(fbct_path, 'rb') as f:
            fbct = pickle.load(f)

        with open(cbct2_path, 'rb') as f:
            cbct2 = pickle.load(f)

        # Normalizacija na [0, 1]
        fbct = (fbct - fbct.min()) / (fbct.max() - fbct.min())
        cbct2 = (cbct2 - cbct2.min()) / (cbct2.max() - cbct2.min())

        # Dodaj dimenzijo kanala
        fbct = fbct[None, ...]
        cbct2 = cbct2[None, ...]

        # Uporabi transformacije, če so definirane
        if self.transforms:
            fbct, cbct2 = self.transforms([fbct, cbct2])

        # Pretvori v PyTorch tenzor
        fbct = torch.from_numpy(fbct)
        cbct2 = torch.from_numpy(cbct2)

        return cbct2, fbct 