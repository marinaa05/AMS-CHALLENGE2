import os, glob
import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle
import numpy as np

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
# def pksave(data, fname):
#     """Funkcija za shranjevanje podatkov v pickle datoteko."""
#     with open(fname, 'wb') as f:
#         pickle.dump(data, f)

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


# Kam shraniti:
data_directory = "Thorax_pairs/Val"
os.makedirs(data_directory, exist_ok=True)  # Ustvari mapo, če ne obstaja

# Definirajte datoteko za shranjevanje parov
# output_file = os.path.join(data_directory, "Thorax_pairs_1.txt")

# data_paths = [("Release_pkl/imagesTr/ThoraxCBCT_0000_0000.pkl",           "Release_pkl/imagesTr/ThoraxCBCT_0000_0001.pkl"),
#               ("Release_pkl/imagesTr/ThoraxCBCT_0001_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0001_0001.pkl"),
#               ("Release_pkl/imagesTr/ThoraxCBCT_0002_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0002_0001.pkl"),
#               ("Release_pkl/imagesTr/ThoraxCBCT_0003_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0003_0001.pkl"),("Release_pkl/imagesTr/ThoraxCBCT_0004_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0004_0001.pkl"),("Release_pkl/imagesTr/ThoraxCBCT_0005_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0005_0001.pkl"),("Release_pkl/imagesTr/ThoraxCBCT_0006_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0006_0001.pkl"),("Release_pkl/imagesTr/ThoraxCBCT_0007_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0007_0001.pkl"),("Release_pkl/imagesTr/ThoraxCBCT_0008_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0008_0001.pkl"),("Release_pkl/imagesTr/ThoraxCBCT_0009_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0009_0001.pkl"),("Release_pkl/imagesTr/ThoraxCBCT_0010_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0010_0001.pkl"),]  # Ustvarite pare iz datotek

data_paths = [("Release_pkl/imagesTr/ThoraxCBCT_0011_0000.pkl",           "Release_pkl/imagesTr/ThoraxCBCT_0011_0002.pkl"),
              ("Release_pkl/imagesTr/ThoraxCBCT_0012_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0012_0002.pkl"),
              ("Release_pkl/imagesTr/ThoraxCBCT_0013_0000.pkl", "Release_pkl/imagesTr/ThoraxCBCT_0013_0002.pkl")]  # Ustvarite pare iz datotek

# Procesiranje in shranjevanje
# for i, (fbct_path, cbct_path) in enumerate(data_paths):
#     # Naloži FBCT in CBCT slike
#     fbct_image = pkload(fbct_path)
#     cbct_image = pkload(cbct_path)

#     # Normalizacija (če je potrebna)  [0 1]
#     fbct_image = fbct_image / np.max(fbct_image)
#     cbct_image = cbct_image / np.max(cbct_image)

#     # Združitev v eno matriko z dimenzijami [2, 256, 192, 192]
#     combined_images = np.stack([fbct_image, cbct_image], axis=0)
#     # Pot za shranjevanje združenega para
#     output_path = os.path.join(data_directory, f"Thorax_pair_{i:03d}.pkl")

#     # Shranjevanje v .pkl datoteko
#     pksave(combined_images, output_path)

#     print(f"Shranjeno: {output_path}")

# print("Vsi pari so bili uspešno shranjeni.")

# print(fbct_image.shape, cbct_image.shape)


# class ThoraxDatasetS2S(Dataset):
#     def __init__(self, file_paths, transforms=None):
#         self.file_paths = file_paths
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):

#         with open(self.file_paths[idx], 'rb') as f:
#             data = pickle.load(f)

#         fixed_image = data[0]
#         moving_image = data[1]

#         fixed_image = fixed_image[None, ...]
#         moving_image = moving_image[None, ...]

#         if self.transforms:
#             fixed_image, moving_image = self.transforms([fixed_image, moving_image])

#         return torch.tensor(fixed_image, dtype=torch.float32), torch.tensor(moving_image, dtype=torch.float32)

class ThoraxDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        # Pridobivanje indeksov za par podatkov (fixed, moving)
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s

        # Pridobitev poti do datotek
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]

        # Naložitev podatkov
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)

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
        return len(self.paths) * (len(self.paths) - 1)
    
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
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)

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

