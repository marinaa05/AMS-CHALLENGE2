from data import datasets
import os, utils
from torch.utils.data import DataLoader
import numpy as np
from data import datasets, trans
from torchvision import transforms

# Testno deformacijsko polje
D, H, W = 10, 10, 10
disp_test = np.zeros((D, H, W, 3))
disp_test[..., 0] = np.linspace(-1, 1, D)[:, None, None]
disp_test[..., 1] = np.linspace(-1, 1, H)[None, :, None]
disp_test[..., 2] = np.linspace(-1, 1, W)[None, None, :]

# Izračun Jacobian determinant
jac_det_test = utils.jacobian_determinant_vxm(disp_test)
print(f"Jacobian determinant test stats: min={jac_det_test.min()}, max={jac_det_test.max()}, mean={jac_det_test.mean()}")



