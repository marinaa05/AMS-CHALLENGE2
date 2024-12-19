import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from skimage.transform import resize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
from models_cuda import ModeTv2_model
import random
import nibabel as nib
from skimage.transform import resize

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

same_seeds(24)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def comput_fig(img, stdy_idx):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    #return fig
    # Ustvari mapo "slike1", če še ne obstaja
    output_dir = "pre_release_slike22"
    os.makedirs(output_dir, exist_ok=True)

    # Določite pot do shranjevanja slike
    output_file = os.path.join(output_dir, f"output_figure_{stdy_idx}.png")

    # Shrani sliko
    plt.savefig(output_file)  # Shranjevanje slike
    plt.close()  # Zapri sliko
    print(f"Slika shranjena: {output_file}")

def visualize_registration(fixed, moving_image, moving_transformed, deformation_field, stdy_idx, slice_idx=None):
    """
    Vizualizira fiksno sliko, transformirano premikajočo sliko in polje deformacije.
    :param fixed: Fiksna slika (3D numpy array ali torch tensor).
    :param moving_transformed: Transformirana premikajoča slika (3D numpy array ali torch tensor).
    :param deformation_field: Deformacijsko polje (3D numpy array, oblika [3, D, H, W]).
    :param slice_idx: Indeks reza za prikaz (če None, vzame sredinski rez).
    """
    if isinstance(fixed, torch.Tensor):
        fixed = fixed.detach().cpu().numpy()
    if isinstance(moving_image, torch.Tensor):
        moving_image = moving_image.detach().cpu().numpy()
    if isinstance(moving_transformed, torch.Tensor):
        moving_transformed = moving_transformed.detach().cpu().numpy()
    if isinstance(deformation_field, torch.Tensor):
        deformation_field = deformation_field.detach().cpu().numpy()

    # Izračun sredinskega reza, če ni podan
    if slice_idx is None:
        slice_idx = [fixed.shape[0] // 2, fixed.shape[1] // 2, fixed.shape[2] // 2]  # [X, Y, Z]

    axes = ['X', 'Y', 'Z']
    slices = [slice_idx[0], slice_idx[1], slice_idx[2]]

    output_dir = "train_300_final_release_slike"
    os.makedirs(output_dir, exist_ok=True)

    # Prikaz za vse osi (x, y, z)
    for i, axis in enumerate(axes):
        plt.figure(figsize=(16, 8))
        plt.suptitle(f"Visualization - Axis: {axis}")
        
        # Fiksna slika
        plt.subplot(3, 4, 1)
        plt.imshow(np.take(fixed, slices[i], axis=i), cmap='gray')
        plt.title(f"Fixed Image ({axis}-axis)")
        
        plt.subplot(3, 4, 2)
        plt.imshow(np.take(moving_image, slices[i], axis=i), cmap='gray')
        plt.title(f"Moving Image ({axis}-axis)")

        plt.subplot(3, 4, 3)
        plt.imshow(np.take(moving_transformed, slices[i], axis=i), cmap='gray')
        plt.title(f"Transformed Moving Image ({axis}-axis)")
        
        # Deformacijsko polje (X, Y, Z komponente)
        for j in range(3):
            plt.subplot(3, 4, 4 + j)
            plt.imshow(np.take(deformation_field[j], slices[i], axis=i), cmap='jet')
            plt.title(f"Deformation Field ({axis}-axis, Dir: {['X', 'Y', 'Z'][j]})")
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"output_figure_{stdy_idx}_{axis}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Slika shranjena: {output_file}")

# Vizualizacija deformacijskega polja
def visualize_and_save_flow(flow, patient_idx, output_dir="def_polje"):
    """
    Vizualizira deformacijsko polje in shrani slike v mapo.
    
    Args:
        flow (numpy array): Deformacijsko polje oblike [3, D, H, W].
        patient_idx (int): Indeks trenutnega pacienta.
        output_dir (str): Izhodna mapa za shranjevanje slik.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ustvari mapo, če ne obstaja

    axes = ['X', 'Y', 'Z']
    for i, axis in enumerate(axes):
        plt.figure(figsize=(8, 8))
        plt.imshow(flow[i, flow.shape[1] // 2, :, :], cmap='jet')  # Sredinski rez
        plt.title(f"Flow {axis}-axis (Patient {patient_idx})")
        plt.colorbar()
        plt.tight_layout()

        # Shrani sliko
        output_path = os.path.join(output_dir, f"patient_{patient_idx}_flow_{axis}.png")
        plt.savefig(output_path)
        plt.close()  # Zapri sliko, da sprosti pomnilnik
        print(f"Slika shranjena: {output_path}")

def save_nifti(data, output_path, affine=np.eye(4)):
    """
    Shrani podatke v NIfTI format (.nii.gz).
    
    Args:
        data (numpy array): 3D ali 4D podatki za shranjevanje.
        output_path (str): Pot za shranjevanje datoteke.
        affine (numpy array): Afina matrika za NIfTI (privzeto enotska matrika).
    """
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_path)
    print(f"NIfTI datoteka shranjena: {output_path}")

def resize_volume(volume, output_shape):
    """
    Spremeni velikost volumna na zahtevano obliko.
    
    Args:
        volume (numpy array): Volumen za spremembo velikosti.
        output_shape (tuple): Ciljna oblika (npr. (256, 192, 192)).
        
    Returns:
        numpy array: Spremenjen volumen.
    """
    return resize(volume, output_shape, mode='constant', anti_aliasing=True)

def main():

    stdy_idx = 0
    output_dir = "dof_final"
    os.makedirs(output_dir, exist_ok=True)

    val_dir = 'Release_pkl/Resized_normalized_imagesTr/Val/'
    weights = [1, 1]  # loss weights
    lr = 0.0001
    head_dim = 6
    num_heads = [8,4,2,1,1]
    model_folder = '55_epoh_post_ModeTv2_cuda_nh({}{}{}{}{})_hd_{}_ncc_{}_reg_{}_lr_{}_54r/'.format(*num_heads, head_dim,weights[0], weights[1], lr)
    model_idx = -1
    model_dir = 'experiments/' + model_folder

    img_size = (170, 128, 128)
    target_size = (256, 192, 192)  # Ciljna velikost

    # Inicializacija modela
    model = ModeTv2_model(img_size)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))

    model.load_state_dict(best_model)
    model.cuda()

    # Izpis uteži iz modela
    # print("\nModel Weights:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.data}")

    # Inicializacija registracijskega modela
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    test_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.float32)),
                                        ])
    
    test_set = datasets.ThoraxDatasetSequential(val_dir, transforms=test_composed)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    eval_det = utils.AverageMeter()

    with torch.no_grad():
        stdy_idx = 0
        for stdy_idx, data in enumerate(test_loader):
            print(f"Processing patient {stdy_idx + 1}/{len(test_loader.dataset)}")
            model.eval()

            # Pridobi pare FBCT in CBCT
            fbct, cbct1 = [d.cuda() for d in data]

            # Napovej deformacijsko polje in transformirano sliko
            fbct_def, flow = model(fbct, cbct1)
            flow = flow / flow.abs().max()   # Normalizacija na interval [-1, 1]
    
            tar = fbct.detach().cpu().numpy()[0, 0, :, :, :]

            print(f"Flow shape: {flow.shape}")
            print(f"Flow stats: min={flow.min().item()}, max={flow.max().item()}, mean={flow.mean().item()}")
            

            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            print(f"Flow stats for patient {stdy_idx + 1}: mean={flow.mean().item()}, max={flow.max().item()}, min={flow.min().item()}")

            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), fbct.size(0))

            # Klic funkcije visualize_registration z novimi podatki
            visualize_registration(
                fixed=fbct[0, 0],  # Ciljna slika (CBCT)
                moving_image=cbct1[0, 0],  # Premaknjena slika (FBCT)
                moving_transformed=fbct_def[0, 0],  # Transformirana premaknjena slika
                deformation_field=flow[0],  # Deformacijsko polje
                stdy_idx=stdy_idx
            )
            
            flow_numpy = flow[0].detach().cpu().numpy()  # Oblika: [3, D, H, W]

            # Vizualiziraj in shrani slike deformacijskega polja
            # visualize_and_save_flow(flow_numpy, stdy_idx)

            # Pretvorba v numpy in sprememba velikosti
            fbct_np = fbct[0, 0].detach().cpu().numpy()
            cbct1_np = cbct1[0, 0].detach().cpu().numpy()
            fbct_def_np = fbct_def[0, 0].detach().cpu().numpy()
            flow_np = flow[0].detach().cpu().numpy()

            fbct_resized = resize_volume(fbct_np, target_size)
            cbct1_resized = resize_volume(cbct1_np, target_size)
            fbct_def_resized = resize_volume(fbct_def_np, target_size)
            flow_resized = np.stack([
                resize_volume(flow_np[i], target_size) for i in range(flow_np.shape[0])
            ], axis=-1)  # Oblika: (256, 192, 192, 3)

            # Shranjevanje v NIfTI formatu
            save_nifti(fbct_resized, os.path.join(output_dir, f"patient_{stdy_idx}_fixed.nii.gz"))
            save_nifti(cbct1_resized, os.path.join(output_dir, f"patient_{stdy_idx}_moving.nii.gz"))
            save_nifti(fbct_def_resized, os.path.join(output_dir, f"patient_{stdy_idx}_transformed.nii.gz"))
            save_nifti(flow_resized, os.path.join(output_dir, f"patient_{stdy_idx}_flow.nii.gz"))

            print(f"Deformacijsko polje za pacienta {stdy_idx + 1} uspešno shranjeno.")



            stdy_idx += 1


        print('Deformed determinant Jacobian: {:.3f} +- {:.3f}'.format(eval_det.avg, eval_det.std))

        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

        print(f"Output shape: {flow.shape}")
        print(f"Flow stats: mean={flow.mean().item()}, max={flow.max().item()}, min={flow.min().item()}")




if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1  # 0 ali 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()