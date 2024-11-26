import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
from models_cuda import ModeTv2_model
import random
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


def visualize_registration(fixed_image, moving_image, moving_transformed, deformation_field, idx, slice_idx=None):
    """
    Funkcija za prikaz fiksne slike, premikajoče slike (ang. moving image), transformirane premikajoče slike in deformacijskega polja.
    Parametri:
        fixed_image: Fiksna slika (3D numpy array ali torch tensor)
        moving_image: Premikajoča slika (3D numpy array ali torch tensor)
        moving_transformed: Transformirana premikajoča slika (3D numpy array ali torch tensor)
        deformation_field: Deformacijsko polje (3D numpy array, oblika [3, D, H, W])
        idx: Indeks za označevanje slik
        slice_idx: Indeks reza za prikaz (če None, vzame sredinski rez)
    """
    # Prevetvorba vhodih podatkov v numpy array in zagotovimo, da so primerni za prikaz z matplotlib
    if isinstance(fixed_image, torch.Tensor):
        fixed_image = fixed_image.detach().cpu().numpy()
    if isinstance(moving_image, torch.Tensor):
        moving_image = moving_image.detach().cpu().numpy()
    if isinstance(moving_transformed, torch.Tensor):
        moving_transformed = moving_transformed.detach().cpu().numpy()
    if isinstance(deformation_field, torch.Tensor):
        deformation_field = deformation_field.detach().cpu().numpy()

    # Izračun sredinskega reza, če ni podan
    if slice_idx is None:
        slice_idx = fixed_image.shape[2] // 2

    # Ustvarimo mapo (če ta ne obstaja), kjer bomo shranjevali slike:
    output_dir = "slike2"
    os.makedirs(output_dir, exist_ok=True)

    # Prikaz fiksne slike
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(fixed_image[:, :, slice_idx], cmap='gray')
    plt.title("Fixed Image")
    #plt.axis('off')

    # Prikaz premikajoče slike
    plt.subplot(1, 4, 2)
    plt.imshow(moving_image[:, :, slice_idx], cmap='gray')
    plt.title("Moving Image (Pre)")

    # Prikaz transformirane premikajoče slike
    plt.subplot(1, 4, 3)
    plt.imshow(moving_transformed[:, :, slice_idx], cmap='gray')
    plt.title("Transformed Moving Image")
    #plt.axis('off')

    # Prikaz deformacijskega polja (v eni smeri, npr. X)
    plt.subplot(1, 4, 4)
    plt.imshow(deformation_field[0, :, :, slice_idx], cmap='jet')  # X-smer deformacije
    plt.title("Deformation Field (X-direction)")
    #plt.axis('off')

    plt.tight_layout()
    #output_file = f"output_figure_{idx}.png"  # Shrani za vsako iteracijo
    output_file = os.path.join(output_dir, f"output_figure_{idx}.png")  # Pot do slike
    plt.savefig(output_file)  # Shranjevanje slike
    plt.close()
    print(f"Slika shranjena: {output_file}")

def main():
    '''
    Evaluacija modela za registracijo slik.
    '''
    val_dir = '/LPBA_path/Val/'  # validacijski podatki
    weights = [1, 1]  # loss weights
    lr = 0.0001  # učna hitrost
    head_dim = 6
    num_heads = [8,4,2,1,1]  # št. glav v vsakem sloju modela
    # Pot do mape, kjer se nahaja model (rezultat train.py):
    model_folder = 'ModeTv2_cuda_nh({}{}{}{}{})_hd_{}_ncc_{}_reg_{}_lr_{}_54r/'.format(*num_heads, head_dim,weights[0], weights[1], lr)
    model_idx = -1  # izbere model, ki ga bomo naložili (zadnji)
    model_dir = 'experiments/' + model_folder

    img_size = (160, 192, 160)
    model = ModeTv2_model(img_size)  # inicializacija modela za registracijo
    # Vse datoteke v model_dir razvrstimo po številkah (min do max št.), izberemo tisto, ki je na indeksu model_idx,
    # nato naloži ta model z torch.load() in iz njega izvleče vrednost, ki je zapisana pod ključem 'state_dict'
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)  # naloži uteži modela
    model.cuda()  # prenese model na gpu (hitrejše računanje)
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    with torch.no_grad():
        idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_def, flow = model(x,y)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])

            visualize_registration(
                fixed_image=y[0, 0], 
                moving_image=x[0, 0], 
                moving_transformed=x_def[0, 0], 
                deformation_field=flow[0], 
                idx=idx
            )
            idx += 1

            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            
        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
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