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

    output_dir = "300_final_release_slike"
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


def main():

    stdy_idx = 0

    val_dir = 'Release_pkl/Resized_merged_imagesTr/Post_therapy/Val/'
    weights = [1, 1]  # loss weights
    lr = 0.0001
    head_dim = 6
    num_heads = [8,4,2,1,1]
    model_folder = '50_epoh_post_ModeTv2_cuda_nh({}{}{}{}{})_hd_{}_ncc_{}_reg_{}_lr_{}_54r/'.format(*num_heads, head_dim,weights[0], weights[1], lr)
    model_idx = -1
    model_dir = 'experiments/' + model_folder

    # img_size = (160, 192, 160)
    img_size = (170, 128, 128)

    # Inicializacija modela
    model = ModeTv2_model(img_size)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))

    model.load_state_dict(best_model)

    # Izpis uteži iz modela
    # print("\nModel Weights:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.data}")


    model.cuda()

    # Inicializacija registracijskega modela
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    test_composed = transforms.Compose([#trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.float32)),
                                        ])
    
    # test_set = datasets.ThoraxInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=test_composed)
    test_set = datasets.ThoraxDatasetPairwise(glob.glob(os.path.join(val_dir, '*.pkl')), transforms=test_composed)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    # eval_dsc_def = AverageMeter()
    # eval_dsc_raw = AverageMeter()
    # eval_det = AverageMeter()
    eval_det = utils.AverageMeter()

    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            # data = [t.cuda() for t in data]
            # x = data[0]
            # y = data[1]
            # x_seg = data[2]
            # y_seg = data[3]

            # Pridobi pare FBCT in CBCT
            fbct, cbct = [t.cuda() for t in data]

            # Napovej deformacijsko polje in transformirano sliko
            fbct_def, flow = model(fbct, cbct)
            
            # x_def, flow = model(x,y)

            # Originalna velikost
            # original_shape = (256, 192, 192)

            # # Resize deformacijsko polje (flow)
            # resized_flow = np.zeros((3, *original_shape), dtype=flow.cpu().numpy().dtype)
            # for i in range(3):  # X, Y, Z komponente deformacijskega polja
            #     resized_flow[i] = resize(flow[0, i].cpu().numpy(), original_shape, mode='reflect', anti_aliasing=True)

            # # Resize transformirane slike (x_def)
            # resized_x_def = resize(x_def[0, 0].cpu().numpy(), original_shape, mode='reflect', anti_aliasing=True)

            # # Resize moving image
            # resized_moving = resize(x[0, 0].cpu().numpy(), original_shape, mode='reflect', anti_aliasing=True)

            # # Resize fixed image
            # resized_fixed = resize(y[0, 0].cpu().numpy(), original_shape, mode='reflect', anti_aliasing=True)

            # # Pretvori nazaj v torch tensorje, če potrebuješ
            # resized_flow_tensor = torch.tensor(resized_flow, device=flow.device)
            # resized_x_def_tensor = torch.tensor(resized_x_def, device=x_def.device)
            # resized_moving_tensor = torch.tensor(resized_moving, device=x.device)
            # resized_fixed_tensor = torch.tensor(resized_fixed, device=y.device)

            # print("Moving, fixed, transformirane slike in deformacijsko polje so preoblikovani nazaj na originalno velikost.")
            # print(f"Polje deformacij: {resized_flow_tensor.shape}, "
            #       f"Transformirana slika: {resized_x_def_tensor.shape}, "
            #       f"Moving slika: {resized_moving_tensor.shape}, "
            #       f"Fixed slika: {resized_fixed_tensor.shape}")
            
            # def_out = reg_model([x_seg.cuda().float(), flow.cuda()])

            #fixed_image = y[0, 0]  # Prva slika iz batch-a, kanal 0
            #moving_image_transformed = x_def[0, 0]  # Transformirana slika
            #deformation_field = flow[0]  # Deformacijsko polje (oblika [3, D, H, W])

            # Klic funkcije za vizualizacijo
            #visualize_registration(fixed_image, moving_image_transformed, deformation_field, stdy_idx)

            # Originalna velikost
            original_shape = (256, 192, 192)

            # Preoblikuj fixed sliko
            # resized_fixed = resize(y[0, 0].detach().cpu().numpy(), original_shape, mode='constant', anti_aliasing=True)

            # # Preoblikuj moving sliko
            # resized_moving = resize(x[0, 0].detach().cpu().numpy(), original_shape, mode='constant', anti_aliasing=True)

            # # Preoblikuj transformirano moving sliko
            # resized_transformed = resize(x_def[0, 0].detach().cpu().numpy(), original_shape, mode='constant', anti_aliasing=True)

            # # Preoblikuj deformacijsko polje (vsaka komponenta posebej)
            # resized_flow = np.zeros((3, *original_shape), dtype=flow.detach().cpu().numpy().dtype)
            # for i in range(3):  # X, Y, Z komponente
            #     resized_flow[i] = resize(flow[0, i].detach().cpu().numpy(), original_shape, mode='constant', anti_aliasing=True)

            # Klic funkcije visualize_registration z novimi podatki
            visualize_registration(
                fixed=cbct[0, 0],  # Ciljna slika (CBCT)
                moving_image=fbct[0, 0],  # Premaknjena slika (FBCT)
                moving_transformed=fbct_def[0, 0],  # Transformirana premaknjena slika
                deformation_field=flow[0],  # Deformacijsko polje
                stdy_idx=stdy_idx
            )
            
            # Prikaz slik - na 2 način:
            # visualize_registration(
            #    fixed=y[0, 0], 
            #    moving_image=x[0, 0], 
            #    moving_transformed=x_def[0, 0], 
            #    deformation_field=flow[0],
            #    stdy_idx=stdy_idx
            # )

            # comput_fig(x_def, stdy_idx)
            
            stdy_idx += 1

            # tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            
            # jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            # dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            # dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())

            tar = cbct.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), fbct.size(0))

            #print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))

        # print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
        #                                                                            eval_dsc_def.std,
        #                                                                            eval_dsc_raw.avg,
        #                                                                            eval_dsc_raw.std))

        print('Deformed determinant Jacobian: {:.3f} +- {:.3f}'.format(eval_det.avg, eval_det.std))

        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

        # print(f"Manjše polje: {flow.shape}, Resizano polje: {resized_flow.shape}")


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