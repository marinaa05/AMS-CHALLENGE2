import glob
import os, losses, utils
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
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
    torch.backends.cudnn.benchmark=True

same_seeds(24)
class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

GPU_iden = 0

def main():
    batch_size = 1
    train_dir = 'LPBA_data\Train'
    val_dir = 'LPBA_data\Val'
    weights = [1, 1]  # loss weights
    lr = 0.0001  # learning rate
    # Parametri, ki določajo arhitekturo pozornosti in št. kanalov:
    head_dim = 6
    num_heads = [8,4,2,1,1]
    channels = 8
    #save_dir = 'ModeTv2_cuda_nh({}{}{}{}{})_hd_{}_c_{}_ncc_{}_reg_{}_lr_{}_54r/'.format(*num_heads, head_dim,channels,weights[0], weights[1], lr)
    # Mapa, kamor bomo shranjevali model:
    save_dir = 'ModeTv2_cuda_nh({}{}{}{}{})_hd_{}_ncc_{}_reg_{}_lr_{}_54r/'.format(*num_heads, head_dim,weights[0], weights[1], lr)

    # Če mapa za model ne obstaja, ustvarimo novo:
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    f = open(os.path.join('logs/'+save_dir, 'losses and dice' + ".txt"), "a")

    epoch_start = 0
    max_epoch = 30
    img_size = (160, 192, 160)
    cont_training = False

    '''
    Initialize model
    '''
    model = ModeTv2_model(img_size, head_dim=head_dim, num_heads=num_heads,channels=channels//2, scale=1)
    model.cuda()  # premik na GPU (hitrejše računanje)

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()


    '''
    If continue from previous training (if cont_training=True)
    '''
    if cont_training:
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        model.load_state_dict(best_model)
        print(model_dir + natsorted(os.listdir(model_dir))[-1])
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(),  # normalizacija
                                       trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.LPBABrainDatasetS2S(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    val_set = datasets.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    # Branje train data:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Branje validacijskih podatkov:
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # Optimizator z AMSGrad za stabilizacijo učenja:
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC_vxm()  # ocena podobnosti slik
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]  # Grad3d - preverja kao zvezve/gladke so spremembe v def. polju
    
    best_dsc = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()  # model nastavljen na train način
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            x = data[0].cuda()  # moving imgs
            y = data[1].cuda()  # fixed imgs

            output = model(x,y)  # izračun napovedi outputa

            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):  #iteracija skozi funkcije izgube
                curr_loss = loss_function(output[n], y) * weights[n]  # trenutna izguba
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # Posodobitev uteži:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')

        '''
        Validation (po zaključku vsake epohe)
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]

                output = model(x,y)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])

                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(epoch, ':',eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        print(eval_dsc.avg, file=f)

        # Po vsaki epohi se model shrani (max št. shranjenih modelov=8 - da ne zavzamejo prevec GB):
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        loss_all.reset()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 160)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''

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
