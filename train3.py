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
import argparse

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
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

def main(train_dir, val_dir, lr, weights, max_epoch, batch_size, save_dir):
    # batch_size = 1

    # train_dir = 'Release_pkl/Resized_normalized_imagesTr/Train/'
    # val_dir = 'Release_pkl/Resized_normalized_imagesTr/Val/'
    
    # weights = [1, 1]  # loss weights
    # lr = 0.0001
    head_dim = 6
    num_heads = [8,4,2,1,1]
    channels = 8
    # save_dir = 'New_Post_55_epoh_ModeTv2/'

    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)

    sys.stdout = Logger('logs/' + save_dir)

    f = open(os.path.join('logs/'+save_dir, 'losses and dice' + ".txt"), "a")

    epoch_start = 0
    # max_epoch = 50
    img_size = (170, 128, 128)
    cont_training = False

    '''
    Initialize model
    '''
    model = ModeTv2_model(img_size, head_dim=head_dim, num_heads=num_heads,channels=channels//2, scale=1)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()


    '''
    If continue from previous training
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

    val_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.float32)),
                                       ])

    train_set = datasets.ThoraxDatasetSequentialPost2(train_dir, transforms=train_composed)
    val_set = datasets.ThoraxDatasetSequentialPost2(val_dir, transforms=val_composed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]

    best_dsc = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        
        for idx, data in enumerate(train_loader):
            model.train()

            cbct1, fbct = [d.cuda() for d in data]

            output = model(fbct, cbct1)

            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], cbct1) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss

            loss_all.update(loss.item(), cbct1.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')

        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                cbct1, fbct = [d.cuda() for d in data]
                
                fbct_def, flow = model(fbct, cbct1)

                dsc = utils.dice_val_VOI(fbct_def, cbct1)
                eval_dsc.update(dsc.item(), cbct1.size(0))
                print(epoch, ':',eval_dsc.avg)

        best_dsc = max(eval_dsc.avg, best_dsc)
        print(eval_dsc.avg, file=f)

        test_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                cbct1, fbct = [d.cuda() for d in data]
                fbct_def, flow = model(fbct, cbct1)

                fig = comput_fig(fbct_def)
                plt.savefig(f'results/test_{idx}_fbct_def.png')

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

    parser = argparse.ArgumentParser(description="Train ModeTv2 Model")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the training data directory")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to the validation data directory")
    parser.add_argument('--lr', type=float, required=True, help="Learning rate for training")
    parser.add_argument('--weights', type=float, nargs=2, required=True, help="Loss weights for the task (two values required)")
    parser.add_argument('--max_epoch', type=int, required=True, help="Maximum number of training epochs")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--save_dir', type=str, required=True, help="Path where to save results")
    args = parser.parse_args()

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
    main(args.train_dir, args.val_dir, args.lr, args.weights, args.max_epoch, args.batch_size, args.save_dir)


# Usage:
    # python train3.py \
    # --train_dir Release_pkl/Resized_normalized_imagesTr/Train/ \
    # --val_dir Release_pkl/Resized_normalized_imagesTr/Val/ \
    # --lr 0.001 \
    # --weights 1 1 \
    # --max_epoch 50 \
    # --batch_size 1 \
    # --save_dir ModeTv2/