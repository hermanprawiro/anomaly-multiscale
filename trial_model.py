import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle

import torch
import torch.nn as nn
import torchvision

from dataset import ucsdped
from models import multiscale1_256
from utils import metrics
from scipy.misc import imresize
from sklearn.metrics import roc_auc_score, roc_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def tensor_to_img(inputs):
    return inputs.cpu() * 0.5 + 0.5

def main():
    root_dir_ped1 = "E:\\Datasets\\UCSD Anomaly\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1"
    root_dir_ped2 = "E:\\Datasets\\UCSD Anomaly\\UCSD_Anomaly_Dataset.v1p2\\UCSDped2"
    # root_dir_ped2 = "E:\\Datasets\\UCSD_Mini\\UCSDped2"
    root_dir_avenue = "E:\\Datasets\\Avenue"

    checkpoint_root = 'checkpoints'
    result_root = 'results'
    learning_rate = 2e-4
    train_batch_size = 2
    test_batch_size = 1
    start_epoch = 0
    max_epoch = 200

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ])

    n_channels = 1 # 1 for UCSD, 3 for others
    in_frames = 4
    out_frames = 4
    frame_length = in_frames

    train_set = ucsdped.UCSDPedDataset(root_dir=root_dir_ped2, train=True, frame_length=frame_length, frame_stride=1, transforms=transforms, dimension=3)
    test_set = ucsdped.UCSDPedDataset(root_dir=root_dir_ped2, train=False, frame_length=frame_length, frame_stride=out_frames, transforms=transforms, dimension=3)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=2)

    netE = multiscale1_256.Encoder(in_channels=n_channels).to(device)
    netD = multiscale1_256.Decoder(in_channels=384, out_channels=n_channels).to(device)

    netE.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.L1Loss().to(device)
    # criterion_gdl = losses.GDLoss3D(n_channels=n_channels, n_frames=out_frames, device=device)

    optimizer = torch.optim.Adam(list(netE.parameters()) + list(netD.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    checkpoint_prefix = 'ped2_multiscale1_256_L1Loss_{}in{}out_nogdl'.format(in_frames, out_frames)
    checkpoint_path = os.path.join(checkpoint_root, '{}.tar'.format(checkpoint_prefix))
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        netE.load_state_dict(checkpoint['state_dict']['encoder'])
        netD.load_state_dict(checkpoint['state_dict']['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('Checkpoint loaded, last epoch = {}'.format(start_epoch))
    
    result_path = os.path.join(result_root, checkpoint_prefix)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    models = (netE, netD)
    for epoch in range(start_epoch, max_epoch):
        train(train_loader, models, optimizer, criterion, None, epoch)
        save_model(models, optimizer, epoch, checkpoint_path)
        visualize(test_loader, models, result_path, epoch)
        test(test_loader, models)

    if not os.path.exists(os.path.join(result_path, 'full')):
        os.makedirs(os.path.join(result_path, 'full'))
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            # inputs, future = torch.split(inputs, 4, 1)

            outputs, maps, foreground, background = netD(netE(inputs))
            outputs = tensor_to_img(outputs)
            inputs = tensor_to_img(inputs)
            foreground = tensor_to_img(foreground)
            background = tensor_to_img(background)
            maps = maps.cpu()

            outputs = outputs.permute(0, 2, 1, 3, 4).view(-1, 1, 256, 256)
            inputs = inputs.view(-1, 1, 256, 256)
            maps = maps.permute(0, 2, 1, 3, 4).view(-1, 1, 256, 256)
            foreground = foreground.permute(0, 2, 1, 3, 4).view(-1, 1, 256, 256)
            background = background.view(-1, 1, 256, 256)
            diff = (inputs - outputs).abs()

            for j in range(4):
                torchvision.utils.save_image(torch.cat((inputs[j:j+1], outputs[j:j+1], diff[j:j+1], maps[j:j+1], foreground[j:j+1], background)), os.path.join(result_path, 'full', '{:05d}.png'.format(i*4 + j)), nrow=3, normalize=False)

def train(loader, models, optimizer, criterion, writer, epoch):
    batch_time = metrics.AverageMeter()
    data_time = metrics.AverageMeter()
    losses_R = metrics.AverageMeter()

    netE, netD = models

    netE.train()
    netD.train()

    total_iter = len(loader)
    end = time.time()
    for i, (inputs, _) in enumerate(loader):
        inputs = inputs.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs.size(0)
        # Reconstruction
        optimizer.zero_grad()
        
        outputs = netD(netE(inputs))[0]
        lossR = criterion(outputs, inputs)
        
        lossR.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        losses_R.update(lossR.item(), batch_size)

        # global_step = (epoch * total_iter) + i + 1
        # writer.add_scalar('train/loss', losses.val, global_step)

        if i % 10 == 0:
            print('Epoch {0} [{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss_R {lossR.val:.4f} ({lossR.avg:.4f})\t'
                .format(
                epoch + 1, i + 1, total_iter, 
                batch_time=batch_time, data_time=data_time, lossR=losses_R
                )
            )

def expand_results(raw_results):
    expanded_result = []
    previous_id = 0
    previous_errors = []
    for video_id, clip_error in raw_results:
        if video_id != previous_id:
            # previous_errors = imresize(np.expand_dims(previous_errors, axis=1), (len(previous_errors) * 4, 1), interp='bicubic', mode='F').flatten()
            expanded_result.append(previous_errors)
            previous_errors = []
        previous_errors.append(clip_error)
        previous_id = video_id
    # previous_errors = imresize(np.expand_dims(previous_errors, axis=1), (len(previous_errors) * 4, 1), interp='bicubic', mode='F').flatten()
    expanded_result.append(previous_errors)
    return expanded_result

def calculate_auc(results):
    gt_all = pickle.load(open(os.path.join(os.getcwd(), 'dataset', 'ucsd', 'ped2_gt.pkl'), 'rb'))
    unused_frames = 0

    y_true = []
    y_pred = []
    for result, gts in zip(results, gt_all):
        gt = np.ones_like(result)
        result = result - np.min(result)
        result = result / np.max(result)
        # result = 1 - result
        y_pred.append(result)
        # ax = plt.gca()
        for gt_id in gts:
            gt[gt_id[0] - 1 - unused_frames: gt_id[1] - unused_frames] = 0
            # ax.add_patch(matplotlib.patches.Rectangle((gt_id[0] - 1 - unused_frames, 0), gt_id[1] - gt_id[0], 1, facecolor='red', alpha=0.4))
        y_true.append(gt)
        # plt.plot(np.arange(result.shape[0]), result)
        # plt.ylim(0, 1)
        # plt.show()
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print(y_true.shape, y_pred.shape)
    return roc_auc_score(y_true, y_pred)

def calculate_psnr(prediction, targets):
    """
    Calculate reconstruction error by calculating L2-norm for (x, y, t)
    """
    assert prediction.size() == targets.size()
    # (batch_size, channel, time, height, width)
    error = torch.pow(targets - prediction, 2)
    error = error.mean((1, 3, 4)).view(-1) # MSE
    error = 10 * torch.log10(4 / error)
    # (batch_size, )
    return error

@torch.no_grad()
def test(loader, models):
    netE, netD = models

    netE.eval()
    netD.eval()

    total_iter = len(loader)
    errors_all = []
    targets_all = []

    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        
        outputs = netD(netE(inputs))[0]

        # error = metrics.calculate_l2dist(outputs, inputs)
        error = calculate_psnr(outputs, inputs)
        targets = targets.repeat(4)
        # print(error.shape, targets.shape)
        
        errors_all.append(error.cpu().numpy())
        targets_all.append(targets.cpu().numpy())

    errors_all = np.concatenate(errors_all)
    targets_all = np.concatenate(targets_all)

    results = list(zip(targets_all, errors_all))
    results = expand_results(results)
    auc = calculate_auc(results)
    print('> AUC = {:.5f}'.format(auc))

@torch.no_grad()
def visualize(loader, models, result_path, epoch):
    netE, netD = models
    netE.eval()
    netD.eval()

    testiter = iter(loader)
    inputs, _ = next(testiter)

    inputs = inputs.to(device)
    input_shape = inputs.shape
    
    outputs, mask, foreground, background = netD(netE(inputs))
    fg_mask = tensor_to_img(mask * foreground).squeeze(1)
    outputs = tensor_to_img(outputs.squeeze(1))
    mask = mask.cpu().squeeze(1)
    foreground = tensor_to_img(foreground.squeeze(1))
    background = tensor_to_img(background)

    if not os.path.exists(os.path.join(result_path, 'out')):
        os.makedirs(os.path.join(result_path, 'out'))
    if not os.path.exists(os.path.join(result_path, 'fg')):
        os.makedirs(os.path.join(result_path, 'fg'))
    if not os.path.exists(os.path.join(result_path, 'bg')):
        os.makedirs(os.path.join(result_path, 'bg'))
    if not os.path.exists(os.path.join(result_path, 'mask')):
        os.makedirs(os.path.join(result_path, 'mask'))
    if not os.path.exists(os.path.join(result_path, 'fg_mask')):
        os.makedirs(os.path.join(result_path, 'fg_mask'))

    torchvision.utils.save_image(outputs.view((-1, 1,) + input_shape[-2:]), os.path.join(result_path, 'out', '{:03d}.png'.format(epoch)), nrow=8, normalize=False)
    torchvision.utils.save_image(foreground.view((-1, 1,) + input_shape[-2:]), os.path.join(result_path, 'fg', '{:03d}.png'.format(epoch)), nrow=8, normalize=False)
    torchvision.utils.save_image(background, os.path.join(result_path, 'bg', '{:03d}.png'.format(epoch)), normalize=False)
    torchvision.utils.save_image(mask.view((-1, 1,) + input_shape[-2:]), os.path.join(result_path, 'mask', '{:03d}.png'.format(epoch)), nrow=8, normalize=False)
    torchvision.utils.save_image(fg_mask.view((-1, 1,) + input_shape[-2:]), os.path.join(result_path, 'fg_mask', '{:03d}.png'.format(epoch)), nrow=8, normalize=False)

def save_model(models, optimizer, epoch, checkpoint_path):
    netE, netD = models
    torch.save({
        'state_dict': {
            'encoder': netE.state_dict(),
            'decoder': netD.state_dict(),
        },
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, checkpoint_path)

if __name__ == "__main__":
    main()