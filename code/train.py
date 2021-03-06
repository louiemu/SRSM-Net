from __future__ import print_function
import argparse
from math import log10
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import pdb
import socket
import time
import train_loss

from dsrnet import Net as dsrnet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# Calculation
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--gpus', default=2, type=float, help='number of gpu')
parser.add_argument('--gpu', type=int, default=0, help='the number of using gpu')
# Networks
parser.add_argument('--model_type', type=str, default='dsrnet')
parser.add_argument('--prefix', default='_x8_', help='description of the model for weights')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--pretrained_sr', default="pretrain.pth", help='sr pretrained base model')
# Data (input-output-train)
parser.add_argument('--data_dir', type=str, default='../MSG-TrainData/', help='root of data')
parser.add_argument('--train_dataset', type=str, default='train_depthCanny_x8/', help='1.down-sampled images for input  2.also for the dataset type in DatasetFolder')
parser.add_argument('--depth_train_dataset', type=str, default='train_depthCanny_all_256/', help='GT depth images for training')
parser.add_argument('--edge_train_dataset', type=str, default='train_edgeCannyRaw_all_256/', help='GT edge images for training')
parser.add_argument('--rgb_train_dataset', type=str, default='train_rgb_all_256', help='GT rgb images for training')

parser.add_argument('--test_dir', type=str, default='../MSG-TestData/test_depth_x8/', help='test data')

parser.add_argument('--patch_size', type=int, default=32, help='Size of cropped LR image')
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')

parser.add_argument('--threads', type=int, default=0, help='number of extra threads for data loader to use')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')

parser.add_argument('--nEpochs', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--hyper_w', type=float, default=0.001, help='the weight parameters.')

opt = parser.parse_args()
print(opt)

"Obtain device info"
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())

def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        gt_edge, input_rgb, input, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
        if cuda:
            gt_edge = gt_edge.cuda(gpus_list[opt.gpu])
            input_rgb = input_rgb.cuda(gpus_list[opt.gpu])
            input = input.cuda(gpus_list[opt.gpu])
            target = target.cuda(gpus_list[opt.gpu])

        optimizer.zero_grad()

        t0 = time.time()
        result = model(input_rgb, input)

        struct_weighted_loss = train_loss.struct_weighted_loss_edge(result[0], result[1], opt.hyper_w)

        depth_struct_loss = bce_loss(result[1], gt_edge)
        
        depth_loss = criterion(result[2], target)

        loss = struct_weighted_loss + depth_struct_loss + depth_loss

        t1 = time.time()

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        avg_loss = epoch_loss / len(training_data_loader)
        loss_list.append(avg_loss)

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.item(), (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    path_1 = opt.save_folder+opt.prefix+opt.train_dataset
    model_out_path = opt.save_folder+opt.prefix+opt.train_dataset+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    if not os.path.exists(path_1):
        os.makedirs(path_1)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


"Check cuda"
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


"Choose Seed"
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)


"Loading Dataset"
print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.train_dataset, opt.depth_train_dataset, opt.edge_train_dataset,opt.rgb_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
# test_set = get_eval_set(opt.test_dir)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

"Building Net"
print('===> Building model ', opt.model_type)
if opt.model_type == 'dsrnet':
    model = dsrnet(num_channels=1, base_filter=64,  feat = 256, num_stages=8, scale_factor=opt.upscale_factor)

"DataParallel"
# model = torch.nn.DataParallel(model, device_ids=gpus_list)

criterion = nn.L1Loss()
bce_loss = nn.BCELoss(size_average=True)
criterion_ce = nn.CrossEntropyLoss()
print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')


"Loading Pre-trained weights"
if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        pretrained_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("*************************Pre-trained SR weights:{} is loaded.************************************".format(opt.pretrained_sr))


"To cuda"
if cuda:
    model = model.cuda(gpus_list[opt.gpu])
    criterion = criterion.cuda(gpus_list[opt.gpu])

"Optimizer"
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)


"Training"
loss_list = []
mad_list = []
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)

    if (epoch+1) % 2 == 0:
        plt.plot(loss_list, linewidth=5)
        plt.title('Loss Table', fontsize=24)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        path_fig = opt.save_folder+opt.prefix+opt.train_dataset
        if not os.path.exists(path_fig):
            os.makedirs(path_fig)
        plt.savefig('{}log_loss_lr1e-4_batch8.png'.format(path_fig))

    if (epoch+1) == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch+1) == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)