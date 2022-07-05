from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_eval_set
from functools import reduce
import scipy.io as sio
import time
import cv2
from collections import OrderedDict
from glob import glob
import numpy as np
import math
from PIL import Image

from dsrnet import Net as dsrnet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# Calculate
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--gpus', default=2, type=float, help='number of gpu')
parser.add_argument('--gpu', type=int, default=0, help='the number of using gpu')
# Networks
parser.add_argument('--model_type', type=str, default='dsrnet')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--weights', default="./weights/train_depthCanny_x8/ubuntudsrnet_epoch_119.pth", help='sr pretrained base model')
# Data
parser.add_argument('--input_dir', type=str, default='../MSG-TestData/')
parser.add_argument('--test_dataset', type=str, default='test_depth_x8/')
parser.add_argument('--test_rgb', type=str, default='test_msg_gt_c_v2/')
parser.add_argument('--output_dir', default='Results_x8/', help='Location to save the prediction')

parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

def eval():
    model.eval()
    torch.set_grad_enabled(False)
    
    with torch.no_grad():
        for batch in testing_data_loader:
            input,input_rgb, name = Variable(batch[0]),Variable(batch[1]), batch[2]
            
            if cuda:
                input = input.cuda(gpus_list[opt.gpu])
                input_rgb = input_rgb.cuda(gpus_list[opt.gpu])
    
            t0 = time.time()

            "Output"
            result = model(input_rgb,input)
            prediction = result[2]
            
            for i in range(3,10):
                print(result[i],end=',')
            print("")
            
            t1 = time.time()
            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            save_img(prediction.cpu().data, name[0])

def save_img(img, img_name):

    save_img = img.squeeze().clamp(0, 1).numpy()

    save_dir=os.path.join(opt.output_dir,opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    cv2.imwrite(save_fn,save_img*255)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)


print('===> Loading datasets')
test_set = get_eval_set(os.path.join(opt.input_dir,opt.test_dataset),os.path.join(opt.input_dir,opt.test_rgb))
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.model_type == 'DDCE':
    model = DDCE(num_channels=1, base_filter=64,  feat = 256, num_stages=4, scale_factor=opt.upscale_factor)
elif opt.model_type == 'dsrnet':
    model = dsrnet(num_channels=1, base_filter=64,  feat = 256, num_stages=4, scale_factor=opt.upscale_factor)
###
flag=1 # for loading SR model

"Single GPU testing"
if cuda:
    model = model.cuda(gpus_list[opt.gpu])

if os.path.exists(opt.weights):
    model.load_state_dict(torch.load(opt.weights, map_location=lambda storage, loc: storage))
    flag=0
    print('<-------------- Pre-trained SR model is loaded. -------------->')

"Multi-GPUs Testing"
# if cuda:
#     model = nn.DataParallel(model).cuda(gpus_list[opt.gpu]) #use parallel

# if os.path.exists(opt.weights):
#     #model= torch.load(opt.model, map_location=lambda storage, loc: storage)
#     # print("11111111111111111111")
#     model.load_state_dict(torch.load(opt.weights, map_location=lambda storage, loc: storage))
#     flag=0
#     print('<--------------Pre-trained SR model is loaded.-------------->')

# if flag == 1:
#     print('!-------------- Cannot load pre-trained model! --------------!')


##Eval Start!!!!
print('<-------------- Writing results to {} -------------->'.format(opt.output_dir))
eval()