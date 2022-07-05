import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageFilter
import random
from random import randrange


"Return all filenames of images in folder"
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


"Define the image loading"
def load_img(filepath):
    # img = Image.open(filepath).convert('RGB')
    ##############
    img = Image.open(filepath)
    # img = Image.open(filepath)
    # y, _, _ = img.split()
    return img

"loading high-frequncy part of the image "
# def load_hf(filepath):
#     # img = Image.open(filepath).convert('RGB')
#     ##############
#     img = Image.open(filepath)
#     hf = img.filter(ImageFilter.FIND_EDGES)
#     # img = Image.open(filepath)
#     # y, _, _ = img.split()
#     return hf

# def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
#     (c, ih, iw) = img_in.shape
#     ####print('input:', ih, iw)
#     (th, tw) = (scale * ih, scale * iw)

#     patch_mult = scale #if len(scale) > 1 else 1
#     tp = patch_mult * patch_size
#     ip = tp // scale

#     if ix == -1:
#         ix = random.randrange(0, iw - ip + 1)
#     if iy == -1:
#         iy = random.randrange(0, ih - ip + 1)

#     (tx, ty) = (scale * ix, scale * iy)
#     img_in = img_in[:, iy:iy + ip, ix:ix + ip]
#     print('get_patch', img_tar.size(), ty, ty+tp, tx, tx+tp)
#     img_tar = img_tar[:, ty:ty + tp, tx:tx + tp]
#     info_patch = {
#         'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

#     ####print('after', img_tar.size())

#     return img_in, img_tar, info_patch


"Define the data augmentation"
def augment(img_in, img_edge, img_rgb, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    # print(img_edge.size(), 'eeeeeeeeeeeeeeeeeeee')
    if random.random() < 0.5 and flip_h:
        ####print('<-------------->', img_tar.size())
        img_in = torch.from_numpy(img_in.numpy()[:, :, ::-1].copy())
        img_edge = torch.from_numpy(img_edge.numpy()[:, :, ::-1].copy())
        img_rgb = torch.from_numpy(img_rgb.numpy()[:, :, ::-1].copy())
        img_tar = torch.from_numpy(img_tar.numpy()[:, :, ::-1].copy())

        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = torch.from_numpy(img_in.numpy()[:, ::-1, :].copy())
            img_edge = torch.from_numpy(img_edge.numpy()[:, ::-1, :].copy())
            img_rgb = torch.from_numpy(img_rgb.numpy()[:, ::-1, :].copy())
            img_tar = torch.from_numpy(img_tar.numpy()[:, ::-1, :].copy())
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = torch.FloatTensor(np.transpose(img_in.numpy(), (0, 2, 1)))
            img_edge = torch.FloatTensor(np.transpose(img_edge.numpy(), (0, 2, 1)))
            img_rgb = torch.FloatTensor(np.transpose(img_rgb.numpy(), (0, 2, 1)))
            img_tar = torch.FloatTensor(np.transpose(img_tar.numpy(), (0, 2, 1)))
            info_aug['trans'] = True
    # print(img_edge.size(), 'ssssssssss')
    return img_in, img_edge, img_rgb, img_tar, info_aug


"Read the data from folder in training"
class DatasetFromFolder(data.Dataset):
    def __init__(self, hr_dir, lr_dir, edge_dir, rgb_dir, patch_size, upscale_factor, dataset, data_augmentation,
                 input_transform=None, input_edge_transform=None, input_rgb_transform=None,target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filepaths = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
        self.lr_dir = lr_dir
        self.edge_dir = edge_dir
        self.rgb_dir = rgb_dir
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.dataset = dataset

        self.input_transform = input_transform
        self.input_edge_transform = input_edge_transform
        self.input_rgb_transform = input_rgb_transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        "Get the filename with postfix"
        _, file = os.path.split(self.image_filepaths[index]) # file like "1.png"
        ##### print('<==============>', self.dataset)

        "Load target data"
        target = load_img(self.image_filepaths[index])
        # print(self.image_filenames[index], target.size)
        # print(self.edge_dir)
        # print(self.rgb_dir)

        "Load the data by determining the type of the dataset"
        # elif self.dataset == 'train_depthCanny_x8/':
        # print(os.path.join(self.edge_dir, os.path.splitext(file)[0] + '.png'))
        input_edge = load_img(os.path.join(self.edge_dir, os.path.splitext(file)[0] + '.png'))
        # print(os.path.join(self.rgb_dir, os.path.splitext(file)[0] + '.png'))
        input_rgb = load_img(os.path.join(self.rgb_dir, os.path.splitext(file)[0] + '.png'))

        input = load_img(os.path.join(self.lr_dir, os.path.splitext(file)[0] + '.png'))

        #######***
        "Transform the image data"
        if self.input_edge_transform:
            input_edge = self.input_edge_transform(input_edge)
        if self.input_rgb_transform:
            input_rgb = self.input_edge_transform(input_rgb)
            # print('input_edge_tttttttttttttttttt:', input_edge.size())
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
            # print('target:', target.size())
        # print('target:', target.size())

        "Augment the image data"
        if self.data_augmentation:
            input, input_edge,input_rgb, target, _ = augment(input, input_edge,input_rgb, target)

        # print('input_edge:', input_edge.size())
        # print('input:', input.size())
        # print('target:', target.size())
        return input_edge, input_rgb, input, target

    def __len__(self):
        return len(self.image_filepaths)


"Read the data from folder in evaluation"
class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir,rgb_dir,input_transform=None,input_rgb_transform=None,target_transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.input_rgb_transform = input_rgb_transform
        self.target_transform = target_transform
        self.lr_dir = lr_dir
        self.rgb_dir = rgb_dir

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        
        # if self.lr_dir == './data/test_x16/':
        input_rgb = load_img(os.path.join(self.rgb_dir,os.path.splitext(file)[0]+'.png'))
        # elif self.dataset == 'DIV2K_train_LR_aug_x4':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4.png'))

        if self.input_transform:
            input = self.input_transform(input)
        if self.input_rgb_transform:
            input_rgb = self.input_rgb_transform(input_rgb)
        return input, input_rgb, file
      
    def __len__(self):
        return len(self.image_filenames)
