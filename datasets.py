import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms  # 图像预处理包，可处理PIL、Tensor格式的数据
import os
import csv
import random
from PIL import Image  # 处理PIL格式的图像
from imresize import imresize  # 调用图像下采样函数
import numpy as np 

class SRDataset(Dataset):
    
    def __init__(self, split, config):
        """
        :param data_folder: # folder with JSON data files
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """
        self.split = split  # 选择所处状态
        self.csv_folder = config.csv_folder  # csv文件所在文件夹
        self.HR_data_folder = config.HR_data_folder  # 高分辨率图像所在文件夹
        self.LR_data_folder = config.LR_data_folder  # 低分辨率图像所在文件夹
        self.crop_size = int(config.crop_size)  # 目标HR图像的裁剪尺寸
        self.scaling_factor = int(config.scaling_factor)  # config.scaling_factor = 4
        self.patch_num = 100  # 

        assert self.split in {'train', 'valid', 'Set5', 'Set14', 'B100', 'Urban100'}

        self.HR_images = []  # 记录高分辨率图像名称，例：bridge.png
        self.LR_images = []  # 记录低分辨率图像名称，例：bridgex4.png
        with open(os.path.join(self.csv_folder, self.split + '_images.csv'), 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                self.HR_images.append(line[0]) # type(line)=list，内容为csv文件的每一行
                self.LR_images.append(line[1])

    def __getitem__(self, i):

        if self.split != 'train':
            hr_image = Image.open(os.path.join(self.HR_data_folder, self.HR_images[i]), mode='r')
            lr_image = Image.open(os.path.join(self.LR_data_folder, self.LR_images[i]), mode='r')
            hr_image = hr_image.convert('RGB')  # 转成RGB格式
            lr_image = lr_image.convert('RGB')  # 转成RGB格式
            
            if lr_image.width * self.scaling_factor != hr_image.width or lr_image.height * self.scaling_factor != hr_image.height:
                x_remainder = hr_image.width % self.scaling_factor
                y_remainder = hr_image.height % self.scaling_factor
                left = x_remainder // 2
                top = y_remainder // 2
                right = left + (hr_image.width - x_remainder)
                bottom = top + (hr_image.height - y_remainder)
                hr_image = hr_image.crop((left, top, right, bottom))
                hr_image = np.asarray(hr_image)
                lr_image = imresize(hr_image, scalar_scale=1.0/self.scaling_factor)  # 下采样
                lr_image = Image.fromarray(np.uint8(lr_image))
                hr_image = Image.fromarray(np.uint8(hr_image))
            
            lr_image = transforms.functional.to_tensor(lr_image)  # Img转成tensor
            hr_image = transforms.functional.to_tensor(hr_image)  # Img转成tensor

            return lr_image, hr_image
        
        if self.split == 'train':
            i = i // self.patch_num
        
        hr_image = Image.open(os.path.join(self.HR_data_folder, self.HR_images[i]), mode='r')
        lr_image = Image.open(os.path.join(self.LR_data_folder, self.LR_images[i]), mode='r')
        hr_image = hr_image.convert('RGB')
        lr_image = lr_image.convert('RGB')

        if lr_image.width * self.scaling_factor != hr_image.width or lr_image.height * self.scaling_factor != hr_image.height:
            x_remainder = hr_image.width % self.scaling_factor
            y_remainder = hr_image.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (hr_image.width - x_remainder)
            bottom = top + (hr_image.height - y_remainder)
            hr_image = hr_image.crop((left, top, right, bottom))
            hr_image = np.asarray(hr_image)
            lr_image = imresize(hr_image, scalar_scale=1.0/self.scaling_factor)
            lr_image = Image.fromarray(np.uint8(lr_image))
            hr_image = Image.fromarray(np.uint8(hr_image))

        lr_image, hr_image = self._sample_patch(lr_image, hr_image)  # 打补丁
        lr_image, hr_image = self._augment(lr_image, hr_image)  # 图片增强
        lr_image = transforms.functional.to_tensor(lr_image)
        hr_image = transforms.functional.to_tensor(hr_image)

        return lr_image, hr_image

    def _sample_patch(self, lr_image, hr_image):

        if self.split == 'train':
            # sample patch while training 训练时的样本补丁
            lr_crop_size = self.crop_size // self.scaling_factor
            left = random.randint(2, lr_image.width - lr_crop_size - 2)
            top = random.randint(2, lr_image.height - lr_crop_size - 2)
            right = left + lr_crop_size
            bottom = top + lr_crop_size
            lr_image = lr_image.crop((left, top, right, bottom))
            hr_image = hr_image.crop((left * self.scaling_factor, top * self.scaling_factor, right * self.scaling_factor, bottom * self.scaling_factor))

        return lr_image, hr_image

    def _augment(self, lr_image, hr_image):

        if self.split == 'train': 
            # augmentation while training 训练时进行图片增强
            if random.random() < 0.5:
                lr_image = lr_image.transpose(Image.FLIP_LEFT_RIGHT)  # 左右对称翻转
                hr_image = hr_image.transpose(Image.FLIP_LEFT_RIGHT)  # 左右对称翻转
            if random.random() < 0.5:
                lr_image = lr_image.transpose(Image.FLIP_TOP_BOTTOM)  # 上下对称翻转
                hr_image = hr_image.transpose(Image.FLIP_TOP_BOTTOM)  # 上下对称翻转
            if random.random() < 0.5:
                lr_image = lr_image.rotate(90)  # 旋转90°
                hr_image = hr_image.rotate(90)  # 旋转90°

        return lr_image, hr_image

    def __len__(self):

        if self.split == 'train':
            return len(self.HR_images) * self.patch_num  # 训练时每次输入patch_num张图片
        else:
            return len(self.HR_images)
