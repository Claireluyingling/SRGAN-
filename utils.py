from PIL import Image
import os
import json
import random
# from work.external_libraries.torchvision.transforms import functional as F  # 报错后自己写的
import torchvision.transforms.functional as F
import torch
import math

def convert_image(img, source, target, device):
    """
    Convert an image from a source format to a target format.
    将图像从源格式转换为目标格式。
    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges) 传入图片的像素值范围
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),  由 imagenet 均值和方差标准化的像素值
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM) YCbCr 颜色格式的亮度通道 Y，用于计算 PSNR 和 SSIM
    :return: converted image 转换后的图像
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Some constants 按照github上tutorial设置的常数
    rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)   
    imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
    imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    # Convert from source to [0, 1] 从源转换为 [0, 1]
    if source == 'pil':
        img = F.to_tensor(img)

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.
    
    elif source == '[0, 1]':
        pass

    # Convert from [0, 1] to target 从 [0, 1] 转换为目标
    if target == 'pil':
        img = F.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[-1, 1]':
        img = 2. * img - 1.
    
    elif target == '[0, 1]':
        pass

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std  # 3维
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda  # 4维
    # 使用torch.matmul()找到一个四维tensor和一个一维tensor的最后维之间的点积
    elif target == 'y-channel':
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    跟踪指标的最新值、平均值、总和和数量。
    """
    # 初始化时重置所有计数
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # 传入值
        self.avg = 0  # 均值
        self.sum = 0  # 总值
        self.count = 0  # 数量

    def update(self, val, n=1):
        self.val = val  # val为最新的传入值
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, filename):
    """
    Save model checkpoint.
    保存checkpoint到指定的文件, state为checkpoint的内容(没用上)
    :param state: checkpoint contents
    """
    torch.save(state, filename)

# 用于修改学习率
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    以指定的因子缩小学习速率，shrink_factor衰减率
    :param optimizer: optimizer whose learning rate must be shrunk. 需要改变学习率的优化器
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with. 将学习率修改成原来的(0, 1)倍
    """

    print("\nDECAYING learning rate.")  
    # 当学习率衰减时输出提醒，param_groups记录下变化前后的参数，排在最前的为latest的参数
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],)) 
