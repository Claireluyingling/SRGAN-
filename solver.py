# 定义了一个Epoch的训练过程，记录内容损失和对抗损失
import time
import torch.backends.cudnn as cudnn
from torch import nn
from models import Generator, Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import *
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter() # 实时监控     使用命令 tensorboard --logdir runs  进行查看

def train(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
          optimizer_g, optimizer_d, epoch, device, beta, print_freq):
    """
    One epoch's training.
    :param train_loader: train dataloader
    :param generator: generator
    :param discriminator: discriminator
    :param truncated_vgg19: truncated VGG19 network
    :param content_loss_criterion: content loss function (Mean Squared-Error loss)  均方误差记录内容损失
    :param optimizer_g: optimizer for the generator
    :param optimizer_d: optimizer for the discriminator
    :param epoch: epoch number
    """
    generator.train()
    discriminator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter() 
    losses_c = AverageMeter()
    losses_a = AverageMeter()
    losses_p = AverageMeter()
    losses_d = AverageMeter()

    start = time.time()
    n_iter = len(train_loader)

    # 按批处理
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24),  imagenet-normed 格式 经过SRDataset下采样HR图像得到LR图像，放到GPU上 
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  imagenet-normed 格式 直接将HR图像放到GPU上 
        lr_imgs = convert_image(lr_imgs, source='[0, 1]', target='imagenet-norm', device=device)  # 转尺寸
        hr_imgs = convert_image(hr_imgs, source='[0, 1]', target='imagenet-norm', device=device)  # 转尺寸

        # GENERATOR UPDATE 更新生成器
        sr_imgs = generator(lr_imgs)
        sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm', device=device)  # 转尺寸
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)  # batchsize X 512 X 6 X 6
        hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()  # batchsize X 512 X 6 X 6

        sr_discriminated = discriminator(sr_imgs)  # (batch X 1)  

        # 截断的vgg19用于计算生成的图像的内容损失
        # 生成器希望生成的图像能够完全迷惑判别器，因此它的预期所有图片真值为1
        content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)  # 计算内容损失
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))  # 计算生成损失
        perceptual_loss = content_loss + beta * adversarial_loss  # 总的感知损失

        optimizer_g.zero_grad()  # 后向传播.
        perceptual_loss.backward()  # 后向传播.
        optimizer_g.step()  # 更新生成器参数

        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(adversarial_loss.item(), lr_imgs.size(0))
        losses_p.update(perceptual_loss.item(), lr_imgs.size(0))

        # DISCRIMINATOR UPDATE 更新判别器
        # 判别器希望能够准确的判断真假，因此凡是生成器生成的都设置为0，原始图像均设置为1
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                           adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))
        optimizer_d.zero_grad()  # 后向传播.
        adversarial_loss.backward()  # 后向传播.

        optimizer_d.step()  # 更新判别器

        losses_d.update(adversarial_loss.item(), hr_imgs.size(0))  # 记录损失

        batch_time.update(time.time() - start)

        
        # Print status 打印
        # 监控图像变化
        if i==(n_iter-2):
            writer.add_image('SRGAN/epoch_'+str(epoch)+'_1', make_grid(lr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
            writer.add_image('SRGAN/epoch_'+str(epoch)+'_2', make_grid(sr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
            writer.add_image('SRGAN/epoch_'+str(epoch)+'_3', make_grid(hr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
        
        # 打印结果
        # print("第 "+str(i)+ " 个batch结束")

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'BatchTime: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'ContentLoss {loss_c.val:.4f} ({loss_c.avg:.4f}) '
                  'AdverageLoss {loss_a.val:.4f} ({loss_a.avg:.4f}) '
                  'PerceptualLoss {loss_p.val:.4f} ({loss_p.avg:.4f}) '
                  'Disc.Loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch,
                                                                          i,
                                                                          len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss_c=losses_c,
                                                                          loss_a=losses_a,
                                                                          loss_p=losses_p,
                                                                          loss_d=losses_d))
    # 删除
    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated

    # 监控损失值变化
    writer.add_scalar('SRGAN/Loss_c', losses_c.val, epoch) 
    writer.add_scalar('SRGAN/Loss_a', losses_a.val, epoch)   
    writer.add_scalar('SRGAN/Loss_p', losses_p.val, epoch)
    writer.add_scalar('SRGAN/Loss_d', losses_d.val, epoch)

writer.close()
