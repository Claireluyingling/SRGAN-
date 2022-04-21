import torch
from torch import nn
import torchvision
import math

class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    由卷积层、批归一化层、激活层组成的卷积块
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :param in_channels: number of input channels  输入通道数
        :param out_channels: number of output channe  输出通道数
        :param kernel_size: kernel size  卷积核大小
        :param stride: stride  步长
        :param batch_norm: include a BN layer?  是否包含批归一化层
        :param activation: Type of activation; None if none  激活函数种类
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}
        #  ParametricReLU函数、LeakyReLU函数、双曲函数
        #  A container that will hold the layers in this convolutional block
        #  用来存放这个卷积块中的图层的容器
        layers = list()

        #  A convolutional layer  增加一层卷积层
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size// 2))

        #  A batch normalization (BN) layer, if wanted  增加一层批归一化层
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        #  An activation layer, if wanted  增加一层激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))  # 论文中α=0.2
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        #  Put together the convolutional block as a sequence of the layers in this container
        #  将卷积块作为这个容器中的层按序列放在一起
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, in_channels, w, h)  输入图像
        :return: output images, a tensor of size (N, out_channels, w, h)  输出图像
        """
        output = self.conv_block(input)  # 得到(N, out_channels, w, h)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    包括卷积、像素shuffle和ParametricReLU函数激活层的次像素卷积块
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution  卷积核大小
        :param n_channels: number of input and output channels  输入、输出通道
        :param scaling_factor: factor to scale input images by (along both dimensions)  按比例缩放输入图像的因子(沿两个维度)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        #  A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        #  卷积层通过缩放因子的2倍、像素shuffle和PReLU函数增加通道数量
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        # 这些额外的通道被打乱以形成额外的像素，并通过缩放因子对每个维度进行缩放
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    由两个卷积块组成的残差快，两个卷积块之间有一个残差连接
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :param kernel_size: kernel size  卷积核大小
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        输入和输出通道的数量(相同，因为输入必须添加到输出中)
        """
        super(ResidualBlock, self).__init__()

        # The first convolutional block  第一层卷积层，有激活函数PReLU
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='prelu')

        # The second convolutional block  第二层卷积层，无激活函数
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class SRResNet(nn.Module):
    """
    The SRResNet, as defined in the paper.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        转换输入和输出的第一个和最后一个卷积的内核大小
        :param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        中间所有卷积的核大小，即残差卷积块和次像素卷积块的核大小
        :param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        中间通道数，即残差卷积块和次像素卷积块的输入输出通道
        :param n_blocks: number of residual blocks  残差快数量
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block  
        将输入图像按次像素卷积块(沿两个维度)缩放
        """
        super(SRResNet, self).__init__()

        # Scaling factor must be 2, 4, or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        # The first convolutional block  第一个卷积块，有激活函数PReLU，核大小为large_kernel_size
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='prelu')

        # A sequence of n_blocks residual blocks, each containing a skip-connection across the block 
        # 一串n_blocks个剩余块，每个块包含一个跨块的跳过连接
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # Another convolutional block  另一个卷积块，无激活函数，核大小为small_kernel_size
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
        # 放大通过次像素卷积完成，每个这样的块放大2倍
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # The last convolutional block  最后一个卷积块，有激活函数Tanh，核大小为large_kernel_size
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='tanh')

    def forward(self, lr_imgs):
        """
        Forward prop.
        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (N, 3, w, h)
        residual = output  # (N, n_channels, w, h)
        output = self.residual_blocks(output)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)
        output = self.subpixel_convolutional_blocks(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        sr_imgs = self.conv_block3(output)  # (N, 3, w * scaling factor, h * scaling factor)

        return sr_imgs


class Generator(nn.Module):
    """
    The generator in the SRGAN, as defined in the paper. Architecture identical to the SRResNet.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        转换输入和输出的第一个和最后一个卷积的内核大小
        :param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        中间所有卷积的核大小，即残差卷积块和次像素卷积块的核大小
        :param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        中间通道数，即残差卷积块和次像素卷积块的输入输出通道
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        """
        super(Generator, self).__init__()

        # The generator is simply an SRResNet  生成器只是一个SRResNet
        self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                            n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)

    def initialize_with_srresnet(self, srresnet_checkpoint):
        """
        Initialize with weights from a trained SRResNet. 用训练好的SRResNet中的权重进行初始化
        :param srresnet_checkpoint: checkpoint filepath
        """
        srresnet = torch.load(srresnet_checkpoint)['model']
        self.net.load_state_dict(srresnet.state_dict())

        print("\nLoaded weights from pre-trained SRResNet.\n")

    def forward(self, lr_imgs):
        """
        Forward prop.
        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h) 低分辨率输入图像
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor) 超分辨率图像输出
        """
        sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return sr_imgs


class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN.
    """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        """
        :param kernel_size: kernel size in all convolutional blocks
        :param n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
        :param n_blocks: number of convolutional blocks
        :param fc_size: size of the first fully connected layer
        """
        super(Discriminator, self).__init__()

        in_channels = 3
        '''
        A series of convolutional blocks
        The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
        The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
        The first convolutional block is unique because it does not employ batch normalization
        第一、第三、第五等卷积块增加了通道的数量，但保留了图像的大小
        第二、第四、第六等卷积块保留相同数量的通道，但图像大小减半
        第一个卷积块是唯一的，因为它不采用批处理归一化   
        '''
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='leakyrelu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)
        '''
        An adaptive pool layer that resizes it to a standard size   
        For the default input size of 96 and 8 convolutional blocks, this will have no effect 
        将其调整到标准大小的自适应池层, 对于默认的输入大小为96和8个卷积块，这将没有任何影响  
        '''
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)

        # Don't need a sigmoid layer because the sigmoid operation is performed by PyTorch's nn.BCEWithLogitsLoss()

    def forward(self, imgs):
        """
        Forward propagation.
        :param imgs: high-resolution or super-resolution images which must be classified as such, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        高分辨率或超分辨率的图像，必须被分类为这样的大小tensor
        :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        是否为高分辨率图像的分数(logit)  
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)  # 对于任何输入大小的输入，可以将输出尺寸指定为H*W，自动计算kernel_size和pad
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class TruncatedVGG19(nn.Module):
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network', as defined in the paper.
    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    一个截断的VGG19网络, 其输出为VGG19网络中第i个maxpooling层之前第j次卷积(激活后)得到的特征图，
    用于计算该VGG特征空间的MSE损失，即VGG损失。
    """

    def __init__(self, i, j):
        """
        :param i: the index i in the definition above
        :param j: the index j in the definition above
        """
        super(TruncatedVGG19, self).__init__()

        # Load the pre-trained VGG19 available in torchvision 下载预训练的vgg19网络
        vgg19 = torchvision.models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterate through the convolutional section ("features") of the VGG19 迭代VGG19的卷积部分特性  
        for layer in vgg19.features.children():
            truncate_at += 1
            # Count the number of maxpool layers and the convolutional layers after each maxpool
            # 计算每个maxpool之后的maxpool层数和卷积层数
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool 
            # 如果在第(i - 1)个maxpool之后达到第j个卷积，就会中断  
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied 检查是否满足条件
        assert maxpool_counter == i - 1 and conv_counter == j, "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (
            i, j)

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        # 在第i个maxpool层之前截断到第j个卷积(+激活) 
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        Forward propagation
        :param input: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: the specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output