# SRGAN-
Undergraduate graduation project 本科毕设项目

使用GAN实现4倍图像超分辨率任务，输入一张低分辨率图像LR，生成器会生成一张4倍超分辨率的图像。

训练集DIV2K数据集，包含800张2K左右高分辨率的图像和800张对应的低分辨率图像。

测试集、Set5、Set14、BSD100、Urban100，分别包括高分辨率图像和对应的低分辨率图像。

DIV2K验证集包含100张2K左右高分辨率的图像和800张对应的低分辨率图像，用于调参。

文件说明：

create_data_lists.py：数据预处理文件，生成训练测试所需的csv文件

utils.py：定义了可能用到的函数

datasets.py：定义符合pytorch标准的Dataset类

imresize.py：用python实现MATLAB resize函数，用于图像下采样

solver.py：定义了一个epoch的训练过程

models.py：定义SRGAN模型结构

train.ipynb：用于训练的jupyter文件

test.ipynb：加载指定的训练好的模型并测试，输出PSNR和SSIM指标

super_resolution.ipynb：加载指定的训练好的模型文件，针对单个图片进行4倍超分辨率，并对结果进行可视化
