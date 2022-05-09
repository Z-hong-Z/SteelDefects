import paddle
import paddlex as pdx
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
import PIL.Image as Image
import cv2
import os
from random import shuffle
from paddlex.det import transforms as T
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib

matplotlib.use('Agg')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 配置显卡

# 训练数据集
trainDateset = open('./PaddleDetection/dataset/SteelDEC_VOCData/train_list.txt', 'r')
trainDateset_jpg = trainDateset.readlines()

# 验证数据集
evalDateset = open('./PaddleDetection/dataset/SteelDEC_VOCData/val_list.txt', 'r')
evalDateset_jpg = evalDateset.readlines()

# 测试数据集
testDateset = open('./PaddleDetection/dataset/SteelDEC_VOCData/test_list.txt', 'r')
testDateset_jpg = testDateset.readlines()
print('训练集样本量: {}，验证集样本量: {}，测试集样本量：{}'.format(len(trainDateset_jpg), len(evalDateset_jpg), len(testDateset_jpg)))


# 对数据集进行数据增强等预处理
# 对数据进行增强

def preprocess(dataType="train"):
    if dataType == "train":
        transform = T.Compose([
            # 对图像进行mixup操作，模型训练时的数据增强操作，目前仅YOLOv3模型支持该transform
            T.MixupImage(mixup_epoch=10),
            # 随机扩张图像
            # T.RandomExpand(),
            # 以一定的概率对图像进行随机像素内容变换
            # T.RandomDistort(brightness_range=1.2, brightness_prob=0.3), 
            # 随机裁剪图像
            T.RandomCrop(),
            # 根据图像的短边调整图像大小
            # T.ResizeByShort(), 
            # 调整图像大小,[’NEAREST’, ‘LINEAR’, ‘CUBIC’, ‘AREA’, ‘LANCZOS4’, ‘RANDOM’]
            T.Resize(target_size=608, interp='RANDOM'),
            # 以一定的概率对图像进行随机水平翻转
            T.RandomHorizontalFlip(),
            # 对图像进行标准化
            T.Normalize()
        ])
        return transform
    else:
        transform = T.Compose([
            T.Resize(target_size=608, interp='CUBIC'),
            T.Normalize()
        ])
        return transform

preprocess()


# 定义 数据集的 transforms
train_transforms = preprocess(dataType="train")
eval_transforms = preprocess(dataType="eval")

label = open('./PaddleDetection/dataset/SteelDEC_VOCData/labels.txt', 'r')
label_ = label.readlines()
print('标签类型：')
print(label_)

train_dataset = pdx.datasets.VOCDetection(
    data_dir='./PaddleDetection/dataset/SteelDEC_VOCData',
    file_list='./PaddleDetection/dataset/SteelDEC_VOCData/train_list.txt',
    label_list='./PaddleDetection/dataset/SteelDEC_VOCData/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='./PaddleDetection/dataset/SteelDEC_VOCData',
    file_list='./PaddleDetection/dataset/SteelDEC_VOCData/val_list.txt',
    label_list='./PaddleDetection/dataset/SteelDEC_VOCData/labels.txt',
    transforms=eval_transforms)

# num_classes有些模型需要加1 比如faster_rcnn
num_classes = len(train_dataset.labels)
# 定义PPYOLO模型
model = pdx.det.PPYOLO(num_classes=num_classes, )

model.train(
    num_epochs=30000,  # 设置训练轮数
    train_dataset=train_dataset,  # 设置训练数据集
    train_batch_size=8,  # 设置Bs
    eval_dataset=eval_dataset,  # 设置评估验证数据集
    learning_rate=3e-7,  # 设置学习率
    warmup_steps=90,
    warmup_start_lr=0.0,
    # 定义保村间隔轮数，即每7轮保存一次训练的模型结果
    save_interval_epochs=10,
    pretrain_weights='./PaddleDetectionoutput/PPYOLO/pretrain/model.pdparams',
    lr_decay_epochs=[50, 100, 150, 200],  # 定义学习率衰减轮数范围
    save_dir='./PaddleDetectionoutput/PPYOLO',  # 定义模型保存输出文件目录
    use_vdl=True)  # 定义使用Visual DL可视化工具
