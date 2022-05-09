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
import pandas as pd

# 使用paddlex加载训练过程中保存的最好的训练模型
model = pdx.load_model('./PaddleDetection/output/PPYOLO/best_model')
# 定义测试集数据文件路径
image_dir_file = './PaddleDetection/dataset/SteelDEC_VOCData/test_list.txt'

# 读取测试集txt文件，获取测试集每张图片的文件路径
txt_png = pd.read_csv(image_dir_file,header=None)
txt_jpgs = txt_png.iloc[:,0].str.split(' ',expand=True)[0].tolist()
txt_jpgs
# images = os.listdir(image_dir)

# 遍历每张测试集图片，进行预测
for img in txt_jpgs:
    image_name = './dataset/SteelDEC_VOCData/' + img
    result = model.predict(image_name)             # 使用predict接口进行预测
    pdx.det.visualize(image_name, result, threshold=0.2, save_dir='./output/PPYOLO/img_predict')
    # 设定阈值，将预测的图片保存到img_predict目录下