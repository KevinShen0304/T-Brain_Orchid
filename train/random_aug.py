# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:26:11 2022

@author: User
"""

import numpy as np
from imgaug import augmenters as iaa
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def random_aug(image):
    aug_some = iaa.SomeOf((3,5), [
    #aug = iaa.Sequential([
        # arithmetic
        iaa.Add((-40, 40),per_channel=0.5),
        iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=True), # 噪聲
        iaa.Multiply((0.7, 1.3)), # 將圖像中的所有像素與特定值相乘，從而使圖像更暗或更亮。
        iaa.Cutout(nb_iterations=2, fill_mode="constant", cval=(0, 255), fill_per_channel=0.5), # 填充隨機區域(方形)
        iaa.Dropout(p=(0, 0.1), per_channel=0.5), #刪除 p圖像中所有像素的百分比
        iaa.CoarseDropout(0.05, size_percent=0.15, per_channel=0.5),
        iaa.CoarseSaltAndPepper(0.10, size_percent=(0.05, 0.2), per_channel=True),
        # blend
        iaa.BlendAlphaMask(iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),iaa.Clouds()), #添加雲 #運算超久...
        iaa.BlendAlphaElementwise((0.0, 1.0),iaa.Affine(rotate=(-20, 20)),per_channel=0.5), # 旋轉&殘影
        iaa.BlendAlphaVerticalLinearGradient(iaa.AveragePooling(11),start_at=(0.0, 1.0), end_at=(0.0, 1.0)), # 部分馬賽克
        # Blur
        iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)), #雙邊模糊

        # pooling
        iaa.AveragePooling(3),
        iaa.Sharpen(alpha=(0.0, 0.4), lightness=(0.75, 1.0)), # 銳利化
        # weather
        iaa.Clouds(),
        #iaa.Rain(speed=(0.1, 0.2)), #不能使用
        ],
        random_order=True)
    
    aug_seq = iaa.SomeOf((3,6),[
    #aug_seq = iaa.Sequential([
        iaa.Fliplr(0.5), # 水平翻轉
        # geometric
        iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}), # 縮放
        iaa.Affine(rotate=(-60, 60),shear=(-16, 16)), # 旋轉
        iaa.ShearX((-30, 30)), # 在 x 軸上應用仿射剪切來輸入數據
        iaa.Rotate((-60, 60)), # 旋轉
        iaa.PiecewiseAffine(scale=(0.01, 0.05)), # 局部扭曲 #運算超久...
        ],
        random_order=True)
    
    image = aug_some.augment_image(image)
    image = aug_seq.augment_image(image)    
    return(image)

def random_aug_v2(image):
    aug_some = iaa.SomeOf((2,3), [
        iaa.Add((-15, 15),per_channel=0.5),
        iaa.Cutout(nb_iterations=2, fill_mode="constant", cval=(0, 255), fill_per_channel=0.5), # 填充隨機區域(方形)
        iaa.CoarseDropout(0.05, size_percent=0.15, per_channel=0.5),
        iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)), #雙邊模糊
        iaa.Sharpen(alpha=(0.0, 0.4), lightness=(0.75, 1.0)), # 銳利化
        ],
        random_order=True)
    
    aug_seq = iaa.SomeOf((2,4),[
        iaa.Fliplr(0.5), # 水平翻轉
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}), # 縮放
        iaa.ShearX((-25, 25)), # 在 x 軸上應用仿射剪切來輸入數據
        iaa.Rotate((-30, 45)), # 旋轉
        ],
        random_order=True)
    
    image = aug_some.augment_image(image)
    image = aug_seq.augment_image(image)    
    return(image)

def random_aug_v3(image):

    aug1 = iaa.SomeOf((1,2), [
        iaa.Add((-40, 40),per_channel=0.5),
        iaa.Cutout(nb_iterations=4, fill_mode="constant", cval=(0, 255), fill_per_channel=0.5), # 填充隨機區域(方形)
        iaa.CoarseDropout(0.05, size_percent=0.15, per_channel=0.5),
        iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)), #雙邊模糊
        iaa.Sharpen(alpha=(0.0, 0.4), lightness=(0.75, 1.0)), # 銳利化
        iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1)), # 粗胡椒粒
        ],
        random_order=True)
    
# =============================================================================
#     # pillike
#     aug_pillike = iaa.SomeOf((3,5),[
#         iaa.pillike.EnhanceContrast(), # 增強對比度
#         iaa.pillike.EnhanceColor(), # 增強色彩
#         iaa.pillike.Autocontrast(), #自動對比度
#         iaa.pillike.Equalize(), # 均衡
#         iaa.pillike.EnhanceBrightness(), # 增強亮度
#         iaa.pillike.EnhanceSharpness(), # 增強清晰度
#         iaa.pillike.FilterBlur()
#         ],
#         random_order=True)
#     
# =============================================================================
    # size
    aug_size = iaa.SomeOf((1,2),[
        iaa.Fliplr(1), # 水平翻轉
        iaa.Affine(scale={"x": (0.7, 1.3), "y": (0.7, 1.3)}), # 縮放
        iaa.ShearX((-40, 40)), # 在 x 軸上應用仿射剪切來輸入數據
        iaa.Rotate((-60, 60)), # 旋轉
        ],
        random_order=True)
    
    image = aug1.augment_image(image)
    #image = aug_pillike.augment_image(image) 
    image = aug_size.augment_image(image)
    return(image)


# In[]
if __name__ == '__main__':
    import cv2
    img = cv2.imread('img/0a1h7votc5.jpg')
    images_aug = random_aug_v3(img)
    assert(img.shape==images_aug.shape)
    plt.imshow(img)
    plt.imshow(images_aug)
    
    # 計算運算時間
    import time
    start = time.time()
    for i in range(100):
        images_aug = random_aug_v3(img)
    end = time.time()
    print('計算100次花費時間')
    print(end - start)

    
    
    
    
    
    