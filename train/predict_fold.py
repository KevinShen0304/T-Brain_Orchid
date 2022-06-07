# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:07:24 2022

@author: User
"""

import tensorflow as tf

# 只使用 % 的 GPU 記憶體
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.60)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


import pandas as pd
import numpy as np
from loss import f1, acc, acc_f1

# 讀train資料集合
df = pd.read_csv('csv/fold5.csv')
df['filepath'] = df['filename'].apply(lambda x: f"img/{x}") 
df['category'] = df['category'].astype(str).str.zfill(4)
df = df[0:50]

# =============================================================================
# # 讀test資料集合
# df = pd.read_csv('csv/test_img.csv')
# df['category'] = ''
# =============================================================================

fold_num = 5
predicts = np.zeros((len(df),219))
for i in range(fold_num):
    print(i)
    fold = i
    model = tf.keras.models.load_model(f'model/EfficientNetV2B3_adam_aug_{fold}.h5', custom_objects={"f1": f1})
    IMAGE_SIZE = (640, 640) #注意model與size的關係
    batch = 30
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_dataframe( 
                                                    dataframe = df,
                                                    x_col = 'filepath',
                                                    y_col = 'category',
                                                    batch_size = batch,
                                                    class_mode = 'categorical',
                                                    target_size=IMAGE_SIZE,
                                                    shuffle = False
                                                  )
    test_generator.reset()
    
    #with tf.device('/cpu:0'):
    #    predict = model.predict_generator(test_generator)
    
    predict = model.predict_generator(test_generator)
    predicts += predict/fold_num

y_pred = np.argmax(predict, axis=1)
max_prod = np.max(predict, axis=1)

df['y_pred'] = y_pred
df['max_prod'] = max_prod

#result = df[['filepath','label','y_pred','max_prod']]
result = df[['filepath','y_pred','max_prod']]
result.to_csv('predict/test_result.csv', index=False)
