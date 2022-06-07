# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:20:57 2022

@author: shen
"""

import glob
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
tf.__version__
gpus = tf.config.list_physical_devices('GPU')
# =============================================================================
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# =============================================================================
# 只使用 % 的 GPU 記憶體
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.60)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# 讀資料集合
df = pd.read_csv('csv/fold5.csv')
df = df.rename(columns={'category':'label', 'filename':'id'})
df['id'] = df.id.apply(lambda x: f"img/{x}") 
df['label'] = df.label.astype(str).str.zfill(4)

for i in range(1,5):
    fold = i
    df_train = df[df[f'fold{fold}']][['id','label']].reset_index(drop=True)
    df_val = df[~df[f'fold{fold}']][['id','label']].reset_index(drop=True)
    
    # KERAS PREPROCESSING
    IMAGE_SIZE = (640, 640)
    batch = 30
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator( 
                                                                rescale=1./255,
                                                                brightness_range=[0.8, 1.1], # 亮度 [0.1, 10]
                                                                rotation_range=60, # 角度值，0~180，影象旋轉
                                                                width_shift_range=0.3, # 水平平移，相對總寬度的比例
                                                                height_shift_range=0.3, # 垂直平移，相對總高度的比例
                                                                shear_range=0.2, # 隨機錯切換角度
                                                                zoom_range=[0.8,1.2], # 隨機縮放範圍 # [lower,upper]
                                                                horizontal_flip=True, # 一半影象水平翻轉
                                                                fill_mode='constant', # 填充新建立畫素的方法 
                                                                )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe( 
                                                    dataframe = df_train,
                                                    x_col = 'id',
                                                    y_col = 'label',
                                                    batch_size = batch,
                                                    class_mode = 'categorical',
                                                    target_size=IMAGE_SIZE,
                                                  )
    
    val_generator = val_datagen.flow_from_dataframe( 
                                                    dataframe = df_val,
                                                    x_col = 'id',
                                                    y_col = 'label',
                                                    batch_size = batch,
                                                    class_mode = 'categorical',
                                                    target_size=IMAGE_SIZE,
                                                  )
    
    # 建立模型，準備訓練
    model_name = 'EfficientNetV2B3_size640_drop0.35'
    optimizer = 'adam' #'sgd'
    log_dir = f'log/{model_name}_{optimizer}_{fold}/'
    epochs = 200
    num_classes = len(set(list(df['label'])))
    
    from tensorflow.python.keras.optimizer_v2.adam import Adam 
    from EfficientNetV2_model import build_model
    from loss import f1, acc, acc_f1
    
    model = build_model(IMAGE_SIZE, num_classes)
    #model = tf.keras.models.load_model('model/EfficientNetV2B3ep000105-loss0.145761.h5', custom_objects={"f1": f1})
    
    model.summary()
    model.compile(optimizer = Adam(lr=0.0001), #lr=0.01
                  #loss = tf.keras.losses.categorical_crossentropy,
                  loss = tf.keras.losses.CategoricalCrossentropy(), #label_smoothing=0.2
                  metrics = ['accuracy', f1], #acc, f1, acc_f1
                  )
    # 開始訓練
    from datetime import datetime
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    # 設定callbacks
    logging = TensorBoard(log_dir=log_dir)
    reduce_lr = ReduceLROnPlateau(monitor='loss', 
                                  patience=10, 
                                  verbose=1, 
                                  factor=0.75, 
                                  min_lr=0.000001
                                  )
    checkpoint = ModelCheckpoint( 
                                  # log_dir + model_name + 'ep{epoch:06d}-loss{loss:.6f}-val_loss{val_loss:.6f}.h5',
                                  log_dir + model_name + 'ep{epoch:03d}-loss{loss:.2f}-val_loss{val_loss:.2f}-val_accuracy{val_accuracy:.2f}-val_f1{val_f1:.2f}.h5',
                                  monitor='loss',
                                  save_weights_only=False,
                                  save_best_only=True,
                                  period=5
                                  ) #訓練過程權重檔名稱由第幾輪加上損失率為名稱
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.01,
                                   patience=5,
                                   verbose=1
                                   )
    
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    
    # 訓練模型
    model.fit_generator(
                        # train_batches,
                        # class_weight=class_weights,
                        generator=train_generator,
                        validation_data = val_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        # initial_epoch = 1,
                        # validation_data = valid_batches,
                        #validation_steps = 1,
                        epochs=epochs,
                        callbacks=[logging, checkpoint, reduce_lr]
                        )
    
    # 儲存訓練好的模型
    model.save( f'{model_name}_{optimizer}_{fold}.h5')