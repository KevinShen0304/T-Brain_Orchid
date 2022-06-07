# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:49:12 2022

@author: User
"""
from random_aug import random_aug_v2, random_aug_v3

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

for i in range(0,5):
    fold = i
    model_file = f'model/EfficientNetV2B3_size640_drop0.35_adam_{fold}.h5'
    df_train = df[df[f'fold{fold}']][['id','label']].reset_index(drop=True)
    df_val = df[~df[f'fold{fold}']][['id','label']].reset_index(drop=True)
    
    # KERAS PREPROCESSING
    IMAGE_SIZE = (640, 640)
    batch = 4
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range= [0.85,1.1],
                                                                    preprocessing_function = random_aug_v3, 
                                                                    rescale=1./255,)
    
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
    model_name = 'EfficientNetV2B3'
    optimizer = 'adam' #'sgd'
    log_dir = f'log/{model_name}_{optimizer}_aug_{fold}/'
    epochs = 200
    num_classes = len(set(list(df['label'])))
    
    from tensorflow.python.keras.optimizer_v2.adam import Adam 
    from EfficientNetV2_model import build_model
    from loss import f1, acc, acc_f1
    
    model = tf.keras.models.load_model(model_file, custom_objects={"f1": f1})
    model.summary()
    model.compile(optimizer = Adam(lr=0.0001), #lr=0.0001
                  # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  loss = tf.keras.losses.categorical_crossentropy,
                  metrics = ['accuracy', f1], #acc, f1, acc_f1
                  )
    # 開始訓練
    from datetime import datetime
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    # 設定callbacks
    logging = TensorBoard(log_dir=log_dir)
    reduce_lr = ReduceLROnPlateau(monitor='loss', 
                                  patience=5, 
                                  verbose=1, 
                                  factor=0.75, 
                                  min_lr=0.000001
                                  )
    checkpoint = ModelCheckpoint( 
                                  # log_dir + model_name + 'ep{epoch:06d}-loss{loss:.6f}-val_loss{val_loss:.6f}.h5',
                                  log_dir + model_name + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-val_accuracy{val_accuracy:.3f}.h5',
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
    model.save( f'{model_name}_{optimizer}_aug_{fold}.h5')