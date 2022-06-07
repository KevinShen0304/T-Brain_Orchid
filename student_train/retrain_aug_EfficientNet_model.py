from random_aug import random_aug_v2, random_aug_v3

import glob
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder

tf.__version__
gpus = tf.config.list_physical_devices('GPU')
# =============================================================================
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# =============================================================================
# 只使用 % 的 GPU 記憶體
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.60)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# 讀train資料集合
df = pd.read_csv('csv/fold5.csv')
df['filepath'] = df.filename.apply(lambda x: f"img/{x}") 
encoder = OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(encoder.fit_transform(df[['category']]).toarray())
encoder_df.columns = encoder_df.columns.astype(str)
df = df.join(encoder_df)

# 讀pseudo_label資料集合
df_pseudo = pd.read_csv('csv/student_img_pseudo_label_step1_filter.csv')
label_cols = [str(i) for i in range(219)]
cols = ['filepath'] + label_cols

# 組合
fold = 0
df_train = df[df[f'fold{fold}']].reset_index(drop=True)[cols]
df_val = df[~df[f'fold{fold}']].reset_index(drop=True)[cols]
df_train = pd.concat([df_train,df_pseudo], axis = 0) # 加上pseudo_label

# KERAS PREPROCESSING
IMAGE_SIZE = (640, 640)
batch = 15

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range= [0.9,1.1],
                                                                preprocessing_function = random_aug_v3, 
                                                                rescale=1./255,)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe( 
                                                dataframe = df_train,
                                                x_col = 'filepath',
                                                y_col = label_cols,
                                                batch_size = batch,
                                                class_mode = 'raw',
                                                target_size=IMAGE_SIZE,
                                              )

val_generator = val_datagen.flow_from_dataframe( 
                                                dataframe = df_val,
                                                x_col = 'filepath',
                                                y_col = label_cols,
                                                batch_size = batch,
                                                class_mode = 'raw',
                                                target_size=IMAGE_SIZE,
                                              )

# 建立模型，準備訓練
model_name = 'Step1_EfficientNetV2_noisy_student'
log_dir = f'log/{model_name}_aug_{fold}/'
epochs = 600
num_classes = len(label_cols)

from tensorflow.python.keras.optimizer_v2.adam import Adam 
from loss import f1, acc, acc_f1

model_file = 'student_model/Step1_ep200.h5'
if model_file == '':
    from EfficientNetV2_m_model import build_model
    model = build_model(IMAGE_SIZE, num_classes)
else:
    model = tf.keras.models.load_model(model_file, custom_objects={"f1": f1})

#model.trainable = True
model.summary()

model.compile(optimizer = Adam(lr=0.00001), #lr=0.0001
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
                              factor=0.85, 
                              min_lr=0.0000001
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
model.save( f'{model_name}_aug_{fold}.h5')