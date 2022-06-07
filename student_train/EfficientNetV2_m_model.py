# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:41:55 2022

@author: shen
"""

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import tensorflow as tf

def build_model(IMAGE_SIZE, num_classes):
    #dropout = 0.4
    dropout = 0.50
    #inputs = layers.Input(shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model = tf.keras.models.load_model('student_model/efficientnetv2-m-21k.h5')
    print(model.summary())

    # Freeze the pretrained weights
    model.trainable = False
    
    # Rebuild top
    x = layers.BatchNormalization()(model.layers[-2].output)
    x = layers.Dropout(dropout, name='dropout_00_')(x)
    x = Dense(1152)(x) #1152
    x = layers.Dropout(dropout, name='dropout_0_')(x)
    x = Dense(768)(x) #768
    x = layers.Dropout(dropout, name='dropout_1_')(x)
    x = Dense(384)(x) #384
    x = layers.Dropout(dropout, name='dropout_2_')(x)
# =============================================================================
#     x = Dense(256)(x) 
#     x = layers.Dropout(dropout, name='dropout_3_')(x)
# =============================================================================
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = Model(model.input, outputs, name="EfficientNetV2-m")
    return model

# In[]
if __name__=='__main__':
    IMAGE_SIZE, num_classes = (320, 320), 219
    model = build_model(IMAGE_SIZE, num_classes)
    model.summary()