# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:24:05 2022
https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Accuracy
https://stackoverflow.com/questions/65023353/difference-between-keras-metrics-accuracy-and-accuracy
@author: shen
"""

from tensorflow.keras.metrics import CategoricalAccuracy, Recall, Precision
import tensorflow.keras.backend as K

def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def acc(y_true, y_pred):
    m = CategoricalAccuracy()
    m.update_state(y_true, y_pred)
    acc_val = m.result().numpy()
    return acc_val

def acc_f1(y_true, y_pred):
    acc_f1_val = (f1(y_true, y_pred) + acc(y_true, y_pred))/2
    return acc_f1_val

# =============================================================================
# # Example
# model.compile(optimizer = Adam(), #lr=0.01
#               loss = tf.keras.losses.categorical_crossentropy,
#               metrics = ['accuracy', f1], #acc, f1, acc_f1
#               #run_eagerly=True
#               )
# =============================================================================

# =============================================================================
# def f1(y_true, y_pred):
#     r = Recall()
#     r.update_state(y_true, y_pred)
#     r_val = r.result().numpy()
#     
#     p = Precision()
#     p.update_state(y_true, y_pred)
#     p_val = p.result().numpy()
#     
#     f1_val = 2*(r_val*p_val)/(r_val+p_val)
#     return f1_val
# =============================================================================
# =============================================================================
# def f1(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     recall = true_positives / (possible_positives + K.epsilon())
#     f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
#     return f1_val
# =============================================================================
