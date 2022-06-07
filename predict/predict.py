import tensorflow as tf

# 只使用 % 的 GPU 記憶體
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.75)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import pandas as pd
import numpy as np
from loss import f1, acc, acc_f1

# 讀test資料集合
df = pd.read_csv('csv/submission_template.csv')
df['filepath'] = df['filename'].apply(lambda x: f"img/{x}")
df['category'] = ''

fold_num = 5
predicts = np.zeros((len(df),219))
for i in range(fold_num):
    print(i)
    fold = i
    model = tf.keras.models.load_model(f'model/deep/model_deep_v2_{fold}.h5', custom_objects={"f1": f1})
    #model = tf.keras.models.load_model(f'model/noisy_student_{fold}.h5', custom_objects={"f1": f1})
    IMAGE_SIZE = (640, 640) #注意model與size的關係
    batch = 30
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_dataframe( 
                                                    dataframe = df,
                                                    x_col = 'filepath',
                                                    y_col = 'category',
                                                    batch_size = batch,
                                                    class_mode = 'raw',
                                                    target_size=IMAGE_SIZE,
                                                    shuffle = False
                                                  )
    #test_generator.reset()
    
    predict = model.predict_generator(test_generator)
    predicts += predict/(fold_num)

# predict_prob
pred = pd.DataFrame(predicts)
predict_prob = df[['filename']]
predict_prob = pd.concat([predict_prob,pred], axis = 1)
predict_prob.to_csv('csv/predict_prob.csv', index=False)

# predict_label
y_pred = np.argmax(predicts, axis=1)
predict_label = df[['filename']]
predict_label['category'] = y_pred
predict_label.to_csv('csv/predict_label.csv', index=False)
