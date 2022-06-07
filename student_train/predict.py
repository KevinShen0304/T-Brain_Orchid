# =============================================================================
# # 防止圖片太大
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# =============================================================================

import tensorflow as tf

# 只使用 % 的 GPU 記憶體
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.60)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import pandas as pd
import numpy as np
from loss import f1, acc, acc_f1

# 讀test資料集合
df = pd.read_csv('csv/student_img.csv')
df['category'] = ''
#df = df[13000:]

# =============================================================================
# # 讀train資料集合
# df = pd.read_csv('csv/label.csv')
# df['filepath'] = df.filename.apply(lambda x: f"img/{x}")
# =============================================================================
fold_num = 5
predicts = np.zeros((len(df),219))
for i in range(fold_num):
    print(i)
    fold = i
    model = tf.keras.models.load_model(f'student_model/deep/model_deep_v2_{i}.h5', custom_objects={"f1": f1})
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
    predicts += predict/fold_num

pre = pd.DataFrame(predicts)
result = df[['filepath']].reset_index(drop=True)
result = pd.concat([result,pre], axis = 1)

result.to_csv('csv/student_img_pseudo_label_step1.csv', index=False)
#result.to_csv('csv/train_predict.csv', index=False)

