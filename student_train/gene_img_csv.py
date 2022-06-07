# -*- coding: utf-8 -*-
"""
Created on Mon May  9 20:02:37 2022
https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black
@author: shen
"""

from PIL import Image
# 防止圖片太大
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pandas as pd
import sys,glob
import os,sys

def make_square(im, min_size=256, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

# In[改副檔名]
#files = glob.glob('student_img/**/*.*',recursive=True)
files = glob.glob('**/*.*',recursive=True)

for infilename in files:
    oldbase = os.path.splitext(infilename)
    if oldbase[1]=='.jpg':
        pass
    else:
        new_infilename = oldbase[0]+'.jpg'
        os.rename(infilename, new_infilename)
        print(infilename)

# In[resize]

#files = glob.glob('student_img/**/*.*',recursive=True)

for file in files:
    img = Image.open(file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # new_img = img.resize((640, 640))
    new_img = make_square(img)
    new_img.save(file)
    print(file)


# In[正式開始]
files = glob.glob('student_img/**/*.*',recursive=True)

df = pd.DataFrame()

df['filepath'] = files

df.to_csv('csv/student_img.csv', index=False)