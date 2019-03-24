from PIL import Image, ImageFont, ImageDraw
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.optimizers import SGD, RMSprop
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
import cv2 as cv
import string



X_test = []

files = list(os.walk('./'))[0][2]

files = list(filter(lambda x: '.jpg' in x, files))

print(files)

def load_std_img(x):
    img = Image.open(x).convert('L')
    bw, bh = img.size
    if bw != 19 or bh != 19:
        if bw > bh:
            image1 = Image.new("L", (bw, bw),255)
            image1.paste(img,(0,(bw-bh)//2))
        else:
            image1 = Image.new("L", (bh, bh),255)
            image1.paste(img,((bh-bw)//2,0))
        img = image1.resize((19,19))
    arr = img_to_array(img)/255
    return arr

for x in files:
    X_test.append(load_std_img(x))

X_test = np.array(X_test).reshape(-1,19,19,1)

model = load_model("./num_model.h5")
results = model.predict(X_test)
results = np.argmax(results,axis = 1)
results = results + 1

submission = {'file':files,'label':results}

submission = pd.core.frame.DataFrame(submission)

print(submission)

submission.to_csv("./answer.csv",index=False)



