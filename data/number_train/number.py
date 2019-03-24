from PIL import Image
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

X_train = []
Y_train = []

files = list(os.walk('./data'))[0][2]

for x in files:
    if '.jpg' not in x:
        files.remove(x)

for x in files:
        Y_train.append(int(re.sub(r"\D", "", x))-1)

for x in files:
        img = load_img("./data/"+x, color_mode="grayscale")
        arr = img_to_array(img)
        arr = arr/255
        X_train.append(arr)

Y_train = np.array(Y_train)
Y_train = Y_train.reshape(9,1)
Y_train = keras.utils.np_utils.to_categorical(Y_train, num_classes=9)

X_train = np.array(X_train)
X_train = X_train.reshape(9,19,19,1)

X_test = []

files = list(os.walk('./test_data'))[0][2]

for x in files:
        img = load_img("./test_data/"+x, color_mode="grayscale")
        arr = img_to_array(img)
        arr = arr/255
        X_test.append(arr)

X_test = np.array(X_test)
X_test = X_test.reshape(-1,19,19,1)

# print(X_train)
# print(Y_train)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                 activation='relu', input_shape=(19, 19, 1)))

model.add(Conv2D(filters=32, kernel_size=(5, 5),
                 padding='same', activation='relu'))

model.add(MaxPool2D())

model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='Same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(9, activation="softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                            patience=3,
                                                            verbose=1,
                                                            factor=0.5,
                                                            min_lr=0.00001)

epochs = 500

batch_size = 2

datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs,
                              verbose = 2, steps_per_epoch= 9 // batch_size
                              , callbacks=[learning_rate_reduction])

results = model.predict(X_test)

results = np.argmax(results,axis = 1)

results = results + 1

print(results)

model.save("./num_model.h5")



print("over")
