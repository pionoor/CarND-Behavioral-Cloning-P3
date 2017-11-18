import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
import numpy as np
import sklearn
import random

#reading the lines from CSV
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
angles = []

#extracting the paths to the frames and load thim into arrays.
for line in lines:
    centerImgName = line[0].split('/')[-1]
    leftImgName = line[1].split('/')[-1]
    rightImgName = line[2].split('/')[-1]
    currentPath = 'data/IMG/'

    center_image = cv2.cvtColor(cv2.imread(currentPath + centerImgName), cv2.COLOR_BGR2YUV)
    
    left_image = cv2.cvtColor(cv2.imread(currentPath + leftImgName), cv2.COLOR_BGR2YUV)
    
    right_image = cv2.cvtColor(cv2.imread(currentPath + rightImgName), cv2.COLOR_BGR2YUV)
    
    image_flipped = np.copy(np.fliplr(center_image))


    images.extend((center_image, left_image, right_image, image_flipped))

     
    center_angle = float(line[3])
    correction = 0.1
    left_angle = center_angle + correction
    right_angle = center_angle - correction
    fliped_angle = - center_angle           

    angles.extend((center_angle, left_angle, right_angle, fliped_angle))
    

X_train = np.array(images)
y_train = np.array(angles)

#nVida Self-Driving Car Model
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 

model.add(Lambda(lambda x: x/255.5 - 0.5, input_shape=(160, 320, 3)))

model.add(keras.layers.Cropping2D(((50,20),(0,0)), input_shape=(160, 320,3)))

model.add(keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation=('relu')))

model.add(keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation=('relu')))

model.add(keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation=('relu')))

model.add(keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation=('relu')))

model.add(keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation=('relu')))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(50))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Dense(1))

model.compile(loss='mse', optimizer='adam')


model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')
            



