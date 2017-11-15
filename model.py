import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import random

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                centerImgName = batch_sample[0].split('/')[-1]
                leftImgName = batch_sample[1].split('/')[-1]
                rightImgName = batch_sample[2].split('/')[-1]
                currentPath = 'data/IMG/'
                
                center_image = cv2.imread(currentPath + centerImgName)
                left_image = cv2.imread(currentPath + leftImgName)
                right_image = cv2.imread(currentPath + rightImgName)
    
                image_flipped = np.copy(np.fliplr(center_image))
                
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                images.append(image_flipped)
                
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle + 0.07
                right_angle = center_angle +0.07
                fliped_angle = -center_angle
                
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
                angles.append(fliped_angle)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255.5 - 0.5, input_shape=(row, col, ch)))

model.add(keras.layers.Cropping2D(((70,25),(0,0)), input_shape=(160,320,3)))

model.add(keras.layers.Conv2D(24, (5, 5)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv2D(36, (5, 5)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv2D(48, (5, 5)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(50))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Dense(1))

model.compile(loss='mse', optimizer='adam')

train_steps = np.ceil( len( train_samples )/32 ).astype( np.int32 )
validation_steps = np.ceil( len( validation_samples )/32 ).astype( np.int32 )

model.fit_generator( train_generator, \
    steps_per_epoch = train_steps, \
    epochs=5, \
    verbose=1, \
    callbacks=None, 
    validation_data=validation_generator, \
    validation_steps=validation_steps, \
    class_weight=None, \
    max_q_size=10, \
    workers=1, \
    pickle_safe=False, \
    initial_epoch=0)

model.save('model.h5')
            



