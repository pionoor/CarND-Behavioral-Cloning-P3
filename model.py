import csv
import cv2
import numpy as np
import keras
import cv2


data = []
with open('data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        data.append(line)
 


from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(data, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_data, batch_size=32)
validation_generator = generator(validation_data, batch_size=32)



#nVidia model
model = keras.models.Sequential()
model.add(keras.layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))
model.add(keras.layers.Cropping2D(((70,25),(0,0))))

model.add(keras.layers.Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))

model.add(keras.layers.Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))

model.add(keras.layers.Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))


model.add(keras.layers.Convolution2D(64, 3, 3, activation="relu"))
model.add(keras.layers.Convolution2D(64, 3, 3, activation="relu"))



model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(50))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_data), validation_data=validation_generator, nb_val_samples=len(validation_data), nb_epoch=3)

model.save('model.h5')
            

