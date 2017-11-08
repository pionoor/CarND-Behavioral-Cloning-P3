import csv
import cv2
import numpy as np
import keras



lines = []
with open('data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        lines.append(line)
        
images = []
SteeringMeasurements = []
for line in lines:
    sourcePath = line[0]
    fileName= sourcePath.split('/')[-1]
    currentPath = 'data/IMG/' + fileName
    image = cv2.imread(currentPath)
    images.append(image)
    SteeringMeasurements.append(float(line[3]))
    
X_train = np.array(images)
y_train = np.array(SteeringMeasurements)


#nVidia model
model = keras.models.Sequential()
model.add(keras.layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
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
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=5)

model.save('model.h5')
            

