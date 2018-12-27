from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Conv2D, Dense, Flatten, Lambda, MaxPool2D
from keras.models import Sequential
from scipy import ndimage

import csv
import numpy as np

def get_data():
    # Read driving log data from csv file
    lines = []
    with open('./data/driving_log.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)

    # Modify image path and extrac outputs
    images = []
    steering_angles = []
    for line in lines:
        image_path = line[0]
        image_path = './data/IMG/' + image_path.split('/')[-1]
        image = ndimage.imread(image_path)
        images.append(image)

        steering_angle = float(line[3])
        steering_angles.append(steering_angle)

    return np.array(images), np.array(steering_angles)

X_train, y_train = get_data()

def assemble_model():
    # Assemble model
    input_shape = (160, 320, 3)

    model = Sequential()
    model.add(Lambda(lambda x:(x / 255.0)-0.5, input_shape=input_shape))

    # Conv layer 1
    model.add(Conv2D(filters=6, kernel_size=5, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=None, padding='valid'))

    # Conv layer 2
    model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=None, padding='valid'))

    # Flatten
    model.add(Flatten())

    # Fully connected layer 3
    model.add(Dense(120))
    model.add(Activation('relu'))

    # Fully connected layer 4
    model.add(Dense(84))
    model.add(Activation('relu'))

    # Fully connected layer 5
    model.add(Dense(1))

    return model

model = assemble_model()

# Train and save model
model.compile(loss='mse', optimizer='adam')
# Train 10 epoches and save the best model

checkpoint = ModelCheckpoint('naive_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2, callbacks=[checkpoint])
