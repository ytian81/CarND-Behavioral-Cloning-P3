from keras.models import Sequential
from keras.layers import Dense, Flatten
from scipy import ndimage

import csv
import numpy as np

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

X_train = np.array(images)
y_train = np.array(steering_angles)

# Train and save model
model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('naive_model.h5')
