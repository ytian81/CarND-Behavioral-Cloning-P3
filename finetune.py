from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPool2D
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from scipy import ndimage
from sklearn.utils import shuffle

import csv
import matplotlib.pyplot as plt
import numpy as np

# data_folder = './data/'
data_folder = './Track2/'

def get_data():
    # Read driving log data from csv file
    lines = []
    with open(data_folder+'/driving_log.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)

    # Modify image path and extract outputs
    images = []
    steering_angles = []
    delta = 0.2
    for line in lines:
        # Use center, left and right images
        angle_corrections = [0.0, delta, -delta]
        for idx in range(3):
            image_path = line[idx]
            image_path = data_folder + '/IMG/' + image_path.split('/')[-1]
            image = ndimage.imread(image_path)
            images.append(image)
            steering_angle = float(line[3]) + angle_corrections[idx]
            steering_angles.append(steering_angle)

            # Augment data (double the amount of data)
            images.append(np.fliplr(image))
            steering_angles.append(-steering_angle)

    images = np.array(images)
    steering_angles = np.array(steering_angles)

    # shuffle data before split validation set
    images, steering_angles = shuffle(images, steering_angles)
    return images, steering_angles

X_train, y_train = get_data()

def assemble_model():
    # Assemble model
    input_shape = (160, 320, 3)
    regularizer_coef = 1e-3

    model = Sequential()
    model.add(Lambda(lambda x:(x / 255.0)-0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    # Conv layer 1
    model.add(Conv2D(filters=6, kernel_size=5, strides=1, padding='valid',
        kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=None, padding='valid'))

    # Conv layer 2
    model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='valid',
        kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=None, padding='valid'))

    # Flatten
    model.add(Flatten())

    # Fully connected layer 3
    model.add(Dense(120, kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully connected layer 4
    model.add(Dense(84, kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully connected layer 5
    model.add(Dense(1, kernel_regularizer=l2(regularizer_coef)))

    return model

model = assemble_model()
model.load_weights('best_model.h5')

model.compile(loss='mse', optimizer='adam')
# Train 15 epoches at most and save the best model, early stop if validation loss stops improving
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=0)
earlystop = EarlyStopping(monitor='val_loss', patience=3, mode='min')
history_object = model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=15,
        callbacks=[checkpoint, earlystop])

# Draw training statistics
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('training_stats.jpg')
