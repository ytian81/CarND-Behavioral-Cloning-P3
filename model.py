from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from math import ceil
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    # Read driving log data from csv file, augment data by lazy flipping and using left and right images
    samples = []
    delta = 0.2
    angle_corrections = [0.0, delta, -delta]
    with open('./data/driving_log.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            # Use center, left and right images
            for idx in range(3):
                # Original image
                sample = [line[idx], (float(line[3])+angle_corrections[idx]), False]
                samples.append(sample)
                # Flipped image
                sample = [line[idx], -(float(line[3])+angle_corrections[idx]), True]
                samples.append(sample)
    return samples

# Shuffle and split training/validation data
validation_split = 0.3
train_samples, validation_samples = train_test_split(get_data(), test_size = validation_split,
        shuffle=True)

shuffle_before_epoch = True
def data_generator(samples, batch_size):
    # Modify image path and extract images. Flip image if requested
    num_samples = len(samples)

    while True:
        if shuffle_before_epoch:
            shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering_angles = []
            for sample in batch_samples:
                image_path = './data/IMG/' + sample[0].split('/')[-1]
                image = ndimage.imread(image_path)
                # Flip image
                if (sample[2]):
                    image = np.fliplr(image)
                images.append(image)
                steering_angles.append(sample[1])

            yield np.array(images), np.array(steering_angles)

batch_size = 32
train_generator = data_generator(train_samples, batch_size)
validation_generator = data_generator(validation_samples, batch_size)

def assemble_model():
    # Assemble model
    input_shape = (160, 320, 3)
    regularizer_coef = 1e-3

    model = Sequential()
    model.add(Lambda(lambda x:(x / 255.0)-0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    # Conv layer 1
    model.add(Conv2D(filters=24, kernel_size=5, strides=2, padding='valid',
        kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))
    #  model.add(MaxPool2D(pool_size=2, strides=None, padding='valid'))

    # Conv layer 2
    model.add(Conv2D(filters=36, kernel_size=5, strides=2, padding='valid',
        kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))
    #  model.add(MaxPool2D(pool_size=2, strides=None, padding='valid'))

    # Conv layer 3
    model.add(Conv2D(filters=48, kernel_size=5, strides=2, padding='valid',
        kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))

    # Conv layer 4
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='valid',
        kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))

    # Conv layer 5
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='valid',
        kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))

    # Flatten
    model.add(Flatten())

    # Fully connected layer 6
    model.add(Dense(100, kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully connected layer 7
    model.add(Dense(50, kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully connected layer 8
    model.add(Dense(10, kernel_regularizer=l2(regularizer_coef)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully connected layer 9
    model.add(Dense(1, kernel_regularizer=l2(regularizer_coef)))

    return model

model = assemble_model()

# Train and save model
adam = Adam(lr=5e-4)
model.compile(loss='mse', optimizer=adam)
# Train 15 epoches at most and save the best model, early stop if validation loss stops improving
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=0)
earlystop = EarlyStopping(monitor='val_loss', patience=3, mode='min')
history_object = model.fit_generator(train_generator,
        steps_per_epoch=ceil(len(train_samples)/batch_size),
        validation_data=validation_generator,
        validation_steps=ceil(len(validation_samples)/batch_size),
        epochs=15,
        callbacks=[checkpoint, earlystop],
        verbose=1)

# Draw training statistics
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('training_stats.jpg')
