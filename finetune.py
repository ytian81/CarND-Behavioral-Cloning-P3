from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPool2D
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from scipy import ndimage
from sklearn.utils import shuffle
from model import assemble_model

import csv
import matplotlib.pyplot as plt
import numpy as np

# data_folder = './data/'
#  data_folder = './Track2/'
data_folder = './turn/'

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

model = assemble_model()
model.load_weights('batch_128_model.h5')

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
