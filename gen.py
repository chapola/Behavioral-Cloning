import csv
import cv2
import numpy as np
import os
import sklearn
from random import shuffle



samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)
del(samples [0])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples,test_size=0.2)

def generator(samples,batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            correction =0.2


            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                source_path_left = batch_sample[1]
                source_path_right = batch_sample[2]
                filename = source_path.split('/')[-1]
                filename_left = source_path_left.split('/')[-1]
                filename_right = source_path_right.split('/')[-1]

                current_path = 'data/IMG/'+filename
                current_path_left = 'data/IMG/'+filename_left
                current_path_right = 'data/IMG/'+filename_right
                
                image = cv2.imread(current_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_left = cv2.imread(current_path_left)
                image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
                image_right = cv2.imread(current_path_right)
                image_right = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                images.append(image)
                images.append(image_left)
                images.append(image_right)

                measurement = float(batch_sample[3])
                measurement_left = measurement + correction
                measurement_right = measurement - correction
                measurements.append(measurement)
                measurements.append(measurement_left)
                measurements.append(measurement_right)


            augmented_images, augmented_measurements =[],[]

            for image, measurement in zip(images,measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement * (-1.0))


    X_train= np.array(augmented_images)
    y_train= np.array(augmented_measurements)
    yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from  keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

ch, row, col = 5, 80, 320

model = Sequential();
model.add(Lambda(lambda x: x/255.0 - 0.5,
        input_shape=( row, col,ch),
        output_shape=( row, col,ch)))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=3)



model.save('model.h5')
