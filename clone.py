import csv
import cv2
import numpy as np



lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
del(lines [0])
images = []
measurements = []
correction =0.2


for line in lines:
    source_path = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
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

    measurement = float(line[3])
    measurement_left = measurement + 0.2
    measurement_right = measurement -0.2
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


from keras.models import Sequential
from  keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential();
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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
model.fit(X_train,y_train,validation_split=0.2, shuffle =True, nb_epoch=2,verbose=1)
model.save('model.h5')
