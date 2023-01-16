from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

# pregătire set de date
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory('/Users/dragosflow/UVT/IA/cats_and_dogs_small/train',
                                                 target_size=(150, 150),
                                                 batch_size=20,
                                                 class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('/Users/dragosflow/UVT/IA/cats_and_dogs_small/test',
                                            target_size=(150, 150),
                                            batch_size=20,
                                            class_mode='binary')

# construire model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# antrenare model
model.fit_generator(training_set,
                    steps_per_epoch=100,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=50)

# salvare model
model.save('model.h5')
