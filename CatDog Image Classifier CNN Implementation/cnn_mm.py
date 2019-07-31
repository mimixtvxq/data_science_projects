# -*- coding: utf-8 -*-

# Convolutional Neural Networks
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialise the CNN
classifier = Sequential()

# Add the convolutional layer
#32 feature detectors with 3X3 dim
classifier.add(Conv2D(32, (3, 3),
                             input_shape = (64,64,3),
                             activation = 'relu'))

# Pooling 
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Flattening 
classifier.add(Flatten())

#Full connection (128 hidden nodes)
classifier.add(Dense(output_dim = 128, activation = 'relu')) #hidden
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #output

#compile CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Image preprocessing https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=50,
        validation_data=test_set,
        validation_steps=2000)

classifier.save('cat_dog_cnn.h5')