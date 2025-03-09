# Importing the necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

# Loading the dataset while augmenting the images

# Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Building the CNN
#  first Convolution layer
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Input(shape=[64,64,3]))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) # pooling layer using maxpool
# second convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# flattening
cnn.add(tf.keras.layers.Flatten())
# full connection
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
# output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Making a single prediction
def predict_img(image_path):
    test_image = image.load_img(path=image_path, target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    
    return 'dog' if result[0][0] else 'cat'

# Predicting the images
predict_img('dataset/single_prediction/cat_or_dog_1.jpg')

