# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from IPython.display import display
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten,BatchNormalization,MaxPool2D

# Part 1 - Data Preprocessing


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,                                
                                   horizontal_flip = True)

# Preprocessing the Training set
training_set = train_datagen.flow_from_directory("dataset\Training_set",
                                                 target_size = (100, 100),
                                                 batch_size = 64,
                                                 color_mode = "grayscale",
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_set = train_datagen.flow_from_directory("dataset\Testing_set",
                                            target_size = (100, 100),
                                            batch_size = 64,
                                            color_mode = "grayscale",
                                            class_mode = 'binary')

training_set.class_indices

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()


#Adding first convolutional layer
# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=[100, 100, 1]))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

# Adding a third convolutional layer
cnn.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(Dropout(0.20))
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(Dropout(0.20))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 20)

cnn.evaluate(test_set)

# Saving CNN
cnn.save('cnn_mask.h5')

