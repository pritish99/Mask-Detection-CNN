# Mask-Detection-CNN

This project was made with the intent of creating a deep learning application aimed at classifying whether the person infront of the camera has worn a mask or not.

The training was done with Cnn algorithm where the data input was of target size (100,100) pixels in grayscale which gives the final input dimension of (100,100,1)

Convolution layers: 3 layers with filter sizes of 5 and 3 with a stride of 1.

Units in Full Connection: 256

Activation function: ReLu for input and hidden layers and Sigmoid for output layer

Optimizer: Adam

Loss funtion: Binary Crossentropy
