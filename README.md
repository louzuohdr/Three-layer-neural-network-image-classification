Neural Network Training and Visualization
This repository contains a simple neural network implementation in Python using NumPy for the backpropagation algorithm. The neural network is trained on the Fashion MNIST dataset and includes functions for training, evaluation, and visualization of the model's weights.

Overview
This project implements a basic neural network with one hidden layer to classify images from the Fashion MNIST dataset. The neural network architecture consists of an input layer with 784 neurons, a hidden layer with 128 neurons, and an output layer with 15 neurons representing different fashion categories. The model is trained using backpropagation with ReLU activation in the hidden layer and softmax activation in the output layer. Regularization with L2 penalty is applied to prevent overfitting.

Files
neural_network.py: Python script containing the neural network class and training functions.
input_size=784, hidden_size=128, output_size=15, reg_lambda=0.01, epochs=30, learning_rate=0.1.npy: Numpy file containing the trained model parameters.
README.txt: Instructions and information about the project.
Usage
Install the required libraries by running pip install numpy matplotlib.
Run the neural_network.py script to train the neural network and visualize the results.
Evaluate the model performance on the test set and view the training history.
Training
The neural network is trained using stochastic gradient descent with mini-batch processing. The model is trained for 30 epochs with a learning rate of 0.1 and a regularization parameter of 0.01.

Evaluation
After training, the model is evaluated on the test set to calculate the accuracy of the predictions. The evaluation results are displayed in the console.

Visualization
The script includes functions to visualize the training loss and accuracy over epochs using matplotlib. Additionally, you can visualize the learned weights of the first layer to understand how the model learns features from the input data.

