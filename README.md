# Quantum vs Classical Neural Network: A Performance Comparison
Overview

In this project, we implement a Quanvolutional Neural Network (QNN), a quantum machine learning model originally introduced in Henderson et al. (2019). A QNN integrates quantum computing techniques into a convolutional neural network (CNN) to leverage the power of quantum mechanics for image classification tasks. Specifically, we explore the advantages of using quantum layers within a neural network to enhance its ability to capture and model complex patterns in image data, which may be difficult for classical neural networks to represent effectively.

We compare the performance of the quantum-enhanced QNN with a classical CNN on a standard image classification task. Both models are trained and evaluated based on key metrics like accuracy, loss, precision, and recall. A custom callback is used to track these metrics at the end of each epoch during training. After training, we visualize and compare the models' performance to assess whether the quantum layer improves the efficiency and accuracy of the classical network.

Through this comparison, we aim to understand how quantum-enhanced models, such as the Quanvolutional Neural Network, may offer advantages over traditional neural networks in machine learning tasks.
This project consists of the following key sections:

Model Definition and Training: We define and train both a quantum-enhanced model (q_model) and a classical neural network (c_model).
Metrics Calculation: A custom callback function is used to calculate precision, recall, and F1 score during training for both models.
Performance Visualization: After training both models, the performance metrics (accuracy, loss, precision, and recall) are plotted and compared for both models using matplotlib and seaborn.
Conclusion: The performance of both models is evaluated, and the quantum-enhanced model’s potential advantages are analyzed based on the visualization.

Requirements

To run this project, you'll need to install the following Python libraries:

TensorFlow: For building and training the neural network models.
Keras: For simplifying the model-building process.
NumPy: For handling arrays and matrices.
Matplotlib: For plotting the training progress.
Seaborn: For enhanced data visualization.
Scikit-learn: For calculating classification metrics like precision, recall, and F1 score.
PennyLane: For quantum machine learning.

You can install the required dependencies using pip:
