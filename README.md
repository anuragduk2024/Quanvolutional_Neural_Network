# Quantum vs Classical Neural Network: A Performance Comparison
Overview

This project compares the performance of a classical neural network with a quantum-enhanced neural network on image classification tasks. We aim to investigate whether the introduction of quantum processing can enhance the ability of neural networks to recognize complex patterns. Using a custom dataset of images, both models are trained and evaluated based on several metrics, including accuracy, loss, precision, and recall. After training, we visualize the models' performance over epochs and compare how well they perform in terms of classification metrics.
Project Structure

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
