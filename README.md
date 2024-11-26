# Quantum vs Classical Neural Network: A Performance Comparison
## Overview

In this project, we implement a Quanvolutional Neural Network (QNN), a quantum machine learning model originally introduced in Henderson et al. (2019). A QNN integrates quantum computing techniques into a convolutional neural network (CNN) to leverage the power of quantum mechanics for image classification tasks. Specifically, we explore the advantages of using quantum layers within a neural network to enhance its ability to capture and model complex patterns in image data, which may be difficult for classical neural networks to represent effectively.

We compare the performance of the quantum-enhanced QNN with a classical CNN on a standard image classification task. Both models are trained and evaluated based on key metrics like accuracy, loss, precision, and recall. A custom callback is used to track these metrics at the end of each epoch during training. After training, we visualize and compare the models' performance to assess whether the quantum layer improves the efficiency and accuracy of the classical network.

Through this comparison, we aim to understand how quantum-enhanced models, such as the Quanvolutional Neural Network, may offer advantages over traditional neural networks in machine learning tasks.
## Project Structure
This project consists of the following key sections:

1. Model Definition and Training: We define and train both a quantum-enhanced model (q_model) and a classical neural network (c_model).
2. Metrics Calculation: A custom callback function is used to calculate precision, recall, and F1 score during training for both models.
3. Performance Visualization: After training both models, the performance metrics (accuracy, loss, precision, and recall) are plotted and compared for both models using matplotlib and seaborn.
4. Conclusion: The performance of both models is evaluated, and the quantum-enhanced model’s potential advantages are analyzed based on the visualization.

## Requirements

To run this project, you'll need to install the following Python libraries:

TensorFlow: For building and training the neural network models.

Keras: For simplifying the model-building process.

NumPy: For handling arrays and matrices.

Matplotlib: For plotting the training progress.

Seaborn: For enhanced data visualization.

Scikit-learn: For calculating classification metrics like precision, recall, and F1 score.

## Performance Comparison

Once the training is complete, the resulting plots will show the following comparisons:

  * Validation Accuracy: This plot shows how accurately each model classifies the test set.
  * Validation Loss: This plot shows how well each model fits the data.
  * Validation Precision: The precision metric is evaluated, showing how many of the positive predictions are correct.
  * Validation Recall: The recall metric shows how many of the actual positives are correctly identified.

By analyzing these plots, we can assess the advantages of the quantum-enhanced model over the classical one.

PennyLane: For quantum machine learning.

# Quantum Circuit for Quanvolutional Neural Network (QNN)
## Description:

The quantum circuit used in this Quanvolutional Neural Network (QNN) processes classical image data by applying quantum gates, specifically RY (rotation around the Y-axis), X (Pauli-X), and Z (Pauli-Z) gates. The RY and X gates are used to encode the classical data into quantum states and manipulate these states to extract features, while the Z gate is applied to measure the quantum states, extracting key information for classification tasks.

Though this example uses RY, X, and Z gates, the circuit design is flexible, allowing for the inclusion of other quantum gates based on the specific requirements of the task.
## Quantum Circuit Diagram:

Below is the diagram of the quantum circuit used in the QNN:
![qcproject circuit](https://github.com/user-attachments/assets/db793ad4-3d2d-4f69-a700-f1be0760d7c6)

## Quanvolutional Layer Overview:

The Quanvolutional layer processes the image data by applying multiple quantum operations over small regions (2x2 patches) of the image, using the quantum circuit described above. This process helps capture intricate features from the image that classical methods may miss, allowing the network to learn complex patterns effectively.

## Quanvolution Layer Diagram:

Below is the diagram representing the Quanvolutional layer in action, where each small region of the image is processed using the quantum circuit:
![quan01](https://github.com/user-attachments/assets/93fcce1d-88f0-4b50-a9ec-da2b466ac850)


## Conclusion

Based on the plots, we compare the performance of the quantum-enhanced model (q_model) with the classical model (c_model). The quantum model is expected to outperform the classical model in terms of accuracy, precision, and recall. This demonstrates the potential advantages of using quantum processing in deep learning models, particularly when dealing with complex data patterns.

