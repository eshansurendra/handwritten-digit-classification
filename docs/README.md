# Handwritten Digit Classification with TensorFlow

This repository contains code for a guided project focused on Basic Digit Image Classification with TensorFlow, hosted by the Coursera Project Network.

![Handwritten Digit Classification with TensorFlow](/jupyter-notebook/images/1_1.png)

> [!TIP]
> For a more comprehensive understanding of the implementation, refer this [notebook on Kaggle](https://www.kaggle.com/code/eshansurendra/handwritten-digit-classification-with-tensorflow)



## Project Overview

The project is structured into several stages, each addressing a specific aspect of image classification with TensorFlow:

- **Encoding Labels:** Transforming categorical labels into a format suitable for neural network training.
- **Neural Networks:** Understanding the fundamental principles of neural networks, including architecture, activation functions, and learning mechanisms.
- **Image Preprocessing:** Applying essential preprocessing techniques to image data for optimal model performance.
- **Creating a Model:** Defining the architecture of your neural network model using TensorFlow's Keras API.
- **Model Training:** Training your model on the prepared dataset to learn patterns and classify images.
- **Evaluation:** Assessing the trained model's performance using a separate test dataset and various evaluation metrics.
- **Visualization:** Visualizing the model's predictions on sample images to understand its decision-making process.

## Setup

1. **Clone the Repository:** Download the project files to your local machine.
   ```bash 
    git clone https://github.com/eshansurendra/handwritten-digit-classification    
    ```
2. **Install Dependencies:** Ensure you have Python installed, along with the following libraries:
   - **TensorFlow:**  `pip install tensorflow`
   - **Keras:** (Included in TensorFlow)
   - **NumPy:**  `pip install numpy`
   - **Matplotlib:**  `pip install matplotlib`

## Getting Started

1. **Open `/src/main.py`:** Launch the script in your preferred Python environment.
2. **Run the Code:** Execute the script to step through the project's tasks, including:
   - Loading and preparing the MNIST dataset.
   - Creating and training a neural network model.
   - Evaluating the trained model.
   - Visualizing predictions.

## Code Structure

The repository is organized into the following directories:

- **`src`:** Contains the core Python code for the project.
    - **`main.py`:**  The primary script that orchestrates the entire image classification workflow, including data loading, preprocessing, model training, evaluation, and visualization.
    - **`utils.py`:**  Holds helper functions for data preprocessing, model building, evaluation, and prediction.
    - **`visualization.py`:**  Provides functions for creating visualizations, such as plots of predictions and probability distributions.
- **`jupyter-notebook`:** Contains a Jupyter Notebook file for interactive exploration and experimentation with the code.
    - **`handwritten-digit-classification.ipynb`:** This notebook allows you to interact with the code, visualize results, and document your steps.
- **`docs`:** Contains project documentation.
    - **`README.md`:** This file provides an overview of the repository, instructions for setup and usage, and other relevant information.
    - **`requirements.txt`:** Lists the Python packages required to run the project.

## Dataset

The project utilizes the widely recognized MNIST dataset, a collection of handwritten digit images (0-9) that has become a standard benchmark for image classification tasks. This dataset is readily available within TensorFlow's `keras.datasets` module, making it easy to load and use for training and evaluating models.

![MNIST Dataset](/docs/asests/MNIST-dataset.png)

**Key Features of the MNIST Dataset:**

* **Size:** It contains a total of 70,000 images, split into 60,000 training images and 10,000 testing images.
* **Image Format:** Each image is a grayscale image with a resolution of 28x28 pixels.
* **Labels:** Each image is labeled with its corresponding digit (0-9).
* **Accessibility:** It is readily accessible through TensorFlow's `keras.datasets` module, simplifying data loading and usage.

**Why MNIST is Used:**

* **Simplicity:** The dataset is relatively small and straightforward to work with, making it ideal for beginners learning about image classification.
* **Well-defined task:** The task of classifying handwritten digits is well-understood and provides a clear objective for model training and evaluation.
* **Benchmarked performance:**  It has been extensively used in research and development, providing a common ground for comparing different models and algorithms.

The MNIST dataset's ease of use, clear task definition, and established benchmark nature make it an excellent choice for this image classification project. 

## Neural Networks

### Single Neuron with 784 Features

![Single Neuron with 784 Features](/jupyter-notebook/images/1_3.png)

This diagram illustrates a single neuron with 784 inputs, representing the pixels of a flattened MNIST image (28x28 = 784). While this approach is simple, it's not powerful enough to learn complex patterns for handwritten digit recognition.

### Neural Network with Two Hidden Layers

![Neural Network with Two Hidden Layers](/jupyter-notebook/images/1_4.png)

This diagram shows a more complex neural network with two hidden layers. These layers, consisting of multiple neurons, allow the network to learn intricate relationships between the input pixels and the corresponding digits. This architecture is much more effective at classifying handwritten digits.

## Contributions

Contributions to this project are welcome! Feel free to open an issue or submit a pull request for improvements, bug fixes, or new features.

## Acknowledgements

This project builds upon the foundational knowledge and structure provided by the "Basic Image Classification with TensorFlow" guided project hosted by the Coursera Project Network. We express our gratitude to Coursera for offering this valuable learning opportunity.

We also acknowledge the invaluable contribution of the MNIST dataset, originally compiled by Yann LeCun, Corinna Cortes, and Christopher Burges. 

**References:**

* Deng, L. (2012). The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6), 141–142.

**Resources:**

* TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
* Coursera Project Link: [Basic Image Classification with TensorFlow](https://www.coursera.org/projects/tensorflow-beginner-basic-image-classification)

