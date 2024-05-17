# Handwritten Digit Classification with TensorFlow

This repository contains code for a guided project focused on Basic Digit Image Classification with TensorFlow, hosted by the Coursera Project Network.

![Handwritten Digit Classification with TensorFlow](/jupyter-notebook/images/1_1.png)

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
3. **Run the Code:** Follow the instructions provided within each script to:
   - Preprocess the data.
   - Create and train your neural network model.
   - Evaluate the model's performance.
   - Visualize predictions.

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

The project uses the well-known MNIST dataset, which consists of handwritten digit images (0-9). This dataset is included within TensorFlow's `keras.datasets` module.

## Getting Started

1. **Open `main.py`:** Launch the script in your preferred Python environment.
2. **Run the Code:** Execute the script to step through the project's tasks, including:
   - Loading and preparing the MNIST dataset.
   - Creating and training a neural network model.
   - Evaluating the trained model.
   - Visualizing predictions.

## Contributions

Contributions to this project are welcome! Feel free to open an issue or submit a pull request for improvements, bug fixes, or new features.

## Acknowledgements

This project is based on the "Basic Image Classification with TensorFlow" guided project provided by the Coursera Project Network. Thanks to Coursera for providing this learning opportunity.

## Resources

- Coursera Project Link: [Basic Image Classification with TensorFlow](https://www.coursera.org/projects/tensorflow-beginner-basic-image-classification)
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
