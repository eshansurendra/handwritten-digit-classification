## src Folder Structure

The `src` folder contains the core Python code for the MNIST handwritten digit image classification project. This folder is structured as follows:

* **`__init__.py`:** This is an empty file that tells Python to treat the `src` directory as a package. This allows for better code organization and makes it easier to import modules from within the package.

* **`main.py`:** This is the primary script for the project. It orchestrates the entire image classification workflow, including:
    * **Data Loading and Preprocessing:**  Loads the MNIST dataset and performs necessary preprocessing steps, such as normalization and one-hot encoding of labels.
    * **Model Creation and Compilation:** Defines the architecture of the neural network model and configures it with an optimizer, loss function, and metrics.
    * **Model Training:** Trains the model on the preprocessed training data.
    * **Model Evaluation:** Evaluates the trained model's performance on the test dataset.
    * **Predictions and Visualizations:** Makes predictions on new data and visualizes the results to understand the model's decision-making process.

* **`utils.py`:** This file contains helper functions to keep the code clean and organized. Here's a summary of the functions included:
    * **`preprocess_data()`:** Preprocesses the MNIST dataset by normalizing the image data and one-hot encoding the labels.
    * **`create_model()`:** Creates the structure of the neural network model, including specifying the layers, activation functions, and input shape.
    * **`evaluate_model()`:** Evaluates the trained model on the test data, calculating metrics like accuracy and loss.
    * **`make_predictions()`:** Generates predictions on new data using the trained model.

* **`visualization.py`:**  This file provides functions for creating visualizations of the model's results:
    * **`plot_predictions()`:** Visualizes the model's predictions on sample images, highlighting correct and incorrect classifications.
    * **`plot_prediction_distribution()`:**  Plots the probability distribution for a single prediction, showing the model's confidence in each possible digit class.
