import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def preprocess_data(x_train, x_test, y_train, y_test):
    """
    Preprocess the MNIST dataset.

    Args:
        x_train: Training images.
        x_test: Testing images.
        y_train: Training labels.
        y_test: Testing labels.

    Returns:
        Tuple of preprocessed data:
            x_train_norm: Normalized training images.
            x_test_norm: Normalized testing images.
            y_train_encoded: One-hot encoded training labels.
            y_test_encoded: One-hot encoded testing labels.
    """
    x_train_reshaped = np.reshape(x_train, (60000, 784))
    x_test_reshaped = np.reshape(x_test, (10000, 784))
    x_mean = np.mean(x_test_reshaped)
    x_std = np.std(x_test_reshaped)
    epsilon = 1e-10
    x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
    x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)
    return x_train_norm, x_test_norm, y_train_encoded, y_test_encoded

def create_model():
    """
    Creates a simple neural network model.

    Returns:
        A compiled TensorFlow model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def evaluate_model(model, x_test_norm, y_test_encoded):
    """
    Evaluates the model's performance on the test set.

    Args:
        model: The trained model.
        x_test_norm: Normalized testing images.
        y_test_encoded: One-hot encoded testing labels.

    Returns:
        Tuple of loss and accuracy.
    """
    loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
    return loss, accuracy

def make_predictions(model, x_test_norm):
    """
    Makes predictions on the test set.

    Args:
        model: The trained model.
        x_test_norm: Normalized testing images.

    Returns:
        Array of predictions.
    """
    preds = model.predict(x_test_norm)
    return preds
