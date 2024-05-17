import tensorflow as tf
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

from src.utils import preprocess_data, create_model, evaluate_model, make_predictions
from src.visualization import plot_predictions, plot_prediction_distribution

# Import TensorFlow
tf.get_logger().setLevel('ERROR')
print('Using TensorFlow version', tf.__version__)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)

# Preprocess the data
x_train_norm, x_test_norm, y_train_encoded, y_test_encoded = preprocess_data(
    x_train, x_test, y_train, y_test
)

# Create the model
model = create_model()

# Train the model
model.fit(x_train_norm, y_train_encoded, epochs=25)

# Evaluate the model
loss, accuracy = evaluate_model(model, x_test_norm, y_test_encoded)
print('Test set accuracy: ', accuracy * 100)

# Make predictions on the test set
preds = make_predictions(model, x_test_norm)
print("Shape of preds: ", preds.shape)

# Visualize predictions on sample images
plot_predictions(preds, x_test, y_test, start_index=0, num_images=25)

# Plot probability distribution of a specific prediction
index = 8  # Choose an index to visualize
plot_prediction_distribution(preds, index)
