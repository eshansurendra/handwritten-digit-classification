from matplotlib import pyplot as plt
import numpy as np

def plot_predictions(preds, x_test, y_test, start_index=0, num_images=25):
    """
    Visualizes the model's predictions on sample images.

    Args:
        preds: Array of predictions.
        x_test: Testing images.
        y_test: Testing labels.
        start_index: Starting index for displaying images.
        num_images: Number of images to display.
    """
    plt.figure(figsize=(12, 12))

    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        pred = np.argmax(preds[start_index + i])
        gt = y_test[start_index + i]

        col = 'g'
        if pred != gt:
            col = 'r'
        plt.xlabel(
            'i={}, pred = {}, gt = {}'.format(
                start_index + i, pred, gt),
            color=col)
        plt.imshow(x_test[start_index + i], cmap='binary')

    plt.show()

def plot_prediction_distribution(preds, index):
    """
    Plots the probability distribution of a single prediction.

    Args:
        preds: Array of predictions.
        index: Index of the prediction to plot.
    """
    plt.plot(preds[index])
    plt.show()
