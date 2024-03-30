Code Breakdown:

    Imports:
        tensorflow as tf: Imports the TensorFlow library for deep learning.
        from tensorflow.keras import layers, models: Imports layers and model building components from Keras, a high-level API on top of TensorFlow.
        import numpy as np: Imports NumPy for numerical computations.
        import matplotlib.pyplot as plt: Imports Matplotlib for plotting training metrics.

    Dummy Data:
        X_train: Creates a random NumPy array representing 1000 training images, each with a shape of (28, 28, 1) (likely grayscale images).
        y_train: Creates a random NumPy array representing labels (categories) for the training images, with 10 possible categories.

    Model Definition:
        model = models.Sequential(): Creates a sequential model, where layers are added one after the other.
        Layers are defined using layers.Conv2D, layers.MaxPooling2D, layers.Flatten, and layers.Dense. These represent convolutional layers, pooling layers, a flattening layer, and fully-connected layers, respectively. The specific configuration defines a Convolutional Neural Network (CNN) architecture.

    Model Compilation:
        model.compile(): Sets up the training process by specifying the optimizer (adam), loss function (sparse_categorical_crossentropy for multi-class classification), and metrics (accuracy).

    Training Loop:
        Defines batch_size and epochs for training.
        Initializes empty lists train_losses and train_accuracies to store training metrics.
        Iterates through epochs:
            Prints the current epoch number.
            Initializes empty lists epoch_losses and epoch_accuracies for the current epoch.
            Iterates through training data in batches using a custom loop:
                Extracts a batch of training images and labels using slicing.
                Trains the model on the batch using model.train_on_batch(), which returns loss and accuracy values.
                Appends the loss and accuracy to the epoch lists.
                Prints batch-level metrics (loss and accuracy).
            Calculates average loss and accuracy for the epoch.
            Appends the averaged metrics to the training lists.
            Plots the training loss and accuracy curves using Matplotlib.

    Model Evaluation (for Demonstration):
        Evaluates the trained model on the same training data (not ideal for real-world scenarios) using model.evaluate(). This gives test loss and accuracy values.


