import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class FeedForwardNN:
    def __init__(self, input_shape, hidden_layers, output_shape, learning_rate=0.001):
        """
        Initializes the feedforward neural network.

        :param input_shape: Number of input features.
        :param hidden_layers: List of integers where each integer denotes
                              the number of neurons in a layer.
        :param output_shape: Number of output neurons.
        :param learning_rate: Learning rate for training the network.
        """
        self.model = Sequential()
        # Input layer
        self.model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_shape,)))

        # Hidden layers
        for layer_size in hidden_layers[1:]:
            self.model.add(Dense(layer_size, activation='relu'))

        # Output layer
        self.model.add(Dense(output_shape, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Trains the neural network.

        :param X_train: Training data features.
        :param y_train: Training data labels.
        :param epochs: Number of epochs to train.
        :param batch_size: Batch size for training.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        """
        Makes predictions using the neural network.

        :param X: Data to make predictions on.
        :return: Predictions.
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the neural network performance on test data.

        :param X_test: Test data features.
        :param y_test: Test data labels.
        :return: Evaluation results.
        """
        return self.model.evaluate(X_test, y_test)

