import numpy as np


class Perceptron:
    _weights = None
    _learning_rate = None

    def __init__(self, input_size, learning_rate=0.01):
        self._weights = np.zeros(input_size)
        self._learning_rate = learning_rate

    def activation_function(self, x):
        return max(0, x)
        # return 1 if x >= 0 else 0

    def predict(self, inputs):
        weighted_sum = np.sum(np.multiply(inputs, self._weights))
        return self.activation_function(weighted_sum)

    def train(self, inputs, labels, epochs=10):
        for epoch_number in range(epochs):
            error_sum = 0
            for inputs_row, label in zip(inputs, labels):
                prediction = self.predict(inputs_row)
                error = (label - prediction)
                error_sum += abs(error)
                self._weights += self._learning_rate * error * inputs_row

            print(f'Epoch: {epoch_number}, Error: {error_sum}')
