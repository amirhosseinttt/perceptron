import numpy as np


class Perceptron:
    _weights = None

    def __init__(self, input_size):
        self._weights = np.zeros(input_size)

    @staticmethod
    def activation_function(x):
        # return max(0, x)
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # check if inputs is a 2d numpy array or 1d
        if len(inputs.shape) == 1:
            inputs = np.array([inputs])

        if len(inputs.shape) != 2:
            raise Exception(f'Inputs shape {inputs.shape} is not supported')

        if inputs.shape != self._weights.shape:
            raise Exception(f'Inputs shape {inputs.shape} does not match weights shape {self._weights.shape}')
        weighted_sum = np.sum(np.multiply(inputs, self._weights))
        return self.activation_function(weighted_sum)



    def train(self, inputs, labels, epochs=10, learning_rate=0.1):
        for epoch_number in range(epochs):
            error_sum = 0
            for inputs_row, label in zip(inputs, labels):
                prediction = self.predict(inputs_row)
                error = (label - prediction)
                error_sum += abs(error)
                self._weights += learning_rate * error * inputs_row

            print(f'Epoch: {epoch_number}, Error: {error_sum}')
