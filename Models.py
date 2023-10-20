import numpy as np


class Perceptron:
    _weights = None

    def __init__(self, input_size):
        self._weights = np.zeros(input_size)
        print(self._weights.shape)

    @staticmethod
    def activation_function(x):
        # return max(0, x)
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        if len(inputs.shape) == 1:
            inputs = np.array([inputs])

        if len(inputs.shape) != 2:
            raise Exception(f'Inputs shape {inputs.shape} is not supported')

        if inputs.shape[1] != self._weights.shape[0]:
            raise Exception(f'Inputs shape {inputs.shape} does not match weights shape {self._weights.shape}')

        weighted_sum = np.sum(inputs * self._weights, axis=1)

        return [self.activation_function(i) for i in weighted_sum]

    def train(self, inputs, labels, epochs=10, learning_rate=0.1):
        for epoch_number in range(epochs):
            error_sum = 0
            for sample, label in zip(inputs, labels):
                prediction = self.predict(sample)[0]
                error = (label - prediction)
                error_sum += abs(error)
                self._weights += learning_rate * error * sample
                # print(learning_rate * error * sample)

            print(f'Epoch: {epoch_number}, Error: {error_sum}')

        print(self._weights)
