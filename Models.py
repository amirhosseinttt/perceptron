import numpy as np
from sklearn.metrics import mean_squared_error


class Perceptron:
    _weights = None
    _bias = 1

    def __init__(self, input_size):
        self._weights = np.zeros(input_size + 1)  # 1 is for bias

    @staticmethod
    def activation_function(x):  # step function
        # return max(0, x)
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        if len(inputs.shape) == 1:
            inputs = np.array([inputs])

        if len(inputs.shape) != 2:
            raise Exception(f'Inputs shape {inputs.shape} is not supported')

        if inputs.shape[1] != self._weights.shape[0] - 1:
            raise Exception(f'Inputs shape {inputs.shape} does not match weights shape {self._weights.shape}')

        model_inputs = np.c_[np.tile(self._bias, inputs.shape[0]), inputs]

        weighted_sum = np.sum(model_inputs * self._weights, axis=1)

        return [self.activation_function(i) for i in weighted_sum]

    def train(self, inputs, labels, epochs=10, learning_rate=0.1):
        for epoch_number in range(epochs):
            error_sum = 0
            for sample, label in zip(inputs, labels):
                prediction = self.predict(sample)[0]
                error = (label - prediction)
                error_sum += abs(error)
                self._weights += learning_rate * error * np.array([self._bias, *sample])

            print(f'Epoch: {epoch_number}, Error: {error_sum}')

        # print(self._weights)

    def get_weights(self):
        return self._weights

    def get_bias(self):
        return self._bias

class Adaline:
    _weights = None
    _bias = 1

    def __init__(self, input_size):
        self._weights = np.zeros(input_size + 1)  # 1 is for bias

    @staticmethod
    def activation_function(x):  # step function
        # return max(0, x)
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        if len(inputs.shape) == 1:
            inputs = np.array([inputs])

        if len(inputs.shape) != 2:
            raise Exception(f'Inputs shape {inputs.shape} is not supported')

        if inputs.shape[1] != self._weights.shape[0] - 1:
            raise Exception(f'Inputs shape {inputs.shape} does not match weights shape {self._weights.shape}')

        model_inputs = np.c_[np.tile(self._bias, inputs.shape[0]), inputs]

        weighted_sum = np.sum(model_inputs * self._weights, axis=1)
        # print(model_inputs, self._weights, weighted_sum)

        return weighted_sum, np.array([self.activation_function(i) for i in weighted_sum])

    def train(self, inputs, labels, epochs=10, learning_rate=0.001):
        for epoch_number in range(epochs):
            error_sum = 0
            for sample, label in zip(inputs, labels):
                reg_pred, cls_pred = self.predict(sample)[0][0], self.predict(sample)[1][0]
                # print(reg_pred, label)
                error = (label - reg_pred)
                error_sum += abs(error)
                self._weights += learning_rate * error * np.array([self._bias, *sample])
                # self._weights[1:] += learning_rate * sample.T.dot(error).reshape(self._weights[1:].shape)

            print(f'Epoch: {epoch_number}, Error: {error_sum}')

        # print(self._weights)

    def get_weights(self):
        return self._weights

    def get_bias(self):
        return self._bias

class CustomModel:
    adaline = None
    perceptron = None

    def __init__(self, input_size):
        self.adaline = Adaline(input_size)
        self.adaline._bias = 0
        self.perceptron = Perceptron(input_size)

    def predict(self, inputs):
        _, cls_pred = self.adaline.predict(inputs)

        output = []
        for pred in cls_pred:
            self.perceptron._bias = pred
            pereceptron_pred = self.perceptron.predict(inputs)[0]
            output.append(pereceptron_pred)
        return np.array(output)

    def train(self, inputs, labels, epochs=10, learning_rate=0.001):
        for epoch_number in range(epochs):
            for sample, label in zip(inputs, labels):
                self.adaline.train([sample], [label], 1, learning_rate)
                self.perceptron._bias = self.adaline.predict(sample)[1][0]
                self.perceptron.train([sample], [label], 1, learning_rate)

    def get_weights(self, perceptron=True):
        if perceptron:
            return self.perceptron._weights
        return self.adaline._weights

    def get_bias(self, perceptron=True):
        if perceptron:
            return self.perceptron._bias
        return self.adaline._bias
