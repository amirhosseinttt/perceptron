

# using sklearn make moon method to generate 1000 sample data
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from Models import Perceptron

# generate 1000 sample data
X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
# print(X)
# print(y)

# plot the data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# split the data into train and test
train_X = X[:800]
train_y = y[:800]

test_X = X[800:]
test_y = y[800:]

# train the model
perceptron = Perceptron(2)
perceptron.train(train_X, train_y, epochs=10)

# test the model
for inputs, label in zip(test_X, test_y):
    prediction = perceptron.predict(inputs)
    print(f'inputs: {inputs}, label: {label}, prediction: {prediction}')

# plot the decision boundary
x1 = np.linspace(-2, 2.5, 100)
x2 = np.linspace(-2, 2, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.c_[xx1.ravel(), xx2.ravel()]
Z = np.array([perceptron.predict(x) for x in xx])
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

