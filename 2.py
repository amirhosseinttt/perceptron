from Models import Perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('linear.csv')

train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)

train_inputs = train_df[['0', '1']].values
train_labels = train_df['y'].values

test_inputs = test_df[['0', '1']].values
test_labels = test_df['y'].values

perceptron = Perceptron(2)

perceptron.train(train_inputs, train_labels, epochs=10)

for inputs, label in zip(test_inputs, test_labels):
    prediction = perceptron.predict(inputs)
    print(f'inputs: {inputs}, label: {label}, prediction: {prediction}')

# plot the decision boundary for the test data based on the trained model
x1 = np.linspace(3, 12, 100)
x2 = np.linspace(-6, 10, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.c_[xx1.ravel(), xx2.ravel()]
Z = np.array([perceptron.predict(x) for x in xx])
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.5)
plt.scatter(test_inputs[:, 0], test_inputs[:, 1], c=test_labels)
plt.show()

