from Models import Perceptron
import pandas as pd


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

