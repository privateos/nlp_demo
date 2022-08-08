from __future__ import unicode_literals
from sklearn.datasets import load_diabetes
import numpy as np
import utils

dataset = load_diabetes()
X = dataset.data
Y = dataset.target
# print(X.shape, Y.shape)
# print(dataset.DESCR)
Y = np.reshape(Y, (Y.shape[0], 1))

train_ratio = 0.8
N = X.shape[0]
train_num = int(N*train_ratio)

X_train = X[0:train_num]
Y_train = Y[0:train_num]
X_test = X[train_num:]
Y_test = Y[train_num:]



x_mean = np.mean(X_train, axis=0)
x_std = np.std(X_train, axis=0)
X_train = (X_train - x_mean)/x_std
X_test = (X_test - x_mean)/x_std

y_mean = np.mean(Y_train, axis=0)
y_std = np.std(Y_train, axis=0)
Y_train = (Y_train - y_mean)/y_std
Y_test = (Y_test - y_mean)/y_std

batch_size = 3
epochs = 20
for epoch in range(epochs):
    train_iter = utils.get_batch_data(X_train, Y_train, batch_size=batch_size)
    for x, y in train_iter:
        pass

    test_iter = utils.get_batch_data(X_test, Y_test, batch_size=batch_size)
    for x, y in test_iter:
        pass
