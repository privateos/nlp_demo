from sklearn.datasets import load_diabetes
import numpy as np


diabetes = load_diabetes()
X = diabetes.data
Y = diabetes.target

print(X.shape, Y.shape)
print('X = ', X[0:3])
print('Y = ', Y[0:3])
# print(diabetes.DESCR)
Y = np.reshape(Y, (Y.shape[0], 1))


#数据划分
N = X.shape[0]
train_num = int(N*0.8)
X_train = X[0:train_num]
Y_train = Y[0:train_num]
X_test = X[train_num:]
Y_test = Y[train_num:]

#标准化

X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train - X_mean)/X_std
X_test = (X_test - X_mean)/X_std

Y_mean = np.mean(Y_train, axis=0)
Y_std = np.std(Y_train, axis=0)
Y_train = (Y_train - Y_mean)/Y_std
Y_test = (Y_test - Y_mean)/Y_std

print(X_train);input()
print(Y_train);input()

#



