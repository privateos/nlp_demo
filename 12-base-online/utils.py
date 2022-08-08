import numpy as np
from sklearn.datasets import load_diabetes
import torch
import torch.nn as nn
import math

#X.shape = (N, in_features)
#Y.shape = (N, out_features)
def get_batch_data(X, Y, batch_size=1, shuffle=False):
    N = X.shape[0]
    num_batch = N//batch_size
    if N%batch_size != 0:
        num_batch += 1
    
    indices = [i for i in range(N)]
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(num_batch):
        start_index = batch_size*i
        end_index = batch_size*(i + 1)
        indices_i = indices[start_index:end_index]
        X_i = X[indices_i]
        Y_i = Y[indices_i]
        yield X_i, Y_i


def get_diabetes_dataset(train_ratio=0.8):
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

    return X_train, Y_train, X_test, Y_test


class MyLinear(nn.Module):
    #x.shape = (batch_size, in_features)
    #y.shape = (batch_size, out_features)
    #y = x@w + b
    #w.shape = (in_features, out_features)
    #b.shape = (out_features)
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.parameter.Parameter(
            torch.empty((in_features, out_features)),
            requires_grad=True
        )
        self.b = nn.parameter.Parameter(
            torch.empty((out_features)),
            requires_grad=True
        )
        self.reset_parameters() 
    def reset_parameters(self):
        k = math.sqrt(1.0/self.in_features)
        nn.init.uniform_(self.w, a=-k, b=k)
        nn.init.constant_(self.b, 0.0)
    def forward(self, x):
        #x.shape = (batch_size, in_features)

        #xw.shape = (batch_size, out_features)
        xw = torch.matmul(x, self.w)

        xwb = xw + self.b
        return xwb
    
    # def parameters(self):
    #     result = []
    #     for key, value in self.__dict__.items():
    #         if isinstance(value, nn.parameter.Parameter):
    #             result.append(value)
    #     return result

