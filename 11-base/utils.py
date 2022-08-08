import numpy as np
from sklearn.datasets import load_diabetes
import torch
import torch.nn as nn
import math


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
        index_i = indices[start_index:end_index]

        x_i = X[index_i]
        y_i = Y[index_i]
        yield x_i, y_i


def get_diabetes_dataset(train_ratio=0.8):
    diabetes = load_diabetes()
    X = diabetes.data#(442, 10)
    Y = diabetes.target#(442, )

    # print(X.shape, Y.shape)
    # print('X = ', X[0:3])
    # print('Y = ', Y[0:3])
    # print(diabetes.DESCR)
    Y = np.reshape(Y, (Y.shape[0], 1))#(442, 1)


    #数据划分
    N = X.shape[0]
    train_num = int(N*train_ratio)
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

    return X_train, Y_train, X_test, Y_test


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.parameter.Parameter(
            torch.empty(   (in_features, out_features)    ),
            requires_grad=True
            )
        self.b = nn.parameter.Parameter(
            torch.empty(   (out_features, )    ),
            requires_grad=True
            )
        self.reset_parameters()


    def reset_parameters(self):
        print('before w = ', self.w)
        print('before b = ', self.b)
        k = math.sqrt(1.0/self.in_features)
        nn.init.uniform_(self.w, a=-k, b=k)
        nn.init.constant_(self.b, 0.0)
        print('after w = ', self.w)
        print('after b = ', self.b)


    def forward(self, x):
        #x.shape = (batch_size, in_features)
        #y = x@w + b
        #y.shape = (batch_size, out_features)
        #w.shape = (in_features, out_features)
        #b.shape = (out_features, )

        xw = torch.matmul(x, self.w)
        xwb = xw + self.b
        return xwb

class MyMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyMSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, predict, target):
        if self.reduction == 'mean':
            return torch.mean(torch.square(predict - target))
        elif self.reduction == 'sum':
            return torch.sum(torch.square(predict - target))

# batch_size = 1
# in_features = 3
# out_features = 4
# x = torch.randn((batch_size, in_features))

# my_linear = MyLinear(in_features, out_features)
# y = my_linear(x)
# print(y)