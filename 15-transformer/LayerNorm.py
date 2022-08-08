from math import gamma
from matplotlib.pyplot import axis
from sklearn.decomposition import KernelPCA
import torch
import torch.nn as nn
import numpy as np
shape = (3, 4, 5)#(batch_size, seq_len, hidden_size)
L = shape[0]*shape[1]*shape[2]
x = np.arange(0, L)
x = x.reshape(shape)
eps = 1e-5
# print(x)

def example1(x):
    #####torch########################
    shape = x.shape #(3, 4, 5)
    x_torch = torch.from_numpy(x).float()
    layer_norm = nn.LayerNorm(shape[-1], eps=eps, elementwise_affine=False)

    y_torch = layer_norm(x_torch)

    y_numpy = y_torch.detach().cpu().numpy()
    #####torch########################

    #####numpy######################
    #x.shape = (3, 4, 5)
    #x_mean = (3, 4, 1)
    x_mean = np.sum(x, axis=2, keepdims=True)/shape[2]
    x2 = (x - x_mean)**2
    x_var = np.sum(x2, axis=2, keepdims=True)/shape[2]
    x_layer_norm = (x - x_mean)/np.sqrt(x_var + eps)
    #####numpy######################

    print(y_numpy.shape, x_layer_norm.shape)

    d = y_numpy - x_layer_norm
    print(np.max(np.abs(d)))


def example2(x):
    #####torch########################
    shape = x.shape#(3, 4, 5)
    x_torch = torch.from_numpy(x).float()
                              #(4, 5)
    layer_norm = nn.LayerNorm((shape[1], shape[2]), eps=eps, elementwise_affine=False)

    y_torch = layer_norm(x_torch)

    y_numpy = y_torch.detach().cpu().numpy()
    #####torch########################

    #####numpy######################
    #x.shape = (3, 4, 5)
    #x_mean = (3, 4, 1)
    x_mean = np.sum(x, axis=2, keepdims=True)
    #x_mean.shape = (3, 1, 1)
    x_mean = np.sum(x_mean, axis=1, keepdims=True)

    x_mean = x_mean/(shape[1]*shape[2])


    x2 = (x - x_mean)**2
    x_var = np.sum(x2, axis=2, keepdims=True)
    x_var = np.sum(x_var, axis=1, keepdims=True)
    x_var = x_var/(shape[1]*shape[2])

    x_layer_norm = (x - x_mean)/np.sqrt(x_var + eps)
    #####numpy######################

    print(y_numpy.shape, x_layer_norm.shape)

    d = y_numpy - x_layer_norm
    print(np.max(np.abs(d)))

# example1(x)
example2(x)





class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            gamma = torch.ones(normalized_shape, dtype=torch.float32)
            beta = torch.zeros(normalized_shape, dtype=torch.float32)
            self.gamma = nn.parameter.Parameter(gamma, requires_grad=True)
            self.beta = nn.parameter.Parameter(beta, requires_grad=True)

        def forward(self, x):
            #x.shape = (3, 4, 5)
            #normalized_shape = (5, )
            #z = (x - E[x])/sqrt(Var(x) + eps)
            z = x - E[x]/sqrt(Var(x) + eps)
            if self.elementwise_affine:
                return z
            else:
                return z*self.gamma + self.beta
            
