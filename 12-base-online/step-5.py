import torch
import torch.nn as nn
import math

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


