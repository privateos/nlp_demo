import torch
import torch.nn as nn
import math

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

batch_size = 1
in_features = 3
out_features = 4
x = torch.randn((batch_size, in_features))

my_linear = MyLinear(in_features, out_features)
# y = my_linear(x)
# print(y)