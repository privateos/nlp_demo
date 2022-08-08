import torch
import torch.nn as nn
import numpy as np

#(3, 4, 5, 6)
#(batch_size, in_channels, H, W)
shape = (3, 4, 5, 6)
L = 3*4*5*6
x = np.arange(0, L)
x = np.reshape(x, shape)

def example(x):
    shape = x.shape
    in_channels = shape[1]
    
    x_torch = torch.from_numpy(x).float()
    bn = nn.BatchNorm2d(in_channels, affine=False, track_running_stats=False)
    bn.eval()
    # print(bn.weight.shape, bn.bias.shape);exit()
    
    y_torch = bn(x_torch)
    y_numpy = y_torch.detach().cpu().numpy()
    
    
    eps = 1e-5
    
    #x_trans.shape= (3, 5, 6, 4)
    x_trans = np.transpose(x, (0, 2, 3, 1))
    
    #x_reshape.shape = (3*5*6, 4)
    x_reshape = np.reshape(x_trans, (3*5*6, 4))
    
    #x_mean.shape = (4,)
    #x_var.shape = (4, )
    x_mean = np.sum(x_reshape, axis=0)/(3*5*6)
    x_var = np.sum((x_reshape - x_mean)**2, axis=0)/(3*5*6)
    
    #x_norm.shape = (3*5*6, 4)
    x_norm = (x_reshape - x_mean)/np.sqrt(x_var + eps)
    
    #x_norm.shape = (3, 5, 6, 4)
    x_norm = np.reshape(x_norm, (3, 5, 6, 4))
    
    #x_norm.shape = (3, 4, 5, 6)
    x_norm = np.transpose(x_norm, (0, 3, 1, 2))
    
    
    dis = x_norm -  y_numpy
    print(np.max(np.abs(dis)))
    
    
    
example(x)


#(Batch_size, channels1, H1, W1)
#Conv()
#(batch_size, channels2, H2, W2)
#BatchNorm2d(channels2)
#RELU
#Conv
#(batch_size, channels3, H3, W3)
#BatchNorm2d(channels3)


class CNN(nn.Module):
    def __init__(self, channels1, channels2, channels3) -> None:
        super(CNN, self).__init__()
        self.seq = nn.Sequential(
            #(batch_size, channels1, H1, W1)
            nn.Conv2d(channels1, channels2),
            #(btch_size, channels2, H2, W2)
            nn.BatchNorm2d(channels2),
            nn.ReLU(),
            #(batch_size, channels2, H2, W2)
            
            nn.Conv2d(channels2, channels3),
            #(batch_size, channels3, H3, W3)
            nn.BatchNorm2d(channels3),
            nn.ReLU(),
            
            
            
            
            
        )
    def forward(self, x):
        return self.seq(x)
    
    

prev_x_mean.shape = (1, channels, 1)
current_x_mean.shape = (1, channels, 1)
running_mean
prev_x_mean = (1 - moment)*prev_x_mean + current_x_mean *moment

(x - prev_x_mean)/sqrt(prev_x_var)
#(batch_size, channels, H)
nn.BatchNorm1d(channels)

#(batch_size, input_size)
nn.BatchNorm1d(input_size)