import torch
import torch.nn as nn

#
kernel_size = 4
stride = 5
max_pool1d = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

#x.shape = (batch_size, in_channels, height)
#x.shape = (10, 3, 800, 1000)
batch_size = 10
in_channels = 3
height = 800
x = torch.randn((batch_size, in_channels, height))


#max_pool2d_x.shape = (batch_size, in_channels, (height - kernel_size)//stride + 1)
#max_pool2d_x.shape = (10, 3, (800 - 4)//5 + 1) = (10, 3, 159.2 + 1) = (10, 3, 160)
max_pool1d_x = max_pool1d(x)
print('max_pool1d_x.shape = ', max_pool1d_x.shape)
