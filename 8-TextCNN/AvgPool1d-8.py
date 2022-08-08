import torch
import torch.nn as nn

#
kernel_size = 4
stride = 5
avg_pool1d = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

#x.shape = (batch_size, in_channels, height)
#x.shape = (10, 3, 800)
batch_size = 10
in_channels = 3
height = 800
x = torch.randn((batch_size, in_channels, height))


#avg_pool2d_x.shape = (batch_size, in_channels, (height - kernel_size)//stride + 1)
#avg_pool2d_x.shape = (10, 3, (800 - 4)//5 + 1) = (10, 3, 159.2 + 1) = (10, 3, 160)
avg_pool1d_x = avg_pool1d(x)
print('max_pool1d_x.shape = ', avg_pool1d_x.shape)
