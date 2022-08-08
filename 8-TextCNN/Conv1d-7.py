#Conv
import torch
import torch.nn as nn

#Conv1d
in_channels = 3
out_channels = 4
kernel_size = 5
stride = 6
conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

#x.shape = (batch_size, in_channels, height)
#x.shape = (10, 3, 800)
batch_size = 10
height = 800
x = torch.randn((batch_size, in_channels, height))
#conv1d_x.shape = (batch_size, out_channels, (height - kernel_size)//stride + 1)
#conv1d_x.shape = (10, 4, (800 - 5)//6 + 1) = (10, 4, 132.5 + 1) = (10, 4, 133)
conv1d_x = conv1d(x)
print('conv1d_x.shape = ', conv1d_x.shape)
