#Conv
import torch
import torch.nn as nn

#Conv2d
in_channels = 3
out_channels = 4
kernel_size = (5, 6)#(kh, kw) = (5, 6)
stride = (1, 2)#(sh, sw) = (1, 2)
conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

#x.shape = (batch_size, in_channels, height, width)
#x.shape = (10, 3, 800, 1000)
batch_size = 10
height = 800
width = 1000
x = torch.randn((batch_size, in_channels, height, width))
#conv2d_x.shape = (batch_size, out_channels, (height - kh)//sh + 1, (width - kw)//sw + 1)
#conv2d_x.shape = (10, 4, (800 - 5)//1 + 1, (1000 - 6)//2 + 1) = (10, 4, 796, 498)
conv2d_x = conv2d(x)
print('conv2d_x.shape = ', conv2d_x.shape)

