import torch
import torch.nn as nn

#
kernel_size = (4, 5)#(kh, kw)#(kernel_height, kernel_width)
stride = (2, 3)#(sh, sw)#(stride_height, stride_width)
max_pool2d = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

#x.shape = (batch_size, in_channels, height, width)
#x.shape = (10, 3, 800, 1000)
batch_size = 10
in_channels = 3
height = 800
width = 1000
x = torch.randn((batch_size, in_channels, height, width))


#max_pool2d_x.shape = (batch_size, in_channels, (height - kh)//sh + 1, (width - kw)//sw + 1)
#max_pool2d_x.shape = (10, 3, (800 - 4)//2 + 1, (1000 - 5)//3 + 1) = (10, 3, 399, 331.66666 + 1) = (10, 3, 399, 332)
max_pool2d_x = max_pool2d(x)
print('max_pool2d_x.shape = ', max_pool2d_x.shape)
