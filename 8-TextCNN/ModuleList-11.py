import torch
import torch.nn as nn

#(batch_size, in_channels, height, width)
batch_size = 2
in_channels = 20
height = 800
width = 1000
x = torch.randn((batch_size, in_channels, height, width))


out_channels_list = [2, 3, 4]
layers = []
for out_channels in out_channels_list:
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)))

module_list = nn.ModuleList(layers)

outs = []
for module in module_list:
    out = module(x)
    outs.append(out)
    print(out.shape)

