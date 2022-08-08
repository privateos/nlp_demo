#Unsqueeze/Squeeze
import torch
x = torch.randn((2,3,4))
print('x.shape=', x.shape)

#(2,3,4)--->(2, 1, 3,4)
x_reshape = torch.reshape(x, (2,1,3,4))
print('x_reshape.shape=' ,x_reshape.shape)

#(2,3,4)--->(2, 1, 3, 4)
x_unsqueeze = torch.unsqueeze(x, dim=1)
print('x_unsqueeze.shape=', x_unsqueeze.shape)



#(2,1,3,4)--->(2,3,4)
x = torch.randn((2,1,3,4))
x_squeeze = torch.squeeze(x, dim=1)
print('x_squeeze.shape = ', x_squeeze.shape)
