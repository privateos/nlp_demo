import torch
import torch.nn as nn
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
    
    def forward(self, x):
        print('TestModel forward')
        return torch.sigmoid(x)

a = TestModel()
x = torch.tensor(1.0)
y = a(x)
print('y = ', y)