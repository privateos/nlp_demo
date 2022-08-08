import torch

#x1.shape = (a, c)
a = 10
# b = 11
c = 12
x1 = torch.randn((a, c))

#x2.shape = (a, d)
d = 13
x2 = torch.randn((a, d))

#x3.shape = (a, e)
e = 14
x3 = torch.randn((a, e))

#x123.shape = (a, c + d + e) = (10, 12 + 13 + 14) = (10, 39)
x123 = torch.cat([x1, x2, x3], dim=1)
print('x123.shape = ', x123.shape)


###
#x1.shape = (a, b, c)
a = 10
b = 11
c = 12
x1 = torch.randn((a, b, c))

#x2.shape = (a, b, d)
d = 13
x2 = torch.randn((a, b, d))

#x3.shape = (a, b, e)
e = 14
x3 = torch.randn((a, b, e))

#x123.shape = (a, b, c + d + e) = (10,11, 12 + 13 + 14) = (10, 11, 39)
x123 = torch.cat([x1, x2, x3], dim=2)
print('x123.shape = ', x123.shape)