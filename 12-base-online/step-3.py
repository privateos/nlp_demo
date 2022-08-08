#__call__
import math
from unittest import result
class A:
    def __init__(self, x, y,):
        print('A.__init__')
        self.x_A = x
        self.y_A = y
        self.xxxxx = '我是大肥猪'
    def do_something(self, x):
        print('A.do_something')
        return math.exp(x)
    def __call__(self, x):
        print('A.__call__ ', x)
        return self.do_something(x)
    
    def get_str_value(self):
        result = []
        if isinstance(self.x_A, str):
            result.append(self.x_A)
        if isinstance(self.y_A, str):
            result.append(self.y_A)
        
        return result
    
    def new_get_str_value(self):
        result = []
        for key, value in self.__dict__.items():
            if isinstance(value, str):
            # if isinstance(value, nn.paramter.Parameter):
                result.append(value)
        return result

    

# a = A(1.23, 'abc')
# print('a.__dict__', a.__dict__)
# print(a.new_get_str_value())
# a.zzzzz = '珊珊爱源源'
# print('a.__dict__:', a.__dict__)
# print(a.new_get_str_value())
# # a.__call__(x=0.123)
# y = a(x=0.123)
# print(y)

class B(A):
    def __init__(self, x, y, z):
        super(B, self).__init__(x, y)
        # super().__init__(x, y)
        print('B.__init__')
        self.z_B = z

    def do_something(self, x):
        print('B.do_something')
        return math.sin(x)

# b = B(x=1.23, y='abc', z='#$')
# print(b.__dict__)
# y = b(x=2.34)
# print(y)
# y = b.do_something(x=2.34)
# print(y)
# b(x=1.23)
# class C(B):
#     def __init__(self, x, y, z, xxx):
#         super(C, self).__init__(x, y, z)
#         print('C.__init__')
#         self.xxx_C = xxx
    
#     def do_something(self, x):
#         print('C.do_something')
#         return math.sin(x) + math.cos(x)

class Model(nn.Module):
    # def parameters(self):
    #     result = []
    #     for key, value in self.__dict__.items():
    #         if isinstance(value, nn.parameter.Paramter):
    #             result.append(value)
    #     return result

model = Model(xxds)
model.parameters()