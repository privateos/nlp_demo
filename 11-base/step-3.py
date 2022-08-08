import math

from numpy import isin


#__call__
class A:
    def __init__(self):
        pass

    def do_something(self, x):
        print('A:do_something')
        return math.exp(x)
    
    def __call__(self, x):
        print('__call__', x)
        return self.do_something(x)


a = A()
c = a(0.123)
print('c = ', c)

#继承以及__call__
class B(A):
    def __init__(self):
        super(B, self).__init__()
    
    def do_something(self, x):
        print('B:do_something')
        return math.exp(x) + 1.0

b = B()
bc = b(0.123)
print('bc = ', bc)





#__dict__
print('_________________dict__________________')
class C:
    def __init__(self, name, sex, age):
        self.name_C = name
        self.sex_C = sex
        self.age_C = age
        self.abc = '我是大肥猪'
        self.bcd = '珊珊爱源源'
    
    def get_str_value(self):
        result = []
        if isinstance(self.name_C, str):
            result.append(self.name_C)
        if isinstance(self.sex_C, str):
            result.append(self.sex_C)
        if isinstance(self.age_C, str):
            result.append(self.age_C)
        if isinstance(self.abc, str):
            result.append(self.abc)
        return result
    
    def new_get_str_value(self):
        result = []
        for key, value in self.__dict__.items():
            if isinstance(value, str):
                result.append(value)
        return result

# c = C('珊珊', '男', 25)
# print(c.get_str_value())
# print(c.new_get_str_value())
# print(c.__dict__)
# print('two methods ', c.name_C, c.__dict__['name_C'])
class D(C):
    def __init__(self, name, sex, age, weight, color):
        super(D, self).__init__(name, sex, age)
        self.weight_d = weight
        self.color_d = color

d = D('珊珊', '男', 25, 70, '黄种人')
print(d.new_get_str_value())
