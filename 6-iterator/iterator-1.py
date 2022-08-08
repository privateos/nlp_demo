a = ['x', 'y', 100, ]
# for v in a:
#     print(v)

class Iterator:
    def __init__(self, a):
        self.a = a
    
    def __iter__(self):
        self.n = len(self.a)
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.n:
            result = self.a[self.i]
            self.i += 1
            return result
        else:
            raise StopIteration

# it = Iterator(a)
# it_for = iter(it)#it.__iter__()

# v = next(it_for)#it.__next__()
# print(v)

# v = next(it_for)
# print(v)

# v = next(it_for)
# print(v)

# v = next(it_for)
# print(v)
# exit()

# it = Iterator(a)
# it_for = it.__iter__()

# v = it_for.__next__()
# print(v)

# v = it_for.__next__()
# print(v)

# v = it_for.__next__()
# print(v)

# v = it_for.__next__()
# print(v)
# exit()

it = Iterator(a)
for i in it:
    print(i)
for i in it:
    print(i)