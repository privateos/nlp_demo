class A:
    h = -1
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
class B(A):
    def __init__(self, x, y) -> None:
        self.x = x
        super().__init__(y, 10)
    
    def print(self):
        print(self.y, self.x, self.h)

b = B(2, 4)
b.print()