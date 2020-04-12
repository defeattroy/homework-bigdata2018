import math
from homework7._globals import EPSILON
class Vector:

    def __init__(self, lst):
        self._values = list(lst)

    @classmethod
    #借用修饰符实现零向量，不知道有没有不借助但一样简洁的方法
    def zero(cls, dim):
        #Returns a dim dimension zero vector
        return cls([0] * dim)

    def __add__(self, another):
        #Vector addition returns the resulting vector
        assert len(self) == len(another), \
            "Error in adding. Length of vectors must be same."

        return Vector([a + b for a, b in zip(self, another)])

    def __sub__(self, another):
        #vector subtraction, return the result vector
        assert len(self) == len(another), \
            "Error in subtracting. Length of vectors must be same."

        return Vector([a - b for a, b in zip(self, another)])

    def norm(self):
        #returns the magnitude of the vector
        return math.sqrt(sum(e**2 for e in self))

    def normalize(self):
        #returns the unit vector of the vector
        if self.norm() < EPSILON:
            raise ZeroDivisionError("Normalize error! norm is zero.")
        return Vector(self._values) / self.norm()

    def dot(self, another):
        #dot product of a vector, returns the resulting scalar
        assert len(self) == len(another), \
            "Error in dot product. Length of vectors must be same."

        return sum(a * b for a, b in zip(self, another))

    def __mul__(self, k):
        #returns the result vector of scalar multiplication:self * k
        return Vector([k * e for e in self])

    def __rmul__(self, k):
        #returns the result vector of the multiplication of quantities: k * self
        return self * k

    def __truediv__(self, k):
        #returns the result vector of number division: self/k
        return (1 / k) * self

    def __pos__(self):
        #returns the result vector that takes a positive vecto
        return 1 * self

    def __neg__(self):
        #returns the result vector with a negative vector
        return -1 * self

    def __iter__(self):
        #the iterator that returns the vector
        return self._values.__iter__()

    def __getitem__(self, index):
        """取向量的第index个元素"""
        return self._values[index]

    def __len__(self):
        #the index element of the vector
        return len(self._values)

    def __repr__(self):
        #the magic methodfor the system to call
        '''utilized from this
        https://blog.csdn.net/qq_38721302/article/details/83859691'''
        return "Vector({})".format(self._values)

    def __str__(self):
        return "({})".format(", ".join(str(e) for e in self._values))
        return self._values[index]