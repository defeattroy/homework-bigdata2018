'''
this homework include three parts: Matrix/Vector, their test code, and a _globals.py to make sure data accuracy
Matrix is run on top of Vector
The references are listed in(''' ''')
All the notes are translated from Chinese into English, so there might exists some little problems
'''
from homework7.Vector import Vector


class Matrix:

    def __init__(self, list2d):
        self._values = [row[:] for row in list2d]

    @classmethod
    #just like Vector, with the help of decorator, it's easy to realize null matrix
    def null(cls, r, c):
        #return a zero matrix of row r and column c
        return cls([[0] * c for _ in range(r)])

    def __add__(self, another):
        #return the addition result of two matrices
        '''
        >>> a=Matrix([[2,3],[4,6]])
        >>> b=Matrix([[4,7],[4,6]])
        >>> a+b
        Matrix([[6, 10], [8, 12]])
        '''
        assert self.shape() == another.shape(), \
            "Error in adding. Shape of matrix must be same."
        return Matrix([[a + b for a, b in zip(self.row_vector(i), another.row_vector(i))]
                       for i in range(self.row_num())])

    def __sub__(self, another):
        '''
        >>> a=Matrix([[2,3],[4,6]])
        >>> b=Matrix([[4,7],[4,6]])
        >>> a-b
        Matrix([[-2, -4], [0, 0]])
        '''
        #return the subtraction of two matrices
        assert self.shape() == another.shape(), \
            "Error in subtracting. Shape of matrix must be same."
        return Matrix([[a - b for a, b in zip(self.row_vector(i), another.row_vector(i))]
                       for i in range(self.row_num())])

    def dot(self, another):
        #return the result of matrix multiplication
        if isinstance(another, Vector):
            # Matrix vector multiplication
            assert self.col_num() == len(another), \
                "Error in Matrix-Vector Multiplication."
            return Vector([self.row_vector(i).dot(another) for i in range(self.row_num())])

        if isinstance(another, Matrix):
            # Multiplication of matrices and matrices
            assert self.col_num() == another.row_num(), \
                "Error in Matrix-Matrix Multiplication."
            return Matrix([[self.row_vector(i).dot(another.col_vector(j)) for j in range(another.col_num())]
                           for i in range(self.row_num())])

    def __mul__(self, k):
        #return the number of matrices multiplied by the result: self * k
        return Matrix([[e * k for e in self.row_vector(i)]
                       for i in range(self.row_num())])

    def __rmul__(self, k):
        #return the number of matrices multiplied by the result: k * self
        return self * k

    def __truediv__(self, k):
        #return the result matrix of quantity division:self / k
        return (1 / k) * self

    def __pos__(self):
        #return the result of taking a positive matrix.
        return 1 * self

    def __neg__(self):
        #return the negative result of the matrix
        return -1 * self

    def row_vector(self, index):
        #return the index th row vector of the matrix
        return Vector(self._values[index])

    def col_vector(self, index):
        #return the index th column vector of the matrix
        return Vector([row[index] for row in self._values])

    def __getitem__(self, pos):
        #return the element of the pos position of the matrix
        r, c = pos
        return self._values[r][c]

    def size(self):
        #return the number of elements in the matrix
        r, c = self.shape()
        return r * c

    def row_num(self):
        #return the number of rows in the matrix
        return self.shape()[0]

    __len__ = row_num

    def col_num(self):
        #return the number of columns in the matrix
        return self.shape()[1]

    def shape(self):
        #Return the shape of the matrix: (number of rows, number of columns)
        return len(self._values), len(self._values[0])

    def __repr__(self):
        return "Matrix({})".format(self._values)

    __str__ = __repr__
    if __name__ == "__main__":
        import doctest
        doctest.testmod()

    '''
    utilized from this
    https://www.cnblogs.com/superxuezhazha/p/5746922.html
    https://blog.csdn.net/zgcr654321/article/details/96883717 
    '''