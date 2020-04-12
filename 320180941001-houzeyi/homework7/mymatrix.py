class Matrix:
    __matrix = []

    def __init__(self, values):
        assert isinstance(values, list), 'matrix is a list'
        # This assertion is made to ensure that the matrix is a list
        assert len(values) > 0, 'matrix has to have one term'
        # This assertion is made to ensure that the matrix has at least one term
        self.__matrix = values
        self.shape = (len(self.matrix), len(self.matrix[0])) # The rows and columns of the matrix

    @property
    def matrix(self):
        return self.__matrix

    @matrix.setter
    def matrix(self, values):
        assert len(values) > 0, 'matrix has to have one term'
        assert isinstance(values, list), 'matrix is a list'
        self.__matrix = values
        self.shape = (len(self.matrix), len(self.matrix[0])) # The rows and columns of the matrix

    # Create an addition operation so that every corresponding numbers can add
    def __add__(self, other):
        assert self.shape == other.shape, 'Two matrices must have same shape'
        # This assertion is made to ensure that the rows and columns of the two matrices are the same
        result = []
        for i in range(self.shape[0]):
            a = self.matrix[i]
            b = other.matrix[i]
            result.append([x+y for x, y in zip(a, b)])
        return result

    # Create an subtraction operation so that every corresponding numbers can subtract
    def __sub__(self, other):
        assert self.shape == other.shape, 'Two matrices must have same shape'
        # This assertion is made to ensure that the rows and columns of the two matrices are the same
        result = []
        for i in range(self.shape[0]):
            a = self.matrix[i]
            b = other.matrix[i]
            result.append([x - y for x, y in zip(a, b)])
        return result

    # Create an multiplication operation so that every corresponding numbers can multiply
    def __mul__(self, other):
        assert self.shape == other.shape, 'Two matrices must have same shape'
        # This assertion is made to ensure that the rows and columns of the two matrices are the same
        result = []
        for i in range(self.shape[0]):
            a = self.matrix[i]
            b = other.matrix[i]
            result.append([x * y for x, y in zip(a, b)])
        return result

    # Create an division operation so that every corresponding numbers can divide
    def __truediv__(self, other):
        assert self.shape == other.shape, 'Two matrices must have same shape'
        # This assertion is made to ensure that the rows and columns of the two matrices are the same
        result = []
        for i in range(self.shape[0]):
            a = self.matrix[i]
            b = other.matrix[i]
            result.append([x / y for x, y in zip(a, b)])
        return result

    # Create an scalar multiplication operation.Use a number multiply every number in matrix
    def scalar(self, number):
        new_matrix = []
        for i in range(self.shape[0]):
            matrix1 = []
            for j in range(self.shape[1]):
                matrix1.append(number*self.matrix[i][j])
            new_matrix.append(matrix1)
        return new_matrix

    # Create an matrix transpose operation
    def transpose(self):
        new_matrix = []
        for i in range(self.shape[1]):
            matrix1 = []
            for j in range(self.shape[0]):
                matrix1.append(self.matrix[j][i])
            new_matrix.append(matrix1)
        return new_matrix

    def __str__(self):
        return str(self.matrix)

if __name__=='__main__':
    import doctest
    doctest.testmod()

'''
>>> a=Matrix([[1,2,3],[4,5,6]])
>>> b=Matrix([[4,5,6],[7,8,9]])
>>> a*b
[[4, 10, 18], [28, 40, 54]]
>>> a+b
[[5, 7, 9], [11, 13, 15]]
>>> a-b
[[-3, -3, -3], [-3, -3, -3]]
>>> a/b
[[0.25, 0.4, 0.5], [0.5714285714285714, 0.625, 0.6666666666666666]]
>>> a.transpose()
[[1, 4], [2, 5], [3, 6]]
>>> b.transpose()
[[4, 7], [5, 8], [6, 9]]
>>> a.scalar(3)
[[3, 6, 9], [12, 15, 18]]
>>> b.scalar(5)
[[20, 25, 30], [35, 40, 45]]
'''