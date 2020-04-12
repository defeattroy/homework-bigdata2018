#!/usr/bin/env python3
# Copyright (c) 2020-3 ZhuTingYuan(Refection). All rights reserved


"""
Here is what you can do with Matrix and wrong usage example:
First: create your matrix
>>> a = Matrix(([1,2,3],[4,5,6]))
Traceback (most recent call last):
...
AssertionError: You should use list to create your matrix
>>> a = Matrix([[1,2,3],[4,5,6]])
>>> print(a)
Matrix:[[1, 2, 3], [4, 5, 6]]
>>> print(a.shape)
[2, 3]
>>> b = Matrix([0])
>>> print(b)
Matrix:[0]
>>> c = Matrix([[1,2,3]])
Traceback (most recent call last):
...
AssertionError: If you want to create an 1xn matrix,please use Matrix([x,x,x,...]) to instead
>>> c = Matrix([1,2,3])
>>> print(c)
Matrix:[1, 2, 3]
>>> print(c.shape)
[1, 3]

second: add(matrix add matrix, matrix add number)
>>> a = Matrix([1, 2, 3])
>>> b = Matrix([4, 5, 6])
>>> print("Matrix([1, 2, 3]) + Matrix([4, 5, 6]) = " + str(a + b))
Matrix([1, 2, 3]) + Matrix([4, 5, 6]) = Matrix:[5, 7, 9]
>>> c = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> d = Matrix([[7, 8, 9], [10, 11, 12], [1, 1, 1]])
>>> print(c + d)
Matrix:[[8, 10, 12], [14, 16, 18], [8, 9, 10]]
>>> e = Matrix([1,2,3])
>>> print(e+1)
Matrix:[2, 3, 4]
>>> print(e+1.1)
Matrix:[2.1, 3.1, 4.1]
>>> f = Matrix([[1,2,3],[4,5,6]])
>>> print(f+1)
Matrix:[[2, 3, 4], [5, 6, 7]]
>>> print(e+f)
Traceback (most recent call last):
...
AssertionError: Only two matrices have the same shape or one matrix and one number can be added

third: subtract(matrix subtract matrix, matrix subtract number)
>>> a = Matrix([1,2,3])
>>> b = Matrix([1,1,1])
>>> print(a-b)
Matrix:[0, 1, 2]
>>> c = Matrix([[1,2,3],[4,5,6]])
>>> print(c-a)
Traceback (most recent call last):
...
AssertionError: Only two matrices have the same shape or one matrix and one number can be subtracted
>>> d = Matrix([[1,2,3],[1,1,1]])
>>> print(c-d)
Matrix:[[0, 0, 0], [3, 4, 5]]
>>> e = Matrix([1,2,3])
>>> print(e-1)
Matrix:[0, 1, 2]
>>> f = Matrix([[1,2,3],[4,5,6]])
>>> print(f-1)
Matrix:[[0, 1, 2], [3, 4, 5]]

fourth: multiply(matrix with matrix, matrix with number)
>>> a = Matrix([[1, 2], [3, 4]])
>>> b = Matrix([[2, 2], [1, 1]])
>>> print(a * b)
Matrix:[[4, 4], [10, 10]]
>>> c = Matrix([1,2,3])
>>> d = Matrix([[1],[1],[1]])
>>> print(c*d)
Matrix:[6]
>>> print(a*d)
Traceback (most recent call last):
...
AssertionError: Only two matrices have the accepted shape or one matrix and one number can be multiplied
>>> print(a*2)
Matrix:[[2, 4], [6, 8]]
>>> print(c*d*2)
Matrix:[12]

fifth: dot(Calculate and return the product matrix of all the corresponding position elements)
>>> a = Matrix([[1, 2], [3, 4]])
>>> b = Matrix([[2, 3],[4, 5]])
>>> print(Matrix.dot(a,b))
Matrix:[[2, 6], [12, 20]]
>>> c = Matrix([1,2,3])
>>> print(Matrix.dot(a,c))
Traceback (most recent call last):
...
AssertionError: Only two matrices have the same shape or one matrix can be dotted

sixth: T(return Matrix's transposed Matrix)
>>> a = Matrix([[1, 2], [3, 4]])
>>> b = Matrix([[2, 2], [1, 1]])
>>> print(a.T())
Matrix:[[1, 3], [2, 4]]
>>> print(b.T())
Matrix:[[2, 1], [2, 1]]
>>> d = Matrix([1,2,3])
>>> print(d.T())
Matrix:[[1], [2], [3]]
>>> c = Matrix([[1], [2], [3]])
>>> print(c.T())
Matrix:[1, 2, 3]

seventh: identity(Create a identity matrix(n x n))
>>> a = Matrix.identity(3)
>>> print(a)
Matrix:[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
>>> b = Matrix.identity(5)
>>> print(b)
Matrix:[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
>>> c = Matrix.identity(1)
>>> print(c)
Matrix:[1]

eighth: new_zeros(use only shape to create a zero Matrix)
>>> a = Matrix.new_zeros([2,3])
>>> print(a)
Matrix:[[0, 0, 0], [0, 0, 0]]
>>> a = Matrix.new_zeros([1,1])
>>> print(a)
Matrix:[0]

ninth: determinant(Return the determinant of a Matrix.thinking from:https://www.khanacademy.org/math/
                                                                    linear-algebra/matrix-transformations/
                                                                    inverse-of-matrices/v/linear-algebra-nxn-determinant)
>>> a = Matrix([[1, 2, -4], [-2, 2, 1], [-3, 4, -2]])
>>> print(a.determinant())
-14
>>> b = Matrix([1])
>>> print(b.determinant())
1
>>> c = Matrix([[1,2,3],[1,1,1],[2,2,2]])
>>> print(c.determinant())
0
"""


class Matrix:

    def __init__(self, value: list):
        """
        create your matrix
        In this, __determinant is used to calculate the determinant of a Matrix, you should try to avoid calling it,
        you can use the method: determinant() instead of it to get a square matrix's determinant
        >>> a = Matrix([[1,2,3],[4,5,6]])
        >>> print(a)
        Matrix:[[1, 2, 3], [4, 5, 6]]
        >>> print(a.shape)
        [2, 3]
        >>> b = Matrix([9])
        >>> print(b)
        Matrix:[9]
        """
        assert not (len(value) == 1 and type(value[0]) is list), 'If you want to create an 1xn matrix,' \
                                                                 'please use Matrix([x,x,x,...]) to instead'
        assert type(value) is list, 'You should use list to create your matrix'
        self.__value = value
        self.__determinant = 0
        for i in value:
            if type(i) is list:
                self.__shape = [len(value), len(value[0])]
            else:
                self.__shape = [1, len(value)]
            break

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, new_value: list):
        assert type(new_value) is list, 'You should use list to change your matrix\'s value'
        self.__value = new_value

    @property
    def shape(self):
        return self.__shape

    def __str__(self):
        return 'Matrix:' + str(self.value)

    def __repr__(self):
        return 'Matrix: {0} shape is {1} '.format(self.value, self.shape)

    def __add__(self, other):
        """
        return Matrix a + Matrix b or Matrix c + number d
        >>> a = Matrix([1, 2, 3])
        >>> b = Matrix([4, 5, 6])
        >>> print("Matrix([1, 2, 3]) + Matrix([4, 5, 6]) = " + str(a + b))
        Matrix([1, 2, 3]) + Matrix([4, 5, 6]) = Matrix:[5, 7, 9]
        >>> c = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> d = Matrix([[7, 8, 9], [10, 11, 12], [1, 1, 1]])
        >>> print((c + d))
        Matrix:[[8, 10, 12], [14, 16, 18], [8, 9, 10]]
        >>> e = Matrix([1,2,3])
        >>> print(e+1)
        Matrix:[2, 3, 4]
        >>> print(e+1.1)
        Matrix:[2.1, 3.1, 4.1]
        >>> f = Matrix([[1,2,3],[4,5,6]])
        >>> print(f+1)
        Matrix:[[2, 3, 4], [5, 6, 7]]
        """
        if type(other) in (int, float):
            if self.shape[0] == 1:
                return Matrix(list(i + other for i in self.value))
            else:
                new_matrix = []
                for i in self.value:
                    column = []
                    for j in i:
                        column.append(j + other)
                    new_matrix.append(column)
                return Matrix(new_matrix)
        else:
            assert self.shape == other.shape, 'Only two matrices have the same ' \
                                              'shape or one matrix and one number can be added'
            if self.shape[0] == 1:
                return Matrix(list(i + j
                                   for i, j in zip(self.value, other.value)))
            else:
                new_matrix = []
                for i in list(Matrix(i) + Matrix(j) for i, j in zip(self.value, other.value)):
                    new_matrix.append(i.value)
                return Matrix(new_matrix)

    def __sub__(self, other):
        """
        return Matrix a - Matrix b
        >>> a = Matrix([1,2,3])
        >>> b = Matrix([1,1,1])
        >>> print(a-b)
        Matrix:[0, 1, 2]
        >>> c = Matrix([[1,2,3],[4,5,6]])
        >>> d = Matrix([[1,2,3],[1,1,1]])
        >>> print(c-d)
        Matrix:[[0, 0, 0], [3, 4, 5]]
        >>> e = Matrix([1,2,3])
        >>> print(e-1)
        Matrix:[0, 1, 2]
        >>> f = Matrix([[1,2,3],[4,5,6]])
        >>> print(f-1)
        Matrix:[[0, 1, 2], [3, 4, 5]]
        """
        if type(other) in (int, float):
            if self.shape[0] == 1:
                return Matrix(list(i - other for i in self.value))
            else:
                new_matrix = []
                for i in self.value:
                    column = []
                    for j in i:
                        column.append(j - other)
                    new_matrix.append(column)
                return Matrix(new_matrix)
        else:
            assert self.shape == other.shape, 'Only two matrices have the same shape ' \
                                              'or one matrix and one number can be subtracted'
            if self.shape[0] == 1:
                return Matrix(list(i - j
                                   for i, j in zip(self.value, other.value)))
            else:
                new_matrix = []
                for i in list(Matrix(i) - Matrix(j) for i, j in zip(self.value, other.value)):
                    new_matrix.append(i.value)
                return Matrix(new_matrix)

    def __mul__(self, other):
        """
        return a * b, a and b should be Matrix , or one Matrix and one number
        >>> a = Matrix([[1, 2], [3, 4]])
        >>> b = Matrix([[2, 2], [1, 1]])
        >>> print(a * b)
        Matrix:[[4, 4], [10, 10]]
        >>> c = Matrix([1,2,3])
        >>> d = Matrix([[1],[1],[1]])
        >>> print(c*d)
        Matrix:[6]
        >>> print(a*2)
        Matrix:[[2, 4], [6, 8]]
        >>> print(c*d*2)
        Matrix:[12]
        """
        if type(other) in (int, float):
            if self.shape[0] == 1:
                return Matrix(list(i * other for i in self.value))
            else:
                new_matrix = []
                for i in self.value:
                    column = []
                    for j in i:
                        column.append(j * other)
                    new_matrix.append(column)
                return Matrix(new_matrix)
        else:
            assert self.shape[1] == other.shape[
                0], 'Only two matrices have the accepted shape or one matrix and one number can be multiplied'
            if self.shape[0] == 1:
                sums = []
                for i in range(self.shape[1]):
                    sums.append(self.value[i] * other.value[i][0])
                return Matrix([sum(sums)])
            else:
                if self.shape[1] == other.shape[0]:
                    new_matrix = [[0] * other.shape[1] for i in range(self.shape[0])]
                    for i in range(self.shape[0]):
                        for j in range(other.shape[1]):
                            for k in range(other.shape[0]):
                                new_matrix[i][j] += self.value[i][k] * other.value[k][j]
                    return Matrix(new_matrix)

    @classmethod
    def dot(cls, these, other):
        """
        Calculate and return the product matrix of all the corresponding position elements
        >>> a = Matrix([[1, 2], [3, 4]])
        >>> b = Matrix([[2, 3],[4, 5]])
        >>> print(Matrix.dot(a,b))
        Matrix:[[2, 6], [12, 20]]
        >>> c = Matrix([1,2,3])
        >>> print(Matrix.dot(a,c))
        Traceback (most recent call last):
        ...
        AssertionError: Only two matrices have the same shape or one matrix can be dotted
        """
        assert these.shape == other.shape, 'Only two matrices have the same ' \
                                           'shape or one matrix can be dotted'
        new_matrix = Matrix.new_zeros(these.shape)
        for i in range(these.shape[0]):
            for j in range(these.shape[1]):
                new_matrix.value[i][j] = these.value[i][j] * other.value[i][j]
        return new_matrix

    def T(self):
        """
        return Matrix's transposed Matrix
        >>> a = Matrix([[1, 2], [3, 4]])
        >>> b = Matrix([[2, 2], [1, 1]])
        >>> print(a.T())
        Matrix:[[1, 3], [2, 4]]
        >>> print(b.T())
        Matrix:[[2, 1], [2, 1]]
        >>> d = Matrix([1,2,3])
        >>> print(d.T())
        Matrix:[[1], [2], [3]]
        >>> c = Matrix([[1], [2], [3]])
        >>> print(c.T())
        Matrix:[1, 2, 3]
        """
        if self.shape[0] == 1:
            new_matrix = Matrix.new_zeros([self.shape[1], self.shape[0]])
            for i in range(self.shape[1]):
                new_matrix.value[i][0] = self.value[i]
            return new_matrix
        elif self.shape[1] == 1:
            new_matrix = Matrix.new_zeros([self.shape[1], self.shape[0]])
            for i in range(self.shape[0]):
                new_matrix.value[i] = self.value[i][0]
            return new_matrix
        else:
            self_t = Matrix.new_zeros(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self_t.value[j][i] = self.value[i][j]
            return self_t

    @classmethod
    def identity(cls, n: int):
        """
        Create a identity matrix(n x n)
        >>> a = Matrix.identity(3)
        >>> print(a)
        Matrix:[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        """
        assert type(n) is int, 'The second argument should be int'
        if n is 1:
            return Matrix([1])
        new_matrix = Matrix.new_zeros([n, n])
        for i in range(n):
            new_matrix.value[i][i] = 1
        return new_matrix

    @classmethod
    def new_zeros(cls, shape):
        """
        use only shape to create a zero Matrix
        >>> a = Matrix.new_zeros([2,3])
        >>> print(a)
        Matrix:[[0, 0, 0], [0, 0, 0]]
        """
        new_matrix = []
        if shape[0] == 1:
            for i in range(0, shape[1]):
                new_matrix.append(0)
            return Matrix(new_matrix)
        else:
            for i in range(0, shape[0]):
                column = []
                for j in range(0, shape[1]):
                    column.append(0)
                new_matrix.append(column)
            return Matrix(new_matrix)

    @staticmethod
    def __matrix_inversion(__column_list: list) -> int:
        """
        This method only open to method: determinant(), you should try to avoid calling it.
        Calculate the inverse number of a sequence.
        """
        count = 0
        for i in range(1, len(__column_list)):
            compare_number = __column_list[i]
            for j in range(i):
                if compare_number < __column_list[j]:
                    count += 1
        return count

    def __full_permutation(self, __matrix_in: list, __index_list: list, __column_list: list, __index: int):
        """
        This method only open to method: determinant(), you should try to avoid calling it.
        Calculate the sequence's full permutation
        """
        for i in range(__index, len(__column_list)):
            if __index == len(__column_list) - 1:
                item = 1
                for diagonal in range(len(__index_list)):
                    i = __index_list[diagonal]
                    j = __column_list[diagonal]
                    item *= __matrix_in[i][j]
                if Matrix.__matrix_inversion(__column_list) % 2 == 0:
                    self.__determinant += item
                else:
                    self.__determinant -= item
                return
            __column_list[__index], __column_list[i] = __column_list[i], __column_list[__index]
            Matrix.__full_permutation(self, __matrix_in, __index_list, __column_list, __index + 1)
            __column_list[__index], __column_list[i] = __column_list[i], __column_list[__index]

    def determinant(self):
        """
        Return the determinant of a Matrix.
        >>> a = Matrix([[1, 2, -4], [-2, 2, 1], [-3, 4, -2]])
        >>> print(a.determinant())
        -14
        >>> b = Matrix([1])
        >>> print(b.determinant())
        1
        """
        assert self.shape[0] == self.shape[1], 'Only a square matrix can compute determinants'
        if self.shape[0] == self.shape[1] == 1:
            self.__determinant = self.value[0]
            return self.__determinant
        column_list = []
        index_list = []
        for index in range(len(self.value)):
            index_list.append(index)
            column_list.append(index)
        Matrix.__full_permutation(self, self.value, index_list, column_list, 0)
        return self.__determinant


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
