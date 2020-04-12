#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   matrix.py
@Contact :   xuerzh18@lzu.edu.cn
@License :

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020-3-19         xrz        1.0         None
"""

import math
import copy

e = math.e
pi = math.pi


class Matrix(object):
    """realize a matrix class"""

    def __init__(self, m, n, value=0.0):
        """Initialize a matrix of M rows and N columns
        with an initial value of 0.0

        >>> matrix_test = Matrix(3, 3)
        >>> matrix_test
        [[0.0, 0.0, 0.0]
         [0.0, 0.0, 0.0]
         [0.0, 0.0, 0.0]]
        (3, 3)
        >>> matrix_test.shape
        (3, 3)
        """

        self._m = m
        self._n = n

        self.shape = (m, n)
        self._matrix = [[value for _ in range(n)] for _ in range(m)]

    def __eq__(self, other):
        """Judging whether two matrices are equal

        >>> m3 = Matrix(3, 3)
        >>> m4 = Matrix(3, 3, 1)
        >>> m3 == m3
        True
        >>> m3 == m4
        False"""
        assert isinstance(other, Matrix), str(other) + "is not the same type as" + str(self)
        flag = True
        if self.shape == other.shape:
            for i in range(self._m):
                for j in range(self._n):
                    if self[i, j] == other[i, j]:
                        continue
                    else:
                        flag = False
        return flag

    def __getitem__(self, item):
        """Enter a number to get the row of the matrix;
         enter a tuple to get the matrix's value of the position.
        >>> matrix = Matrix(2,2)
        >>> matrix.__getitem__(1)
        [0.0, 0.0]
        >>> matrix.__getitem__((1,1))
        0.0
        """
        if isinstance(item, int):
            return self._matrix[item]
        elif isinstance(item, tuple):
            return self._matrix[item[0]][item[1]]

    def __setitem__(self, key, value):
        """Set values in matrix
        >>> a = Matrix(2,2)
        >>> a.__setitem__(1,[2, 1, 1])
        >>> a
        [[0.0, 0.0]
         [2, 1, 1]]
        (2, 2)
        >>> a.__setitem__((1,2),2)
        >>> a
        [[0.0, 0.0]
         [2, 1, 2]]
        (2, 2)
        """
        if isinstance(key, int):
            self._matrix[key] = copy.deepcopy(value)
        elif isinstance(key, tuple):
            self._matrix[key[0]][key[1]] = value

    def __add__(self, other):
        """Add two matrices with the same dimension
        >>> m1 = Matrix(1,2, 1)
        >>> m2 = Matrix(1,2, 3)
        >>> m3 = m1 + m2
        >>> m3
        [[4, 4]]
        (1, 2)"""
        assert other.shape == self.shape, "Two matrices cannot be added because of different dimensions"
        M = Matrix(self._m, self._n)
        for r in range(self._m):
            for c in range(self._n):
                M[r, c] = self[r, c] + other[r, c]
        return M

    def __sub__(self, other):
        """Subtract two matrices with the same dimension
        >>> m1 = Matrix(1,2, 1)
        >>> m2 = Matrix(1,2, 3)
        >>> m3 = m1 - m2
        >>> m3
        [[-2, -2]]
        (1, 2)"""
        assert other.shape == self.shape, "Two matrices cannot be subtracted because of different dimensions"
        M = Matrix(self._m, self._n)
        for r in range(self._m):
            for c in range(self._n):
                M[r, c] = self[r, c] - other[r, c]
        return M

    def __mul__(self, other):
        """Find the number multiplication of a matrix
        or the product of two matrices

        Multiplying a constant
        >>> b = Matrix(2, 2, 1)
        >>> b * 3
        [[3, 3]
         [3, 3]]
        (2, 2)

        Multiplying two matrices
        >>> b * b
        [[2, 2]
         [2, 2]]
        (2, 2)
        """
        if isinstance(other, int) or isinstance(other, float):
            m = Matrix(self._m, self._n)
            for r in range(self._m):
                for c in range(self._n):
                    m[r, c] = self[r, c] * other
        elif isinstance(other, Matrix):
            assert other._m == self._n, "Can't multiply, another matrix's columns are not equal to the matrix's rows"
            m = Matrix(self._m, self._n)
            for r in range(self._m):
                for c in range(self._n):
                    _sum = 0
                    for k in range(self._n):
                        _sum += self[r, k] * other[k, r]
                    m[r, c] = _sum
        return m

    def transpose(self):
        """Transpose the matrix
        >>> d = Matrix(3, 3, 1)
        >>> d.__setitem__(1,[2, 2, 2])
        >>> d
        [[1, 1, 1]
         [2, 2, 2]
         [1, 1, 1]]
        (3, 3)
        >>> d.transpose()
        [[1, 2, 1]
         [1, 2, 1]
         [1, 2, 1]]
        (3, 3)"""
        M = Matrix(self._n, self._m)
        for r in range(self._n):
            for c in range(self._m):
                M[r, c] = self[c, r]
        return M

    def cofactor(self, i, j):
        """Finding the algebraic cofactor
        >>> e = Matrix(4, 4, 1)
        >>> e[1, 1] = 2
        >>> e.cofactor(3, 3)
        [[1, 1, 1]
         [1, 2, 1]
         [1, 1, 1]]
        (3, 3)
        """
        assert self._m == self._n, "can't be calculated because it's not a square matrix"
        assert self._m >= 3, "At least 3×3"
        assert i <= self._m and j <= self._n, "Out of range"
        m = Matrix(self._n - 1, self._m - 1)
        for r in range(self._m):
            if r == i:
                continue
            for c in range(self._n):
                if c == j:
                    continue
                rr = (r - 1) if r > i else r
                cc = c - 1 if c > j else c
                m[rr, cc] = self[r, c]
        return m

    def determinant(self):
        """Finding the determinant value of matrix
        >>> f = Matrix(1, 1, 1)
        >>> g = Matrix(4, 4)
        >>> f.determinant()
        1
        >>> g.__setitem__(0,[1, 1, 1, 1])
        >>> g.__setitem__(1,[1, 2, 3, 4])
        >>> g.__setitem__(2,[1, 4, 9, 16])
        >>> g.__setitem__(3,[1, 8, 27, 64])
        >>> g.determinant()
        12.0"""
        assert self._m == self._n, "Can't find a determinant because it's not a square matrix"

        sum_det = 0.0
        if self._m == 1:
            sum_det = self[0, 0]
        elif self._m == 2:
            sum_det = self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
        else:
            for c in range(self._n):
                sum_det += ((-1) ** (c + 1)) * self[1, c] * self.cofactor(1, c).determinant()

        return sum_det

    def __repr__(self):
        """Print out the matrix with its row and column number"""
        s = '['
        for r in range(self._m):
            s += str(self._matrix[r]) + ('\n ' if r != self.shape[0] - 1 else ']')
        return s + '\n' + str(self.shape)

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)
