class Matrix:
    """
    Matrix can only be constructed from a list.
    empty list ([]) and list having different lengths for each row are illegal.

    Matrix class supports operations bellow:
    1. a+b or a.add(b), 'b' can be a Matrix or a number (int, float)
    2. a-b or a.sub(b), 'b' can be a Matrix or a number (int, float)
    3. a*b or a.mul(b), 'b' can be a Matrix or a number (int, float)
    4. a/b or a.div(b), 'b' can be a Matrix or a number (int, float)
    5. a.dot(b) : dot product between Matrix a and b
    6. a.transpose() : the transpose of Matrix a
    """
    __values = []

    def __init__(self, val):
        """
        >>> matrix = Matrix([[1, 2, 3], [1, 2, 3]])
        >>> matrix.shape
        (2, 3)
        >>> matrix
        values:
        [[1, 2, 3]
         [1, 2, 3]]
        shape: (2, 3)

        >>> vector = Matrix([1, 2, 3])
        >>> vector.shape
        (3, None)
        >>> vector
        values:
        [1, 2, 3]
        shape: (3, None)

        >>> matrix_false1 = Matrix([])
        Traceback (most recent call last):
            ...
        AssertionError: the matrix must have at least one element

        >>> matrix_false2 = Matrix(1)
        Traceback (most recent call last):
            ...
        AssertionError: value of the matrix must be a list

        >>> matrix_false3 = Matrix([[1,2,3], [2]])
        Traceback (most recent call last):
            ...
        AssertionError: length of each row can't be different

        """

        assert isinstance(val, list), 'value of the matrix must be a list'
        assert len(val) > 0, 'the matrix must have at least one element'
        if isinstance(val[0], list):
            assert len(val[0]) > 0, 'the matrix must have at least one element'
            assert len(set([len(i) for i in val])) == 1, "length of each row can't be different"
        self.__values = val
        self.shape = (len(self.values),
                      len(self.values[0]) if isinstance(self.values[0], list) else None)

    @property
    def values(self):
        """
       >>> matrix = Matrix([[1, 2, 3], [1, 2, 3]])
       >>> matrix.values
       [[1, 2, 3], [1, 2, 3]]

       >>> matrix.values = []
       Traceback (most recent call last):
           ...
       AssertionError: the matrix must have at least one element
       >>> matrix.values = 1
       Traceback (most recent call last):
           ...
       AssertionError: value of the matrix must be a list
       >>> matrix.values = [[1], [2, 3]]
       Traceback (most recent call last):
           ...
       AssertionError: length of each row can't be different

       """
        return self.__values

    @values.setter
    def values(self, val):
        assert isinstance(val, list), 'value of the matrix must be a list'
        assert len(val) > 0, 'the matrix must have at least one element'
        if isinstance(val[0], list):
            assert len(val[0]) > 0, 'the matrix must have at least one element'
            assert len(set([len(i) for i in val])) == 1, "length of each row can't be different"
        self.__values = val
        self.shape = (len(self.values),
                      len(self.values[0]) if isinstance(self.values[0], list) else None)

    def __str__(self):
        """
        >>> a = Matrix([[1, 2, 3], [1, 2, 3]])
        >>> a
        values:
        [[1, 2, 3]
         [1, 2, 3]]
        shape: (2, 3)
        """
        if self.shape[1] is not None:
            s = '['
            for i in range(self.shape[0]):
                s += str(self.values[i]) + ('\n ' if i != self.shape[0] - 1 else ']\n')
            return 'values:\n' + s + 'shape: {}'.format(self.shape)
        else:
            return 'values:\n' + str(self.values) + '\nshape: {}'.format(self.shape)

    def __repr__(self):
        return self.__str__()

    def add(self, para):
        """
        >>> a = Matrix([1,2,3])
        >>> b = Matrix([2,3,4])
        >>> a+1
        values:
        [2, 3, 4]
        shape: (3, None)
        >>> a+b
        values:
        [3, 5, 7]
        shape: (3, None)
        >>> a.add(b)
        values:
        [3, 5, 7]
        shape: (3, None)
        >>> ma = Matrix([[1,2,3],[4,5,6]])
        >>> mb = Matrix([[1,2,3],[4,5,6]])
        >>> ma + 1
        values:
        [[2, 3, 4]
         [5, 6, 7]]
        shape: (2, 3)
        >>> ma + mb
        values:
        [[2, 4, 6]
         [8, 10, 12]]
        shape: (2, 3)
        >>> ma.add(mb)
        values:
        [[2, 4, 6]
         [8, 10, 12]]
        shape: (2, 3)

        >>> m_fail = Matrix([[1,2], [3,4]])
        >>> m_fail + ma
        Traceback (most recent call last):
           ...
        AssertionError: matrix can only operate with a number or matrix that have the same shape
        """
        return self.__add__(para)

    def __add__(self, para):
        if isinstance(para, Matrix):
            assert para.shape == self.shape, "matrix can only operate with a number or matrix that have the same shape"
            r = []
            if self.shape[1] is not None:
                for i in range(self.shape[0]):
                    a = self.values[i]
                    b = para.values[i]
                    r.append([x + y for x, y in zip(a, b)])
                return Matrix(r)
            elif isinstance(para.values, list):
                return Matrix([x + y for x, y in zip(self.values, para.values)])  # 列表每个元素的相加
        assert isinstance(para, int) or isinstance(para, float), "matrix can only operate with a number or matrix that have the same shape"
        return self.__operate('+', para)

    def sub(self, para):
        """
        >>> a = Matrix([1,2,3])
        >>> b = Matrix([2,3,4])
        >>> a-1
        values:
        [0, 1, 2]
        shape: (3, None)
        >>> a-b
        values:
        [-1, -1, -1]
        shape: (3, None)
        >>> a.sub(b)
        values:
        [-1, -1, -1]
        shape: (3, None)
        >>> ma = Matrix([[1,2,3],[4,5,6]])
        >>> mb = Matrix([[1,2,3],[4,5,6]])
        >>> ma - 1
        values:
        [[0, 1, 2]
         [3, 4, 5]]
        shape: (2, 3)
        >>> ma - mb
        values:
        [[0, 0, 0]
         [0, 0, 0]]
        shape: (2, 3)
        >>> ma.sub(mb)
        values:
        [[0, 0, 0]
         [0, 0, 0]]
        shape: (2, 3)

        >>> m_fail = Matrix([[1,2], [3,4]])
        >>> m_fail - ma
        Traceback (most recent call last):
           ...
        AssertionError: matrix can only operate with a number or matrix that have the same shape
        """
        return self.__sub__(para)

    def __sub__(self, para):
        if isinstance(para, Matrix):
            assert para.shape == self.shape, "matrix can only operate with a number or matrix that have the same shape"
            r = []
            if self.shape[1] is not None:
                for i in range(self.shape[0]):
                    a = self.values[i]
                    b = para.values[i]
                    r.append([x - y for x, y in zip(a, b)])
                return Matrix(r)
            elif isinstance(para.values, list):
                return Matrix([x - y for x, y in zip(self.values, para.values)])
        assert isinstance(para, int) or isinstance(para, float), "matrix can only operate with a number or matrix that have the same shape"
        return self.__operate('-', para)

    def mul(self, para):
        """
        >>> a = Matrix([1,2,3])
        >>> b = Matrix([2,3,4])
        >>> a*2
        values:
        [2, 4, 6]
        shape: (3, None)
        >>> a*b
        values:
        [2, 6, 12]
        shape: (3, None)
        >>> a.mul(b)
        values:
        [2, 6, 12]
        shape: (3, None)

        >>> ma = Matrix([[1,2,3],[4,5,6]])
        >>> mb = Matrix([[1,2,3],[4,5,6]])
        >>> ma * 2
        values:
        [[2, 4, 6]
         [8, 10, 12]]
        shape: (2, 3)
        >>> ma * mb
        values:
        [[1, 4, 9]
         [16, 25, 36]]
        shape: (2, 3)
        >>> ma.mul(mb)
        values:
        [[1, 4, 9]
         [16, 25, 36]]
        shape: (2, 3)

        >>> m_fail = Matrix([[1,2], [3,4]])
        >>> m_fail * ma
        Traceback (most recent call last):
           ...
        AssertionError: matrix can only operate with a number or matrix that have the same shape
        """
        return self.__mul__(para)

    def __mul__(self, para):
        if isinstance(para, Matrix):
            assert para.shape == self.shape, "matrix can only operate with a number or matrix that have the same shape"
            r = []
            if self.shape[1] is not None:
                for i in range(self.shape[0]):
                    a = self.values[i]
                    b = para.values[i]
                    r.append([x * y for x, y in zip(a, b)])
                return Matrix(r)
            else:
                return Matrix([x * y for x, y in zip(self.values, para.values)])
        assert isinstance(para, int) or isinstance(para,
                                                   float), "Matrix can only operate with a number or matrix that have the same shape"
        return self.__operate('*', para)

    def dot(self, other):
        """
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> n = Matrix([[1, 1], [2, 2], [3, 3]])
        >>> m.dot(n)
        values:
        [[14, 14]
         [32, 32]]
        shape: (2, 2)

        >>> wrong_shape = Matrix([[1, 2, 3]])
        >>> m.dot(wrong_shape)
        Traceback (most recent call last):
            ...
        AssertionError: You can't dot two matrices because of the wrong shape
        """
        assert isinstance(other, Matrix), 'the type of dot parameter is wrong, must be a matrix'
        assert self.shape[1] == other.shape[0], "You can't dot two matrices because of the wrong shape"

        def dot(m1: list, m2: list) -> list:
            # m1:mxn m2:nxm
            m3 = []
            m = len(m1)
            n = len(m1[0])
            p = len(m2[0])
            # result: m*p
            for i in range(m):
                n_r = []
                for j in range(p):
                    t = 0
                    for k in range(n):
                        t += m1[i][k] * m2[k][j]
                    n_r.append(t)
                m3.append(n_r)
            return m3

        # matrix
        if self.shape[1] is not None and other.shape[1] is not None:
            return Matrix(dot(self.values, other.values))
        else:  # vector
            return Matrix([i * j for i, j in zip(self.values, other.values)])

    def div(self, other):
        """
        >>> a = Matrix([4, 6, 8])
        >>> b = Matrix([2, 3, 4])
        >>> a/2
        values:
        [2.0, 3.0, 4.0]
        shape: (3, None)
        >>> a/b
        values:
        [2.0, 2.0, 2.0]
        shape: (3, None)
        >>> ma = Matrix([[2, 2, 2], [4, 6, 8]])
        >>> mb = Matrix([[1, 2, 1], [2, 3, 4]])
        >>> ma/2
        values:
        [[1.0, 1.0, 1.0]
         [2.0, 3.0, 4.0]]
        shape: (2, 3)
        >>> fail = Matrix([1, 2, 3])
        >>> ma/fail
        Traceback (most recent call last):
            ...
        AssertionError: matrix can only operate with a number or matrix that have the same shape
        """
        return self.__truediv__(other)

    def __truediv__(self, para):
        if isinstance(para, Matrix):
            assert para.shape == self.shape, "matrix can only operate with a number or matrix that have the same shape"
            r = []
            if self.shape[1] is not None:
                for i in range(self.shape[0]):
                    a = self.values[i]
                    b = para.values[i]
                    r.append([x / y for x, y in zip(a, b)])
                return Matrix(r)
            else:
                return Matrix([x / y for x, y in zip(self.values, para.values)])
        assert isinstance(para, int) or isinstance(para,
                                                   float), "The matrix can only be divided by a number"
        return self.__operate('/', para)

    def __operate(self, op, num):
        a = []
        for i in self.values:
            if isinstance(i, list):
                exec('a.append([j{}{} for j in i])'.format(op, num))
            else:
                exec('a.append(i{}{})'.format(op, num))
        return Matrix(a)

    def transpose(self):
        """
        >>> a = Matrix([1, 2, 3])
        >>> a
        values:
        [1, 2, 3]
        shape: (3, None)
        >>> a.transpose()
        values:
        [1, 2, 3]
        shape: (3, None)

        >>> b = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> b
        values:
        [[1, 2, 3]
         [4, 5, 6]]
        shape: (2, 3)
        >>> b.transpose()
        values:
        [[1, 4]
         [2, 5]
         [3, 6]]
        shape: (3, 2)
        """
        def transpose(a: list) -> list:
            m = len(a)
            n = len(a[0]) if isinstance(a[0], list) else 1
            r = []
            for i in range(n):
                b = []
                for k in range(m):
                    b.append(a[k][i])
                r.append(b)
            return Matrix(r)

        if self.shape[1] is not None:  # matrix
            return transpose(self.values)
        else:  # vector
            return Matrix(self.values)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
