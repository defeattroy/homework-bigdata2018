"""
This class provides creation and operation of matrix.

It implements some functions:
    a+b: a.__add__(b) Matrix addition.
    a-b: a.__sub__(b) Matrix subtraction.
    a*b(b:num): a.sca_mul(b) Matrix scalar multiplication.
    a/b(b:num): a.__truediv__(b) Matrix division.
    a*b(b:matrix): a.__mul__(b) Matrix multiplication.
    a·b: a.dot_mul(b) Matrix dot product.
And others:
    a.show(): Show it in matrix form.
    a.transpose(): Matrix transpose.
    a==b?: a.__eq__(b) Judge if a is equal to b.

ATTENTION: All the 'a' above are matrixes. It depends on the situation whether 'b' is a matrix or a number.
"""


class Matrix:
    def __init__(self, val):
        """
        Initialize the matrix
        >>> a = Matrix([[1,1,1],[2,2,2]])
        >>> b = Matrix(1)
        Traceback (most recent call last):
            ...
        AssertionError: You must input a list.
        >>> c = Matrix([[3,3,3],[4,4]])
        Traceback (most recent call last):
            ...
        AssertionError: Each row has as many elements as the other rows.
        >>> a.value
        [[1, 1, 1], [2, 2, 2]]
        >>> a.shape
        (2, 3)
        """
        assert isinstance(val, list), "You must input a list."
        if isinstance(val[0], list):
            le = []
            for i in range(len(val)):
                le.append(len(val[i]))
            assert len(set(le)) == 1, "Each row has as many elements as the other rows."

        self.value = val
        self.shape = (len(val), len(val[0]) if isinstance(val[0], list) else 0)

    def __str__(self):
        """Return value of the matrix
        >>> a = Matrix([[1,1,1],[2,2,2]])
        >>> a
        [[1, 1, 1], [2, 2, 2]]

        :return:str
        """
        return str(self.value)

    def __repr__(self):
        return self.__str__()

    def show(self):
        """
        Show it in matrix form.
        >>> a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        >>> a.show()
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
        """
        s = '['
        for i in range(self.shape[0] - 1):
            s += str(self.value[i]) + ',' + '\n' + ' '
        s += str(self.value[self.shape[0] - 1]) + ']'
        print(s)

    def __add__(self, another):
        """
        Matrix addition.
        >>> a = Matrix([[1,1,1],[2,2,2]])
        >>> b = Matrix([[3,3,3],[4,4,4]])
        >>> c = Matrix([[5,5],[6,6]])
        >>> a+b
        [[4, 4, 4], [6, 6, 6]]
        >>> a+c
        Traceback (most recent call last):
        AssertionError: The shapes of these two matrixes must be the same.

        :param another: Matrix
        :return: Matrix
        """
        assert self.shape == another.shape, "The shapes of these two matrixes must be the same."
        result = []
        if self.shape[1] > 0:
            for i in range(self.shape[0]):
                r = []
                for j in range(self.shape[1]):
                    r.append(self.value[i][j] + another.value[i][j])
                result.append(r)
        else:
            for i in range(self.shape[0]):
                result.append(self.value[i] + another.value[i])
        return Matrix(result)

    def __sub__(self, another):
        """
        Matrix subtraction.
        >>> a = Matrix([[1,1,1],[2,2,2]])
        >>> b = Matrix([[3,3,3],[4,4,4]])
        >>> c = Matrix([[5,5],[6,6]])
        >>> a-b
        [[-2, -2, -2], [-2, -2, -2]]
        >>> a-c
        Traceback (most recent call last):
            ...
        AssertionError: The shapes of these two matrixes must be the same.

        :param another:Matrix
        :return:Matrix
        """
        assert self.shape == another.shape, "The shapes of these two matrixes must be the same."
        result = []
        if self.shape[1] > 0:
            for i in range(self.shape[0]):
                r = []
                for j in range(self.shape[1]):
                    r.append(self.value[i][j] - another.value[i][j])
                result.append(r)
        else:
            for i in range(self.shape[0]):
                result.append(self.value[i] - another.value[i])
        return Matrix(result)

    def sca_mul(self, another):
        """
        Matrix scalar multiplication.
        >>> a = Matrix([[1,2,3],[4,5,6]])
        >>> a.sca_mul(2)
        [[2, 4, 6], [8, 10, 12]]
        >>> a.sca_mul(1.5)
        [[1.5, 3.0, 4.5], [6.0, 7.5, 9.0]]
        >>> a.sca_mul([[1,2],[3,4]])
        Traceback (most recent call last):
            ...
        AssertionError: 'another' must be a number.

        :param another: int or float
        :return:Matrix
        """
        assert isinstance(another, int) or isinstance(another, float), "'another' must be a number."
        result = []
        if self.shape[1] > 0:
            for i in range(self.shape[0]):
                r = []
                for j in range(self.shape[1]):
                    r.append(self.value[i][j] * another)
                result.append(r)
        else:
            for i in range(self.shape[0]):
                result.append(self.value[i] * another)
        return Matrix(result)

    def __truediv__(self, another):
        """
        Matrix division.
        >>> a = Matrix([[1,2,3],[4,5,6]])
        >>> a.sca_mul(1.5)
        [[1.5, 3.0, 4.5], [6.0, 7.5, 9.0]]
        >>> a = Matrix([[1,2,3],[3,4,5]])
        >>> a/2
        [[0.5, 1.0, 1.5], [1.5, 2.0, 2.5]]
        >>> a/(a/2)
        Traceback (most recent call last):
            ...
        AssertionError: 'another' must be a number.

        :param another: int or float
        :return:Matrix
        """
        assert isinstance(another, int) or isinstance(another, float), "'another' must be a number."
        result = []
        if self.shape[1] > 0:
            for i in range(self.shape[0]):
                r = []
                for j in range(self.shape[1]):
                    r.append(self.value[i][j] / another)
                result.append(r)
        else:
            for i in range(self.shape[0]):
                result.append(self.value[i] / another)
        return Matrix(result)

    def __mul__(self, another):
        """
        Matrix multiplication.
        >>> a = Matrix([[1,2,3],[2,3,4]])
        >>> b = Matrix([[3,1,2,5],[-2,4,-1,2],[1,4,-3,2]])
        >>> c = Matrix([[1,1],[2,2]])
        >>> a*b
        [[2, 21, -9, 15], [4, 30, -11, 24]]
        >>> a*c
        Traceback (most recent call last):
            ...
        AssertionError: The number of columns in the first matrix must be equal to the number of rows in the second matrix.
        """
        assert self.shape[1] == another.shape[0], \
            "The number of columns in the first matrix must be equal to the number of rows in the second matrix."
        result = []
        a = self.value
        b = another.value
        for i in range(self.shape[0]):
            r = []
            for p in range(another.shape[1]):
                temp = 0
                for j in range(self.shape[1]):
                    temp += a[i][j] * b[j][p]
                r.append(temp)
            result.append(r)
        return Matrix(result)

    def dot_mul(self, another):
        """
        Matrix dot product.
        >>> a = Matrix([[1,2,3],[2,3,4]])
        >>> b = Matrix([[2,2,2],[5,5,5]])
        >>> c = Matrix([[2],[5]])
        >>> a.dot_mul(b)
        [[2, 4, 6], [10, 15, 20]]
        >>> a.dot_mul(c)
        Traceback (most recent call last):
            ...
        AssertionError: The shapes of these two matrixes must be same.

        :param another: Matrix
        :return: Matrix
        """
        assert self.shape == another.shape, "The shapes of these two matrixes must be same."
        result = []
        if self.shape[1] > 0:
            for i in range(self.shape[0]):
                r = []
                for j in range(self.shape[1]):
                    r.append(self.value[i][j] * another.value[i][j])
                result.append(r)
        else:
            for i in range(self.shape[0]):
                result.append(self.value[i] * another.value[i])
        return Matrix(result)

    def transpose(self):
        """
        Matrix transpose.
        >>> a = Matrix([[1,2,3],[2,3,4]])
        >>> a.transpose()
        [(1, 2), (2, 3), (3, 4)]

        :return: Matrix
        """
        return Matrix(list(zip(*self.value)))

    def __eq__(self, another):
        """
        Judge if the two matrixes are the same.
        >>> a = Matrix([[1,2,3],[2,3,4]])
        >>> b = Matrix([[1,2,3],[2,3,4]])
        >>> c = Matrix([[1,2],[3,4]])
        >>> a==b
        True
        >>> a==c
        False
        """
        return self.value == another.value


if __name__ == "__main__":
    import doctest

    doctest.testmod()
