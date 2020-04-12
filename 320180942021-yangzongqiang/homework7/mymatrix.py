class Matrix:
    """This class implements matrix and partial operations of matrix

    Methods:
        __add__: Matrix addition.
        __sub__: Matrix subtraction.
        __mul__: Matrix scalar multiplication.
        dot_product: Matrix multiplication.
        transpose: Matrix transpose.
        __eq__: Judge whether the matrix is equal
        __str__: Print matrix details

     Attributes:
        X: Input matrix.
        row: Row of matrix.
        column: Column of matrix.

    >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> A.X
    [[1, 2, 3], [4, 5, 6]]
    >>> A.row
    2
    >>> A.column
    3
    """
    def __init__(self, X: list):
        """Inits Matrix"""
        self.X = X
        self.row = len(self.X)
        self.column = len(self.X[0])

    def __add__(self, other):
        """
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> B = Matrix([[7, 8, 9], [1, 2, 3]])
        >>> C = Matrix([[7, 8], [1, 2], [3, 4]])
        >>> A+B
        [[8, 10, 12], [5, 7, 9]]
        >>> A+C
        The shape of the two added matrices is different

        :param other: Matrix
        :return: Matrix
        """
        if self.row == other.row and self.column == other.column:
            added = []
            for i in range(0, self.row):
                _ = []
                for j in range(0, self.column):
                    _.append(self.X[i][j]+other.X[i][j])
                added.append(_)
            return added
        else:
            print("The shape of the two added matrices is different")
            return None

    def __sub__(self, other):
        """
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> B = Matrix([[7, 8, 9], [1, 2, 3]])
        >>> C = Matrix([[7, 8], [1, 2], [3, 4]])
        >>> A-B
        [[-6, -6, -6], [3, 3, 3]]
        >>> A-C
        The shape of the two subtractive matrices is different

        :param other: Matrix
        :return: Matrix
        """
        if self.row == other.row and self.column == other.column:
            subed = []
            for i in range(0, self.row):
                _ = []
                for j in range(0, self.column):
                    _.append(self.X[i][j]-other.X[i][j])
                subed.append(_)
            return subed
        else:
            print("The shape of the two subtractive matrices is different")
            return None

    def __mul__(self, other):
        """
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> A*2
        [[2, 4, 6], [8, 10, 12]]

        :return: Matrix
        """
        muled = []
        for i in range(0, self.row):
            _ = []
            for j in range(0, self.column):
                _.append(other * self.X[i][j])
            muled.append(_)
        return muled

    def __truediv__(self, other):
        """
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> A/2
        [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]

        :return: Matrix
        """
        if other == 0:
            print("Divisor is zero")
            return None
        truedived = []
        for i in range(0, self.row):
            _ = []
            for j in range(0, self.column):
                _.append(self.X[i][j]/other)
            truedived.append(_)
        return truedived

    def dot_product(self, other):
        """
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> B = Matrix([[7, 8, 9], [1, 2, 3]])
        >>> C = Matrix([[7, 8], [1, 2], [3, 4]])
        >>> A.dot_product(C)
        [[18, 24], [51, 66]]
        >>> A.dot_product(B)
        Inconsistent number of rows and columns

        :param other: Matrix
        :return: Matrix
        """
        if self.column == other.row:
            n = self.column
            doted = []
            for i in range(0, self.row):
                r = self.X[i]
                _ = []
                for j in range(0, other.column):
                    c = [other.X[k][j] for k in range(0, n)]
                    sum = 0
                    for l in range(0, n):
                        sum+=r[l]*c[l]
                    _.append(sum)
                doted.append(_)
            return doted
        else:
            print("Inconsistent number of rows and columns")
            return None

    def transpose(self):
        """
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> A.transpose()
        [(1, 4), (2, 5), (3, 6)]

        :return: Matrix
        """
        return list(zip(*(self.X)))

    def __eq__(self, other):
        """
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> C = Matrix([[7, 8], [1, 2], [3, 4]])
        >>> D = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> A==D
        True
        >>> A==C
        False

        :param other: Matrix
        :return: boolean
        """
        return self.X == other.X

    def __str__(self):
        """
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> print(A)
        ... # doctest: +NORMALIZE_WHITESPACE
        [[1, 2, 3], [4, 5, 6]]
        row：2   column: 3

        :return: str
        """
        return str(self.X) + "\nrow：" + str(self.row) + '\tcolumn: ' + str(self.column)


if __name__ == '__main__':
    import doctest
    doctest.testmod()