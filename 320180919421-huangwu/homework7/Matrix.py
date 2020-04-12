class Matrix():
    """
    Matrix creation must be a two-dimensional array,such as '[[]]'
    The subscripts of the columns and rows start from 0

    Output the matrix in a customized way
    __repr__
    __call__
    Output mode:
    Matrix([[a,b],[c,d]])
    a  b
    c  c

    Overloaded index and slice operators enable the matrix to support index value and index assignment operations
    __getitem__()
    __setitem__()

    We implement the function of matrix addition, subtraction, multiplication and division by overloading operators
    (the division between matrices is meaningless, so this function is not implemented)

    a.__add__(b)       a+b
    a.__sub__(b)       a-b
    a.__mul__(b)       a*b (b:number or Matrix)
    a.__truediv__(b)   a/b(b:must be number)
    a.__pow__(b)       a**b(b:number)

    Realize specific functions:

    a.indentity()     indentity matrix
    a.cofactor(i,j)   Cofactor of square matrix
    a.det()           Value of determinant
    a.adjoin()        Adjoint matrix
    a.inverse()       Calculate the inverse matrix of a
    """

    def __init__(self, value):
        """
        >>> m=Matrix([[1,2,3],[4,5,6]])
        >>> m.shape,m.row,m.col,m._matrix
        ((2, 3), 2, 3, [[1, 2, 3], [4, 5, 6]])

        >>> m=Matrix([1,2,3])
        Traceback (most recent call last):
        ...
        AssertionError: Matrix creation must be a two-dimensional array

        >>> m=Matrix(2)
        Traceback (most recent call last):
        ...
        AssertionError: Matrix creation must be a two-dimensional array

        >>> m=Matrix([[1,2],[3]])
        Traceback (most recent call last):
        ...
        AssertionError: length of each row can't be different
        """
        assert isinstance(value, list) and isinstance(value[0], list), 'Matrix creation must be a two-dimensional array'
        assert len(set([len(i) for i in value])) == 1, "length of each row can't be different"
        self.shape = (len(value), len(value[0]))
        self.row = self.shape[0]
        self.col = self.shape[1]
        self._matrix = value

    def __repr__(self):
        """
        '__repr__' is used to customize and return the string representation of the instance object

        Through this method, we can see the row and column structure of the matrix and the location
        of the elements clearly and intuitively

        >>> a=Matrix([[1,2],[3,4]])
        >>> a   #doctest:+NORMALIZE_WHITESPACE
        Matrix([[1, 2], [3, 4]])
        1  2
        3  4
        """
        res = "Matrix({})\n".format(self._matrix)
        for i in range(self.row):
            for j in range(self.col):
                res += str(self._matrix[i][j]) + '  '
            res += '\n'
        return res

    def __call__(self):
        """
        The essence of '__call__()' is to turn a class into a function (so that
        instances of this class can be called like a function)

        By overloading '__call__', matrix class can be called in its own way
        and output relevant information(number of rows and columns)

        >>> a=Matrix([[1,2],[3,4]])
        >>> a()        #doctest:+NORMALIZE_WHITESPACE
        1  2
        3  4
        Row:2
        Col:2
        """
        for i in range(self.row):
            for j in range(self.col):
                print(self._matrix[i][j], end='  ')
            print()
        print('Row:{}\nCol:{}'.format(self.row, self.col))

    def __getitem__(self, i):
        """
        Overload value index

        >>> a=Matrix([[1,2,3],[4,5,6],[7,8,9]])
        >>> a[1,2]
        6
        """
        return self._matrix[i[0]][i[1]]

    def __setitem__(self, i, v):
        """
        Overload assignment index
        >>> a=Matrix([[1,2,3],[4,5,6],[7,8,9]])
        >>> a[1,2]=7
        >>> a()  #doctest:+NORMALIZE_WHITESPACE
        1  2  3
        4  5  7
        7  8  9
        Row:3
        Col:3
        """
        self._matrix[i[0]][i[1]] = v

    def zero_value_matrix(self, shape):
        """
        This is all zero matrix, which is used to initialize the matrix calculation function

        '+Normalize_whiteface' is used to remove leading and trailing spaces and extra line breaks.

        >>> a=Matrix([[1,2],[3,4]])
        >>> a.zero_value_matrix(a.shape)    #doctest:+NORMALIZE_WHITESPACE
        Matrix([[0, 0], [0, 0]])
        0  0
        0  0

        >>> a=Matrix([[1,2],[3,4]])
        >>> a.zero_value_matrix((1,2)) #doctest:+NORMALIZE_WHITESPACE
        Matrix([[0, 0]])
        0  0
        """
        assert isinstance(shape, tuple), 'Shape of matrix must be appointed as tuple'
        result_matrix = []
        for i in range(shape[0]):
            result_matrix.append([])
            for j in range(shape[1]):
                result_matrix[i].append(0)
        return Matrix(result_matrix)

    def __add__(self, second_matrix):
        """
        This is the addition of matrix

        >>> a=Matrix([[1,2],[3,4]])
        >>> b=Matrix([[1,2],[3,4]])
        >>> a.__add__(b)   #doctest:+NORMALIZE_WHITESPACE
        Matrix([[2, 4], [6, 8]])
        2  4
        6  8

        >>> a+b    #doctest:+NORMALIZE_WHITESPACE
        Matrix([[2, 4], [6, 8]])
        2  4
        6  8

        >>> a=Matrix([[1,2],[3,4]])
        >>> b=Matrix([[1],[3]])
        >>> a+b
        Traceback (most recent call last):
        ...
        AssertionError: Matrix must be of the same type

        >>> a=Matrix([[1,2],[3,4]])
        >>> a+1
        Traceback (most recent call last):
        ...
        AssertionError: second_matrix must be of Matrix type
        """
        assert isinstance(second_matrix, Matrix), "second_matrix must be of Matrix type"
        assert second_matrix.shape == self.shape, 'Matrix must be of the same type'
        result_matrix = self.zero_value_matrix(self.shape)
        for i in range(self.row):
            for j in range(self.col):
                result_matrix._matrix[i][j] = self._matrix[i][j] + second_matrix._matrix[i][j]
        return result_matrix

    def __sub__(self, second_matrix):
        """
        This is the subtraction of matrix

        >>> a=Matrix([[1,2],[3,4]])
        >>> b=Matrix([[1,2],[3,4]])
        >>> a.__sub__(b)  #doctest:+NORMALIZE_WHITESPACE
        Matrix([[0, 0], [0, 0]])
        0  0
        0  0

        >>> a-b  #doctest:+NORMALIZE_WHITESPACE
        Matrix([[0, 0], [0, 0]])
        0  0
        0  0
        """
        assert isinstance(second_matrix, Matrix), "second_matrix must be of Matrix type"
        assert second_matrix.shape == self.shape, 'Matrix must be of the same type'
        result_matrix = self.zero_value_matrix(self.shape)
        for i in range(self.row):
            for j in range(self.col):
                result_matrix._matrix[i][j] = self._matrix[i][j] - second_matrix._matrix[i][j]
        return result_matrix

    def __mul__(self, second):
        """
        Decide which method to call by judging whether it is a matrix or a number (multiplication)

        >>> a=Matrix([[1,2],[3,4]])
        >>> b=2
        >>> a*b   #doctest:+NORMALIZE_WHITESPACE
        Matrix([[2, 4], [6, 8]])
        2  4
        6  8

        >>> a=Matrix([[1,2],[3,4]])
        >>> b=Matrix([[1,2],[3,4]])
        >>> a*b  #doctest:+NORMALIZE_WHITESPACE
        Matrix([[7, 10], [15, 22]])
        7  10
        15  22
        """
        if isinstance(second, Matrix):
            return self.point_mul(second)
        elif isinstance(second, int) or isinstance(second, float):
            return self.num_mul(second)

    def point_mul(self, second_matrix):
        """
        This is the multiplication of matrix and matrix

        >>> a=Matrix([[1,2],[3,4]])
        >>> b=Matrix([[1,2],[3,4]])
        >>> a.point_mul(b) #doctest:+NORMALIZE_WHITESPACE
        Matrix([[7, 10], [15, 22]])
        7  10
        15  22

        >>> a=Matrix([[1,2],[3,4]])
        >>> b=Matrix([[1,2],[3,4],[5,6]])
        >>> a.point_mul(b)
        Traceback (most recent call last):
        ...
        AssertionError: the first matrix col and the second matrix row must be the same
        """
        assert isinstance(second_matrix, Matrix), "second_matrix must be of Matrix type"
        assert self.col == second_matrix.row,'the first matrix col and the second matrix row must be the same'
        result_matrix = self.zero_value_matrix((self.row, second_matrix.col))
        for i in range(self.row):
            for j in range(second_matrix.col):
                result_matrix._matrix[i][j] = sum(
                    [self._matrix[i][k] * second_matrix._matrix[k][j] for k in range(self.col)])
        return result_matrix

    def num_mul(self, number):
        """
        This is the multiplication of matrix and scalar

        >>> a=Matrix([[1,2],[3,4]])
        >>> a.num_mul(2) #doctest:+NORMALIZE_WHITESPACE
        Matrix([[2, 4], [6, 8]])
        2  4
        6  8

        """
        assert isinstance(number, int) or isinstance(number, float), "number must be of int or float type"
        result_matrix = self.zero_value_matrix(self.shape)
        result_matrix._matrix = [[self._matrix[i][j] * number for j in range(self.col)] for i in range(self.row)]
        return result_matrix

    def __truediv__(self, number):
        """
        >>> a=Matrix([[1,2],[3,4]])
        >>> a/0.5 #doctest:+NORMALIZE_WHITESPACE
        Matrix([[2.0, 4.0], [6.0, 8.0]])
        2.0  4.0
        6.0  8.0

        >>> a=Matrix([[1,2],[3,4]])
        >>> b=Matrix([[1,2],[3,4]])
        >>> a/b
        Traceback (most recent call last):
        ...
        AssertionError: number must be of int or float type
        """
        assert isinstance(number, int) or isinstance(number, float), "number must be of int or float type"
        result_matrix = self.num_mul(1 / number)
        return result_matrix

    def __pow__(self, exp):
        """
        >>> a=Matrix([[1,1,2],[1,2,1],[2,1,1]])
        >>> a.__pow__(2)   #doctest:+NORMALIZE_WHITESPACE
        Matrix([[6, 5, 5], [5, 6, 5], [5, 5, 6]])
        6  5  5
        5  6  5
        5  5  6

        >>> a=Matrix([[1,1,2],[1,2,1],[2,1,1]])
        >>> a**3   #doctest:+NORMALIZE_WHITESPACE
        Matrix([[21, 21, 22], [21, 22, 21], [22, 21, 21]])
        21  21  22
        21  22  21
        22  21  21

        >>> a=Matrix([[1,2,3],[2,3,4]])
        >>> a**2
        Traceback (most recent call last):
        ...
        AssertionError: Matrix must be square
        """
        assert self.row == self.col, 'Matrix must be square'
        result_matrix = self
        for i in range(exp - 1):
            result_matrix = result_matrix.point_mul(self)
        return result_matrix

    def transpose(self):  # matrix transpose
        """
        >>> a=Matrix([[1,2],[3,4]])
        >>> a.transpose()  #doctest:+NORMALIZE_WHITESPACE
        Matrix([[1, 3], [2, 4]])
        1  3
        2  4
        """
        result_matrix = self.zero_value_matrix((self.col, self.row))
        result_matrix._matrix = [[self._matrix[j][i] for j in range(self.row)] for i in range(self.col)]
        return result_matrix

    def indentity(self):  # indentity matrix
        """
        First, the number of rows and columns is determined by the original matrix, then all elements of the
        matrix are set to 0 by the 'zero_value_matrix' method, and then the elements on the main diagonal are
        set to 1 by the list derivation

        >>> a=Matrix([[1,2],[3,4]])
        >>> a.indentity() #doctest:+NORMALIZE_WHITESPACE
        Matrix([[1, 0], [0, 1]])
        1  0
        0  1
        """
        assert self.row == self.col, 'Matrix must be square'
        result_matrix = self.zero_value_matrix(self.shape)
        result_matrix._matrix = [[1 if i == j else 0 for j in range(self.row)] for i in range(self.col)]
        return result_matrix

    def cofactor(self, row, col):  # cofactor of determinant
        """The order of a square matrix should be greater than 1, and then the complementary formula
        of a square matrix can be obtained by cyclic and binomial operations

        >>> a=Matrix([[2,1,2,4],[3,2,-1,3],[2,1,1,0],[0,1,2,3]]) #doctest:+NORMALIZE_WHITESPACE
        >>> a.cofactor(1,1)  #doctest:+NORMALIZE_WHITESPACE
        Matrix([[2, 2, 4], [2, 1, 0], [0, 2, 3]])
        2  2  4
        2  1  0
        0  2  3

        >>> a=Matrix([[1]])
        >>> a.cofactor(0,0)
        Traceback (most recent call last):
        ...
        AssertionError: Only above the second order can there be a cofactor

        >>> a=Matrix([[2,1,2,4],[3,2,-1,3],[2,1,1,0],[0,1,2,3]])
        >>> a.cofactor(4,4)
        Traceback (most recent call last):
        ...
        AssertionError: Subscript out of range
        """
        assert self.row == self.col, 'Matrix must be square'
        assert self.row >= 2, 'Only above the second order can there be a cofactor'
        assert self.row - 1 >= row and self.col - 1 >= col, 'Subscript out of range'
        result_matrix = self.zero_value_matrix((self.row - 1, self.col - 1))
        for i in range(self.row):
            if i == row:
                continue
            for j in range(self.col):
                if j == col:
                    continue
                r = i if i < row else i - 1
                c = j if j < col else j - 1
                result_matrix._matrix[r][c] = self._matrix[i][j]
        return result_matrix

    def det(self):  # Value of determinant
        """
         we make sure that the matrix is a square matrix by asserting,and expand the square matrix according to the
         first line. Then we call the 'cofacator' function to find the cofactor and recursively find the value of
         the determinant

        >>> a=Matrix([[2,1,2,4],[3,2,-1,3],[2,1,1,0],[0,1,2,3]])
        >>> a.det()
        31
        """
        assert self.row == self.col, 'Matrix must be square'
        if self.row == 1:
            return self._matrix[0][0]
        elif self.row == 2:
            return self._matrix[0][0] * self._matrix[1][1] - self._matrix[0][1] * self._matrix[1][0]
        else:
            # Expand determinants by row
            return sum([((-1) ** i * self._matrix[0][i] * self.cofactor(0, i).det()) for i in range(self.row)])

    def adjoin(self):  # adjoin matrix
        """
        >>> b=Matrix([[2,2,3],[1,-1,0],[-1,2,1]])
        >>> b.adjoin()   #doctest:+NORMALIZE_WHITESPACE
        Matrix([[-1, 4, 3], [-1, 5, 3], [1, -6, -4]])
        -1  4  3
        -1  5  3
        1  -6  -4
        """
        assert self.row == self.col, 'Matrix must be square'
        result_matrix = self.zero_value_matrix(self.shape)
        for i in range(self.row):
            for j in range(self.col):
                result_matrix._matrix[i][j] = (-1) ** (i + j) * self.cofactor(j, i).det()
        return result_matrix

    def inverse(self):  # inverse matrix
        """
        according to A-1=A*/|A|

        >>> b=Matrix([[2,2,3],[1,-1,0],[-1,2,1]])
        >>> b.inverse()  #doctest:+NORMALIZE_WHITESPACE
        Matrix([[1.0, -4.0, -3.0], [1.0, -5.0, -3.0], [-1.0, 6.0, 4.0]])
        1.0  -4.0  -3.0
        1.0  -5.0  -3.0
        -1.0  6.0  4.0

        >>> a=Matrix([[1,0,3],[2,1,2],[3,2,1]])
        >>> a.inverse()
        The matrix has no inverse
        """

        assert self.row == self.col, 'Matrix must be square'
        if self.det() == 0:
            print('The matrix has no inverse')
        else:
            result_matrix = self.adjoin().num_mul(1 / self.det())
            return result_matrix


if __name__ == '__main__':
    import doctest

    doctest.testmod()
