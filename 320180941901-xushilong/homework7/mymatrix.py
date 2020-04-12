# Author: XuShilong
# Time: 2020/3/19

import copy


class Matrix:
    """
    This class implements matrix addition, subtraction, dot multiplication, cross multiplication,
    permutation, comparison and other operations. This class also implements the slicing style of the matrix.

    This Matrix class can do the following operations:
    a=a+b , a+=b
    a=a-b , a-=b
    a=a*b , a*=b
    a=a.dot(b) , a.idot(b)
    a=a.mul(b) , a.imul(b)
    a.transpose()
    a.reset_Matrix()
    a.get_identity_matrix()
    a.tolist()

    """

    def __init__ (self, row: int = None, col: int = None, filler=0.0, datalist: list = None):
        """
        This is a matrix initialization method, which provides two methods based on
        row and column and padding value initialization and two-dimensional list initialization.

        >>> a=Matrix(3,3,2)
        >>> a
        2 2 2
        2 2 2
        2 2 2

        >>> b=Matrix(datalist=[[1,2,3],[4,5,6],[7,8,9]])
        >>> b
        1 2 3
        4 5 6
        7 8 9

        :param row: row
        :param col: column
        :param filler: the number fills the Matrix
        :param datalist: create a Matrix from a 2-d list
        """
        assert isinstance(row, int) or row is None, "row must be Integer！"  # make sure that row and col is int
        assert isinstance(col, int) or col is None, "col must be Integer！"
        assert (datalist is None) or isinstance(datalist, list) or \
               isinstance(datalist, tuple), "datalist must be list or tuple!"  # make sure that data is list or None
        self.row = row
        self.col = col
        self.shape = (row, col)
        if datalist is None:
            self._datalist = [[filler] * col for i in range(row)]
        else:
            self._datalist = datalist
            self.row = len(datalist)
            self.col = len(datalist[0])
            self.shape = (self.row, self.col)

    def __getitem__ (self, item):
        """
        This method achieves geting the value of the corresponding position through an integer pair,
        At the same time, this method allows the matrix class to support the complete slice style.
        >>> a=Matrix(datalist=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        >>> a[1,4]
        4
        >>> a[2:3,2:-2]
        6	7
        10	11
        >>> a[1,:]
        1	2	3	4
        >>> a[:,2]
        2
        6
        10
        14

        :param item: integer pair or slice
        :return: a Matrix if slice is given or the value of corresponding position if integer pair is given
        """

        def check (s1, e1):  # Check if the index is negative, if it is negative, reverse index
            s1 = self.row + s1 if s1 < 0 else s1
            e1 = self.col + e1 + 1 if e1 < 0 else e1
            return s1, e1

        if isinstance(item, int):  # Return one line directly
            return Matrix(datalist=[self._datalist[item - 1]])
        else:
            if isinstance(item[0], slice) and isinstance(item[1], int):  # Row slice or column slice or both slice
                s1 = 0 if item[0].start is None else item[0].start - 1
                e1 = self.row if item[0].stop is None else item[0].stop
                s1, e1 = check(s1, e1)  # Check if the index is negative
                rangetuple = (s1, e1) if item[0].step is None else (s1, e1, item[0].step)
                return Matrix(datalist=[[self._datalist[i][item[1] - 1]] for i in range(*rangetuple)])

            elif isinstance(item[1], slice) and isinstance(item[0], int):
                s2 = 0 if item[1].start is None else item[1].start - 1
                e2 = self.col if item[1].stop is None else item[1].stop
                s2, e2 = check(s2, e2)  # Check if the index is negative
                rangetuple = (s2, e2) if item[1].step is None else (s2, e2, item[0].step)
                return Matrix(datalist=[[self._datalist[item[0] - 1][i] for i in range(*rangetuple)]])

            elif isinstance(item[1], slice) and isinstance(item[0], slice):
                s2 = 0 if item[1].start is None else item[1].start - 1
                e2 = self.col if item[1].stop is None else item[1].stop
                s1 = 0 if item[0].start is None else item[0].start - 1
                e1 = self.row if item[0].stop is None else item[0].stop
                s1, e1 = check(s1, e1)  # Check if the index is negative
                s2, e2 = check(s2, e2)
                rangetuple1 = (s1, e1) if item[0].step is None else (s1, e1, item[0].step)
                rangetuple2 = (s2, e2) if item[0].step is None else (s2, e2, item[1].step)
                return Matrix(
                    datalist=[[self._datalist[i][j] for j in range(*rangetuple2)] for i in range(*rangetuple1)])

            elif isinstance(item, tuple):
                assert isinstance(item[0], int) and isinstance(item[1], int), "Index must be integer!"
                return self._datalist[item[0] - 1][item[1] - 1]
            else:
                raise AssertionError("Index must be integer!")

    def __setitem__ (self, key, value):
        """
        This method allows the matrix class to index positions by integer pairs to assign values to specified positions
        >>> a=Matrix(datalist=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        >>> a[1,1]=666
        >>> a
        666	2	3	4
        5	6	7	8
        9	10	11	12
        13	14	15	16

        :param key: integer pairs
        :param value: the value to be assign to specified positions
        :return:None
        """
        assert isinstance(key, tuple) and isinstance(key[0], int) and isinstance(key[1],
                                                                                 int), "index must be integer tuple"
        assert isinstance(value, int) or isinstance(value, float)
        self._datalist[key[0] - 1][key[1] - 1] = value

    def __add__ (self, other):
        """
        This method allows two matrix instances with same shape to be added using the "+"

        >>> a = Matrix(2,2,1)
        >>> b = Matrix(2,2,2)
        >>> c = a+b
        >>> c
        3   3
        3   3

        :param other: other Matrix to be added
        :return: a new Matrix which is the result of add
        """
        assert isinstance(other, Matrix), "Unsupported data format！"
        assert self.shape == other.shape, "Different shape!"  # Make sure it's a matrix and make sure it's the same shape
        return Matrix(datalist=[[self._datalist[i][j] + other._datalist[i][j]
                                 for j in range(self.col)] for i in range(self.row)])

    def __iadd__ (self, other):
        """
        This method allows two matrix instances with same shape to be added using the "+="

        >>> a=Matrix(2,2,1)
        >>> b=Matrix(2,2,2)
        >>> a+=b
        >>> a
        3 3
        3 3

        :param other: other Matrix to be added to this Matrix
        :return: this Matrix which is the result of add
        """
        self._datalist = (self + other)._datalist
        return self

    def __sub__ (self, other):
        """
        This method allows two matrix instances with same shape to be subed using the "-"

        >>> a=Matrix(2,2,2)
        >>> b=Matrix(2,2,1)
        >>> c=a-b
        >>> c
        1	1
        1	1

        :param other: Another matrix to subtract from this matrix
        :return: a new Matrix which is the result of sub
        """
        assert isinstance(other, Matrix), "Unsupported data format！"
        assert self.shape == other.shape, "Different shape!"
        return Matrix(datalist=[[self._datalist[i][j] - other._datalist[i][j]
                                 for j in range(self.col)] for i in range(self.row)])

    def __isub__ (self, other):
        """
        This method allows two matrix instances with same shape to be subed using the "-="

        >>> a=Matrix(2,2,2)
        >>> b=Matrix(2,2,1)
        >>> a-=b
        >>> a
        1	1
        1	1

        :param other: Another matrix to subtract from this matrix
        :return: this Matrix which is the result of sub
        """
        self._datalist = (self - other)._datalist
        return self

    def __mul__ (self, other):
        """
        This method allows the matrix to do cross product by "*"

        >>> a=Matrix(2,2,2)
        >>> b=a*2
        >>> b
        4	4
        4	4

        >>> a=Matrix(2,2,2)
        >>> b=Matrix(2,2,3)
        >>> c=a*b
        >>> c
        6	6
        6	6

        :param other: an integer or a Matrix
        :return:a new Matrix which is the result of cross product
        """
        if isinstance(other, int) or isinstance(other, float):  # Realized multiplication with real numbers
            data = []
            for i in range(self.row):
                line = []
                for j in range(self.col):
                    line.append(self[i + 1, j + 1] * other)
                data.append(line)
            return Matrix(datalist=data)
        elif isinstance(other, Matrix):  # Realized multiplication with Matrix
            assert self.shape == other.shape, "shape error, check the shape of tow Matrix."
            data = []
            for i in range(self.row):
                line = []
                for j in range(other.col):
                    line.append(self[i + 1, j + 1] * other[i + 1, j + 1])
                data.append(line)
            return Matrix(datalist=data)
        else:
            raise AssertionError("Unsupported data format！")

    def __imul__ (self, other):
        """
        This method allows the matrix to do cross product by "*="

        >>> a=Matrix(2,2,2)
        >>> a*=2
        >>> a
        4	4
        4	4

        >>> a=Matrix(2,2,2)
        >>> b=Matrix(2,2,3)
        >>> a*=b
        >>> a
        6	6
        6	6

        :param other: an integer or a Matrix
        :return: this Matrix which is the result of cross product
        """
        tmp = (self * other)
        self._datalist = tmp._datalist
        self.shape = tmp.shape
        self.row = tmp.row
        self.col = tmp.row
        return self

    def __eq__ (self, other):
        """
        Allows to compare whether two matrices are equal by "=="
        >>> a=Matrix(2,2,1)
        >>> b=Matrix(2,2,1)
        >>> a==b
        True

        >>> a=Matrix(2,2,1)
        >>> b=Matrix(2,3,1)
        >>> a==b
        False

        >>> a=Matrix(2,2,1)
        >>> b=Matrix(2,2,2)
        >>> a==b
        False

        :param other:
        :return:
        """
        assert isinstance(other, Matrix), "Only tow Matrix can be compared."
        if other.shape != self.shape:
            return False
        for i in range(self.row):
            for j in range(self.col):
                if self[i, j] != other[i, j]:
                    return False
        return True

    def mul (self, other):
        """
        Allows users to call multiplication as a function, with the same effect and implementation as" __mul__"

        >>> a=Matrix(2,2,2)
        >>> b=a.mul(2)
        >>> b
        4 4
        4 4

        >>> a=Matrix(2,2,2)
        >>> b=Matrix(2,2,3)
        >>> c=a.mul(b)
        >>> c
        6   6
        6   6

        :param other: an integer or a Matrix
        :return:a new Matrix which is the result of cross product
        """
        return self * other

    def imul (self, other):
        """
        Allows users to call i-multiplication as a function, with the same effect and implementation as" __imul__"

        >>> a=Matrix(2,2,2)
        >>> a.imul(2)
        4 4
        4 4

        >>> a=Matrix(2,2,2)
        >>> b=Matrix(2,2,3)
        >>> a.imul(b)
        6	6
        6	6

        :param other: an integer or a Matrix
        :return: this Matrix which is the result of cross product
        """
        return self.__imul__(other)

    def dot (self, other):
        """
        This method implements dot multiplication of matrix classes.

        >>> a=Matrix(2,2,1)
        >>> b=Matrix(2,2,2)
        >>> c=a.dot(b)
        >>> c
        4 4
        4 4

        :param other: a Matrix
        :return:a new Matrix which is the result of dot multiplication
        """
        assert isinstance(other, Matrix), "Unsupported data format！you can only dot with Matrix!"
        assert self.col == other.row, "shape error, check the shape of tow Matrix."
        data = []
        for i in range(self.row):
            line = []
            for j in range(other.col):
                tmp = 0
                for k in range(self.col):
                    tmp += self[i + 1, k + 1] * other[k + 1, j + 1]
                line.append(tmp)
            data.append(line)
        return Matrix(datalist=data)

    def idot (self, other):
        """
        This method implements dot multiplication of matrix classes,but the result is this Matrix.

        >>> a=Matrix(2,2,1)
        >>> b=Matrix(2,2,2)
        >>> a.idot(b)
        4	4
        4	4

        :param other: a Matrix
        :return:this Matrix which is the result of dot multiplication.
        """
        tmp = (self.dot(other))
        self._datalist = tmp._datalist
        self.shape = tmp.shape
        self.row = tmp.row
        self.col = tmp.row
        return self

    def transpose (self):
        """
        This method performs a transpose operation on the matrix.

        >>> a=Matrix(datalist=[[1,2,3],[4,5,6],[7,8,9]])
        >>> a
        1 2 3
        4 5 6
        7 8 9

        >>> b = Matrix(datalist=[[1,2,3],[4,5,6],[7,8,9]])
        >>> b.transpose()
        >>> b
        1   4   7
        2   5   8
        3   6   9

        :return:None
        """
        for i in range(self.row):
            for j in range(i, self.col):
                self[j + 1, i + 1], self[i + 1, j + 1] = self[i + 1, j + 1], self[j + 1, i + 1]

    def reset_Matrix (self, datalist):
        """
        This method can re-assign a matrix using a two-dimensional list, and the shape of the matrix will also change.

        >>> a=Matrix(2,2,1)
        >>> print(a,a.shape)
        1	1
        1	1	 (2, 2)
        >>> a.reset_Matrix([[6,6,6],[6,6,6]])
        >>> print(a,a.shape)
        6	6	6
        6	6	6	 (2, 3)

        :param datalist: a two-dimensional list
        :return: None
        """
        assert isinstance(datalist, list) or isinstance(datalist, Matrix), "Only support list and Matrix format."
        if isinstance(datalist, list):
            self._datalist = datalist
            self.row = len(datalist)
            self.col = len(datalist[0])
            self.shape = (self.row, self.col)
        if isinstance(datalist, Matrix):
            self = copy.deepcopy(datalist)

    @staticmethod
    def get_identity_matrix (n: int):
        """
        create a identity matrix.
        >>> a=Matrix.get_identity_matrix(5)
        >>> a
        1	0	0	0	0
        0	1	0	0	0
        0	0	1	0	0
        0	0	0	1	0
        0	0	0	0	1

        :param n: an integer
        :return: n-th order identity matrix
        """
        assert isinstance(n, int), "n must be integer!"
        return Matrix(datalist=[[1 if i == j else 0 for i in range(n)] for j in range(n)])

    def __str__ (self):
        line = ""
        for i in range(self.row):
            for j in range(self.col):
                line += str(self._datalist[i][j]) + "\t"
            line += "\n" if i != self.row - 1 else ""
        return line

    def __repr__ (self):
        return self.__str__()

    def tolist (self):
        """
        Convert matrix to list

        >>> a=Matrix(2,2,1)
        >>> print(type(a.tolist()),a.tolist())
        <class 'list'> [[1, 1], [1, 1]]

        :return: list format of Matrix
        """
        return self._datalist


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE)
