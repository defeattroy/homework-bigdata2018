class mymatrix(list):
    """
    This is a two-dimensional matrix.
    So you must create it with a two-dimensional array,such as '[[],[]]'

    You can use print() to output the matrix.
    Or call the instance to describe the row and column of the matrix.
    Overloaded the two opertors.
    __str__
    __call__
    Output mode:
    >>> a = mymatrix([[1,2],[3,4]])
    >>> print(a) #doctest:+NORMALIZE_WHITESPACE
    1 2
    3 4
    >>> a()
    This is a 2x2 matrix!

    Overloaded index and slice operators enable the matrix to support index value and index assignment operations
    __getitem__()
    __setitem__()

    We implemented the function of matrix addition, subtraction, multiplication and division by overloading operators

    __add__       a+b
    __sub__       a-b
    __mul__       a*b (b:a number or a Matrix)
    __truediv__   a/b(b:must be a number)
    __floordiv__  a//b((b:must be a number)
    __pow__       a**b(b:must be a number)

    Realize specific functions:
    a.transpose() get a transposed matrix
    a.point_mul(b) each elements of a mutiple with each elements of b(the same shape with a and b)
    mymatrix.stable_matrix(cls,row,column,t) Create a matrix with a certain row and a certain column
    mymatrix.eye_matrix(cls,row) Create an n-dimensional identity matrix
    a.determinant() Get the determinant of a(the square matrix)
    a.inverse() Get the inverse matrix of a(the square matrix and determinant does not equal 0)
    """
    def __init__(self,llist):
        """
        >>> m=mymatrix([[1,2,3],[4,5,6]])
        >>> m.shape,m.row,m.column,m.matrix
        ((2, 3), 2, 3, [[1, 2, 3], [4, 5, 6]])

        >>> m=mymatrix([1,2,3])
        Error input! Please try a two-dimensional array again!

        >>> m=mymatrix(2)
        Traceback (most recent call last):
        ...
        AssertionError: Matrix creation must be a two-dimensional array

        >>> m=mymatrix([[1,2],[3]])
        Error input! Please try a two-dimensional array again!
        """
        assert isinstance(llist,list),'Matrix creation must be a two-dimensional array'

        try:
            self.__column = len(llist[0])
            for i in range(len(llist)):
                if len(llist[i])==self.__column:
                    continue
                else:
                    print("Error input! Please try a two-dimensional array again!")
                    break
            self.__matrix = llist
            self.__row = len(llist)
            self.__shape = (self.__row, self.__column)
        except Exception as e:
            print('Error input! Please try a two-dimensional array again!')

    @property
    def row(self):
        return self.__row

    @property
    def column(self):
        return  self.__column

    @property
    def shape(self):
        return  self.__shape

    @property
    def matrix(self):
        return self.__matrix

    def __str__(self):
        """
        When you print mymatrix, you can get a matrix with the element.
        >>> a = mymatrix([[1,2],[3,4]])
        >>> print(a) #doctest:+NORMALIZE_WHITESPACE
        1 2
        3 4
        """
        message = ''
        for r in range(self.row):
            for c in range(self.column):
                message += str(self.matrix[r][c]) + ' '
            message += '\n'
        return message

    def __call__(self, *args, **kwargs):
        """
        Because of the function,you can call the mymatrix()
        >>> a = mymatrix([[1,2],[3,4]])
        >>> a()#doctest:+NORMALIZE_WHITESPACE
        This is a 2x2 matrix!
        """
        print("This is a {0}x{1} matrix!".format(self.row,self.column))

    def __getitem__(self, x):
        """
        You can use[] to visit the element of the matrix like the list.
        >>> a = mymatrix([[1,2],[3,4]])
        >>> a[0][0]
        1
        >>> a[0]
        [1, 2]
        """
        return self.__matrix[x]

    def __setitem__(self, x, v):
        """
        This function is just used to change the element of one row by a list
        or a specific element in specific row and column by one data.
        >>> a = mymatrix([[1,2],[3,4]])
        >>> a[0][0]
        1
        >>> a[0][0]=20
        >>> a[0][0]
        20
        >>> a[0]
        [20, 2]
        >>> a[0] = [5,6]
        >>> a[0]
        [5, 6]
        """
        self.__matrix[x] = v

    @classmethod
    def stable_matrix(cls,row,column,t):
        """
        This is a classmethod.You don't have to create a instance to use the function.
        You can get a certain list with t.
        >>> t = mymatrix.stable_matrix(3,4,0)
        >>> print(t)
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        """
        stable_list = []
        for r in range(row):
            stable_list.append([])
            for c in range(column):
                stable_list[r].append(t)
        return stable_list

    @classmethod
    def eye_matrix(cls,row):
        """
        This is a classmethod.You don't have to create a instance to use the function.
        You can get a certain dismension list just like a indentity matrix.
        >>> e = mymatrix.eye_matrix(2)
        >>> e
        [[1, 0], [0, 1]]
        """
        eye_list = []
        for r in range(row):
            eye_list.append([])
            for c in range(row):
                if r==c:
                    eye_list[r].append(1)
                else:
                    eye_list[r].append(0)
        return eye_list

    def transpose(self):
        """
        You can get a transposed matrix with original matrix.
        >>> a = mymatrix([[1,2],[3,4]])
        >>> b = a.transpose()
        >>> print(b)#doctest:+NORMALIZE_WHITESPACE
        1 3
        2 4
        """
        m = mymatrix(self.stable_matrix(self.column,self.row,0))
        for i in range(self.row):
            for j in range(self.column):
                m.matrix[j][i]=self.matrix[i][j]
        return m

    def  __eq__(self,other):
        """
        You can use == to compare two matrixes.
        >>> a = mymatrix([[1,2],[3,4]])
        >>> b = mymatrix([[4,3],[2,1]])
        >>> print(a==b)
        False
        >>> c = mymatrix([[1,2],[3,4]])
        >>> print(a==c)
        True
        """
        assert isinstance(other, mymatrix), 'Only two same dimension can be compared!'
        if self.row != other.row:
            return False
        if self.column != other.column:
            return  False
        for r in range(self.row):
            for c in range(self.column):
                if self.matrix[r][c] == other.matrix[r][c]:
                    continue
                else:
                    return False
        return True


    def __add__(self, other):
        """
        You can use + to add two matrixes and get the result.
        >>> a = mymatrix([[1,2],[3,4]])
        >>> b = mymatrix([[4,3],[2,1]])
        >>> print(a+b)#doctest:+NORMALIZE_WHITESPACE
        5 5
        5 5
        >>> print(a+3)
        Traceback (most recent call last):
        ...
        AssertionError: Only two same dimension can be added!
        >>> c = mymatrix([[1,2]])
        >>> print(a+c)
        Traceback (most recent call last):
        ...
        AssertionError: Only two same dimension can be added!
        """
        assert isinstance(other, mymatrix),'Only two same dimension can be added!'
        assert self.row == other.row and self.column == other.column,'Only two same dimension can be added!'

        m  = mymatrix(self.stable_matrix(self.row,self.column,0))

        for r in range(self.row):
            for c in range(self.column):
                m.matrix[r][c] = self.matrix[r][c]+other.matrix[r][c]
        return m

    def __sub__(self, other):
        """
        You can use - to substract two matrixes and get the result.
        >>> a = mymatrix([[1,2],[3,4]])
        >>> b = mymatrix([[4,3],[2,1]])
        >>> print(a-b)#doctest:+NORMALIZE_WHITESPACE
        -3 -1
        1 3
        >>> print(a-3)
        Traceback (most recent call last):
        ...
        AssertionError: Only two same dimension can be subed!
        >>> c = mymatrix([[1,2]])
        >>> print(a-c)
        Traceback (most recent call last):
        ...
        AssertionError: Only two same dimension can be subed!
        """
        assert isinstance(other, mymatrix),'Only two same dimension can be subed!'
        assert self.row == other.row and self.column == other.column,'Only two same dimension can be subed!'

        m = mymatrix(self.stable_matrix(self.row,self.column,0))

        for r in range(self.row):
            for c in range(self.column):
                m.matrix[r][c] = self.matrix[r][c]-other.matrix[r][c]
        return m

    def __mul__(self, other):
        """
        You can use * to multiply two matrixes or one matrix with a number.
        >>> a = mymatrix([[1,2],[3,4]])
        >>> b = mymatrix([[4,3],[2,1]])
        >>> print(a*b)#doctest:+NORMALIZE_WHITESPACE
        8 5
        20 13
        >>> print(a*3)#doctest:+NORMALIZE_WHITESPACE
        3 6
        9 12
        >>> c = mymatrix([[1,2]])
        >>> print(a*c)
        Traceback (most recent call last):
        ...
        AssertionError: Only two same dimension can be muled!
        """
        if isinstance(other,mymatrix):
            assert self.row == other.column and self.column == other.row,'Only two same dimension can be muled!'

            m = mymatrix(self.stable_matrix(self.row,other.column,0))

            for r in range(self.row):
                for c in range(other.column):
                    for x in range(self.column):
                         m.matrix[r][c] += self.matrix[r][x]*other.matrix[x][c]
            return m
        elif isinstance(other,int) or isinstance(other,float):
            m = mymatrix(self.stable_matrix(self.row, self.column, 0))
            for r in range(self.row):
                for c in range(self.column):
                    m.matrix[r][c] = self.matrix[r][c]*other
            return m

    def point_mul(self,other):
        """
        It corresponds to every element to mul for two matrixes with the same shape.
        >>> a = mymatrix([[1,2],[3,4]])
        >>> b = mymatrix([[4,3],[2,1]])
        >>> print(a.point_mul(b))#doctest:+NORMALIZE_WHITESPACE
        4 6
        6 4
        >>> c = mymatrix([[1,2]])
        >>> print(a.point_mul(c))#doctest:+NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        AssertionError: Only two same dimension can be muled!
        """
        assert isinstance(other,mymatrix)
        assert self.row == other.row and self.column == other.column,'Only two same dimension can be muled!'

        m = mymatrix(self.stable_matrix(self.row, self.column, 0))

        for r in range(self.row):
            for c in range(self.column):
                m.matrix[r][c] = self.matrix[r][c]*other.matrix[r][c]
        return m

    def __truediv__(self, other):
        """
        You can use / to divide one matrix with a number.
        >>> a = mymatrix([[3,6],[9,12]])
        >>> print(a/3)#doctest:+NORMALIZE_WHITESPACE
        1.0 2.0
        3.0 4.0
        >>> c = mymatrix([[1,2]])
        >>> print(a/c)
        Traceback (most recent call last):
        ...
        AssertionError: Only a number can be divied!
        """
        assert isinstance(other, int) or isinstance(other, float),'Only a number can be divied!'
        m = mymatrix(self.stable_matrix(self.row, self.column, 0))

        for r in range(self.row):
            for c in range(self.column):
                m.matrix[r][c] = self.matrix[r][c]/other
        return m

    def __floordiv__(self, other):
        """
        You can use / to floor divide one matrix with a number.
        >>> a = mymatrix([[3,6],[9,12]])
        >>> print(a//3)#doctest:+NORMALIZE_WHITESPACE
        1 2
        3 4
        >>> c = mymatrix([[1,2]])
        >>> print(a//c)
        Traceback (most recent call last):
        ...
        AssertionError: Only a number can be floor divied!
        """
        assert isinstance(other, int) or isinstance(other, float),'Only a number can be floor divied!'

        m = mymatrix(self.stable_matrix(self.row, self.column, 0))

        for r in range(self.row):
            for c in range(self.column):
                m.matrix[r][c] = self.matrix[r][c] // other
        return m

    def __pow__(self, power):
        """
        You can use ** to power one square matrix.
        >>> a = mymatrix([[3,6],[9,12]])
        >>> print(a**3)#doctest:+NORMALIZE_WHITESPACE
        999 1458
        2187 3186
        >>> c = mymatrix([[1,2]])
        >>> print(c**2)
        Traceback (most recent call last):
        ...
        AssertionError: Only square matrix can be powered!
        """
        assert self.row == self.column,'Only square matrix can be powered!'
        e = mymatrix(self.eye_matrix(self.row))
        for i in range(power):
            e = e*self
        return e

    def yuzi(self, row, column):  #
        """
        You can get the cofactor of determinant of a square matrix.

        >>> a=mymatrix([[1,12,3,4],[4,56,-1,34],[23,13,12,0],[2,12,42,53]])
        >>> print(a.yuzi(0,0))#doctest:+NORMALIZE_WHITESPACE
        56 -1 34
        13 12 0
        12 42 53
        >>> print(a.yuzi(4,4))
        Traceback (most recent call last):
        ...
        AssertionError: Overindex!
        """
        assert self.row == self.column,'You can onlu use the function to operate the square matrix!'
        assert self.row >= 2,'You cannot operate 1 dimension square matrix!'
        assert self.row - 1 >= row and self.column - 1 >= column,'Overindex!'
        rm = mymatrix(self.stable_matrix(self.row - 1, self.column - 1,0))
        for r in range(self.row):
            if r == row:
                continue
            for c in range(self.column):
                if c == column:
                    continue
                if r < row:
                    x = r
                else:
                    x = r-1
                if c < column:
                    y = c
                else:
                    y = c-1
                rm.matrix[x][y] = self.matrix[r][c]
        return rm

    def determinant(self):
        """
        This is the determinant of square matrix,which we use the
        >>> a=mymatrix([[1,3,2,4],[5,2,-1,3],[3,1,1,2],[2,1,2,1]])
        >>> a.determinant()
        24
        >>> b = mymatrix([[1,2,3],[2,2,2]])
        >>> b.determinant()
        Traceback (most recent call last):
        ...
        AssertionError: You can only operate the square matrix!
        """
        assert self.row == self.column,'You can only operate the square matrix!'
        if self.row == 1:
            return self.matrix[0][0]
        else:
            result = 0
            for r in range(self.row):
                result += ((-1)**r) * self.yuzi(0,r).determinant() * self.matrix[0][r]
        return result

    def inverse(self):
        """
        You can use the yuzishi to get the adjoint matrix.And then you can get an inverse matrix!
        >>> a = mymatrix([[-1,2,-6],[-1,2,-7],[1,-1,2]])
        >>> print(a.inverse())#doctest:+NORMALIZE_WHITESPACE
        3.0 -2.0 2.0
        5.0 -4.0 1.0
        1.0 -1.0 -0.0
        >>> b = mymatrix([[1,2,3],[2,2,2]])
        >>> b.inverse()
        Traceback (most recent call last):
        ...
        AssertionError: Only square matrix has the inverse matrix!
        """
        assert self.row == self.column,'Only square matrix has the inverse matrix!'
        assert self.determinant() != 0,'The matrix does not have a matrix!'
        rm = mymatrix(self.stable_matrix(self.row,self.column,0))
        for r in range(self.row):
            for c in range(self.column):
                rm.matrix[r][c] = (-1) ** (r + c) * self.yuzi(c, r).determinant()
        rm = rm*(1 / self.determinant())
        return rm


if __name__=='__main__':
    import doctest
    doctest.testmod()