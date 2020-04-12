class mymatrix(list):

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
        return matrix coefficient
        >>> a = mymatrix([[1,2],[3,4]])
        >>> a()#doctest:+NORMALIZE_WHITESPACE
        This is a 2x2 matrix!
        """
        print("This is a {0}x{1} matrix!".format(self.row,self.column))

    def __getitem__(self, x):
        """
        give the position to get the matrix elements
        >>> a = mymatrix([[1,2],[3,4]])
        >>> a[0][0]
        1
        >>> a[0]
        [1, 2]
        """
        return self.__matrix[x]

    def __setitem__(self, x, v):
        """
        change matrix elements
        >>> a = mymatrix([[1,2],[3,4]])
        >>> a[0][0]
        1
        >>> a[0][0]=9
        >>> a[0][0]
        9
        >>> a[0]
        [9, 2]
        >>> a[0] = [5,6]
        >>> a[0]
        [5, 6]
        """
        self.__matrix[x] = v

    @classmethod
    def stable_matrix(cls,row,column,t):
        """
        certain a list with t.
        >>> t = mymatrix.stable_matrix(2,4,0)
        >>> print(t)
        [[0, 0, 0, 0], [0, 0, 0, 0]]
        """
        stable_list = []
        for r in range(row):
            stable_list.append([])
            for c in range(column):
                stable_list[r].append(t)
        return stable_list

    def transpose(self):
        """
        You can get a transposed matrix
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
        dot product
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
        divide a num
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

    @classmethod
    def eye_matrix(cls, row):
        """
        get a indentity matrix.
        >>> e = mymatrix.eye_matrix(2)
        >>> e
        [[1, 0], [0, 1]]
        """
        eye_list = []
        for r in range(row):
            eye_list.append([])
            for c in range(row):
                if r == c:
                    eye_list[r].append(1)
                else:
                    eye_list[r].append(0)
        return eye_list

    def __pow__(self, power):
        """
        ** to power one square matrix.
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

if __name__=='__main__':
    import doctest
    doctest.testmod()
