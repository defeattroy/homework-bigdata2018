import copy

class Matrix:
    '''class Matrix
    Methods:
    __add__:Matrix addition.
    __sub__:Matrix subtraction.
    __mul__:Matrix multiplication.
    __pow__:Matrix power.
    __transpose__:Matrix transpose.
    '''
    def __init__(self, row, column, fill=0.0):
        self.shape = (row, column)
        self.row = row
        self.column = column
        self._matrix = [[fill]*column for i in range(row)]
        
    #return the value of element m(i,j):m[i, j]
    def __getitem__(self, index):
        if isinstance(index, int):
            return self._matrix[index-1]
        elif isinstance(index, tuple):
            return self._matrix[index[0]-1][index[1]-1]

    # set the value of element m(i,j) is s:  m[i, j] = s
    def __setitem__(self, index, value):
        if isinstance(index, int):
            self._matrix[index-1] = copy.deepcopy(value)
        elif isinstance(index, tuple):
            self._matrix[index[0]-1][index[1]-1] = value
    def __eq__(self, N):
        '''equal'''
        # A == B
        assert isinstance(N, Matrix), "Type mismatch,cannot compare"
        return N.shape == self.shape  # Comparative dimension
    

        
    def __add__(self, N):
        
        # A + B
        assert N.shape == self.shape, "Dimension mismatch,cannot add"
        M = Matrix(self.row, self.column)
        for r in range(self.row):
            for c in range(self.column):
                M[r, c] = self[r, c] + N[r, c]
        return M
        '''
        >>> m = Matrix(3,3,fill=2.0)
        >>> n = Matrix(3,3,fill=3.5)
        >>> m[1] = [1,1,3]
        >>> m[2] = [1,2,1]
        >>> m[3] = [2,1,1]
        >>> m+n
        >>> [[4.5,4.5,6.5],[4.5,5.5,4.5],[5.5,4.5,4.5]]
        '''
    
    def __sub__(self, N):
        '''subtration'''
        # A - B
        assert N.shape == self.shape, "Dimension mismatch,cannot subtract"
        M = Matrix(self.row, self.column)
        for r in range(self.row):
            for c in range(self.column):
                M[r, c] = self[r, c] - N[r, c]
        return M
        '''
        >>> m = Matrix(3,3,fill=2.0)
        >>> n = Matrix(3,3,fill=3.5)
        >>> m[1] = [1,1,3]
        >>> m[2] = [1,2,1]
        >>> m[3] = [2,1,1]
        >>> m-n
        >>> [[-2.5,-2.5,-0.5],[-2.5,-1.5,-2.5],[-1.5,-2.5,-2.5]]
        '''
    
    def __mul__(self, N):
        '''multiplication'''
        # A * B (或：A * 2.0)
        if isinstance(N, int) or isinstance(N,float):
            M = Matrix(self.row, self.column)
            for r in range(self.row):
                for c in range(self.column):
                    M[r, c] = self[r, c]*N
        else:
            assert N.row == self.column, "Dimension mismatch,cannot multiply"
            M = Matrix(self.row, N.column)
            for r in range(self.row):
                for c in range(N.column):
                    sum = 0
                    for k in range(self.column):
                        sum += self[r, k] * N[k, r]
                    M[r, c] = sum
        return M
        '''
        >>> m = Matrix(3,3,fill=2.0)
        >>> n = Matrix(3,3,fill=3.5)
        >>> m[1] = [1,1,3]
        >>> m[2] = [1,2,1]
        >>> m[3] = [2,1,1]
        >>> m*n
        >>> [[17.5,17.5,17.5],[14.0,14.0,14.0],[14.0,14.0,14.0]]
        '''

    def __pow__(self, k):
        '''power'''
        # A**k
        assert self.row == self.column, "It's not a square,cannot power"
        M = copy.deepcopy(self)
        for i in range(k):
           M = M * self 
        return M
        '''
        >>> m = Matrix(3,3,fill=2.0)
        >>> m[1] = [1,1,3]
        >>> m[2] = [1,2,1]
        >>> m[3] = [2,1,1]
        >>> m**3
        >>> [[128,128,128],[96,96,96],[200,200,200]]
        '''

  
    def transpose(self):
        '''Transposition'''
        M = Matrix(self.column, self.row)
        for r in range(self.column):
            for c in range(self.row):
                M[r, c] = self[c, r]
        return M
        '''
        >>> m = Matrix(3,3,fill=2.0)
        >>> n = Matrix(3,3,fill=3.5)
        >>> m[1] = [1,1,3]
        >>> m[2] = [1,2,1]
        >>> m[3] = [2,1,1]
        >>> m.transpose()
        >>> [[1,1,2],[1,2,1],[3,1,1]]
        '''
    
    
    def show(self):
        '''print matrix'''
        for r in range(self.row):
            for c in range(self.column):
                print(self[r+1, c+1],end='  ')
            print()
    

if __name__ == '__main__':
    import doctest
    doctest.testmod() 

