import copy

class Matrix:
    
    def __init__(self, row, column, fill=0.0):
        self.shape = (row, column)
        self.row = row
        self.column = column
        self._matrix = [[fill]*column for i in range(row)]
        

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._matrix[index-1]
        elif isinstance(index, tuple):
            return self._matrix[index[0]-1][index[1]-1]


    def __setitem__(self, index, value):
        if isinstance(index, int):
            self._matrix[index-1] = copy.deepcopy(value)
        elif isinstance(index, tuple):
            self._matrix[index[0]-1][index[1]-1] = value
        
    def __eq__(self, N):
        '''相等'''
        # A == B
        assert isinstance(N, Matrix)
        return N.shape == self.shape  
    
    def __add__(self, N):
        '''加法'''
        # A + B
        assert N.shape == self.shape
        M = Matrix(self.row, self.column)
        for r in range(self.row):
            for c in range(self.column):
                M[r, c] = self[r, c] + N[r, c]
        return M
    
    def __sub__(self, N):
        '''减法'''
        # A - B
        assert N.shape == self.shape
        M = Matrix(self.row, self.column)
        for r in range(self.row):
            for c in range(self.column):
                M[r, c] = self[r, c] - N[r, c]
        return M
    
    def __mul__(self, N):
        '''乘法'''
        # A * B (或：A * 2.0)
        if isinstance(N, int) or isinstance(N,float):
            M = Matrix(self.row, self.column)
            for r in range(self.row):
                for c in range(self.column):
                    M[r, c] = self[r, c]*N
        else:
            assert N.row == self.column
            M = Matrix(self.row, N.column)
            for r in range(self.row):
                for c in range(N.column):
                    sum = 0
                    for k in range(self.column):
                        sum += self[r, k] * N[k, r]
                    M[r, c] = sum
        return M
    
    def __pow__(self, k):
        '''乘方'''
        # A**k
        assert self.row == self.column
        M = copy.deepcopy(self)
        for i in range(k):
           M = M * self 
        return M
    
    def transpose(self):
        '''转置'''
        M = Matrix(self.column, self.row)
        for r in range(self.column):
            for c in range(self.row):
                M[r, c] = self[c, r]
        return M
    
    def show(self):
        '''打印矩阵'''
        for r in range(self.row):
            for c in range(self.column):
                print(self[r+1, c+1],end='  ')
            print()
            
if __name__ == '__main__':
    m = Matrix(3,3,fill=2.0)
    n = Matrix(3,3,fill=3.5)

    m[1] = [1.,1.,2.]
    m[2] = [1.,2.,1.]
    m[3] = [2.,1.,1.]
    
    p = m * n
    q = m*2.1
    r = m**3
    r.show()
    print()
    q.show()
    print()
    print(p[1,1])
    print()
    m.show()
    print()
    r.show()
    print()    
