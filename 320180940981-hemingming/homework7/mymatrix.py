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
        
        # A == B
        assert isinstance(N, Matrix), "Type mismatch, cannot compare"
        return N.shape == self.shape  
    
    def __add__(self, N):
        # A + B
        assert N.shape == self.shape, "Dimension mismatch, cannot add"
        M = Matrix(self.row, self.column)
        for r in range(self.row):
            for c in range(self.column):
                M[r, c] = self[r, c] + N[r, c]
        return M
    
    def __sub__(self, N):
        # A - B
        assert N.shape == self.shape, "Dimension mismatch, cannot subtract"
        M = Matrix(self.row, self.column)
        for r in range(self.row):
            for c in range(self.column):
                M[r, c] = self[r, c] - N[r, c]
        return M
    
    def __mul__(self, N):
        # A * B 
        if isinstance(N, int) or isinstance(N,float):
            M = Matrix(self.row, self.column)
            for r in range(self.row):
                for c in range(self.column):
                    M[r, c] = self[r, c]*N
        else:
            assert N.row == self.column, "Dimension mismatch, cannot multiply"
            M = Matrix(self.row, N.column)
            for r in range(self.row):
                for c in range(N.column):
                    sum = 0
                    for k in range(self.column):
                        sum += self[r, k] * N[k, r]
                    M[r, c] = sum
        return M
    
    def __div__(self, N):
        # A / B
        pass
    def __pow__(self, k):
        # A**k
        assert self.row == self.column, "It's not a square array. It can't be multiplied"
        M = copy.deepcopy(self)
        for i in range(k):
           M = M * self 
        return M

    def invert(self):
    
        assert self.row == self.column, "It's not a square array."
        M = Matrix(self.row, self.column*2)
        I = self.identity()
        I.show()#############################
        
        
        for r in range(1,M.row+1):
            temp = self[r]
            temp.extend(I[r])
            M[r] = copy.deepcopy(temp)
        M.show()############################# 
        
    def transpose(self):
        
        M = Matrix(self.column, self.row)
        for r in range(self.column):
            for c in range(self.row):
                M[r, c] = self[c, r]
        return M
    
    def zeros(self):
        
        M = Matrix(self.column, self.row, fill=0.0)
        return M
    
    def ones(self):
        
        M = Matrix(self.column, self.row, fill=1.0)
        return M
    
    def identity(self):
        
        assert self.row == self.column, "Not n*n matrix，no matrix"
        M = Matrix(self.column, self.row)
        for r in range(self.row):
            for c in range(self.column):
                M[r, c] = 1.0 if r == c else 0.0
        return M
    
    def show(self):
       
        for r in range(self.row):
            for c in range(self.column):
                print(self[r+1, c+1],end='  ')
            print()
    

if __name__ == '__main__':
    m = Matrix(3,3,fill=2.0)
    n = Matrix(3,3,fill=3.5)

   

    m[1] = [1.,3.,2.]
    m[2] = [1.,2.,1.]
    m[3] = [2.,1.,1.]
    n[1] = [1.,2.,3.]
    n[2] = [2.,2.,2.]
    n[3] = [3.,2.,1.]
    
    a = m+n
    b = m-n
    c = m*n
    d = m*3
    f = m.transpose()
    g = m**3
'''
>>> a.show()
2.0  5.0  5.0  
3.0  4.0  3.0  
5.0  3.0  2.0  
>>> b.show()
0.0  1.0  -1.0  
-1.0  0.0  -1.0  
-1.0  -1.0  0.0  
>>> c.show()
13.0  13.0  13.0  
8.0  8.0  8.0  
9.0  9.0  9.0  
>>> d.show()
3.0  9.0  6.0  
3.0  6.0  3.0  
6.0  3.0  3.0  
>>> f.show()
1.0  1.0  2.0  
3.0  2.0  1.0  
2.0  1.0  1.0  
>>> g.show()
128.0  128.0  128.0  
288.0  288.0  288.0  
96.0  96.0  96.0  
'''
