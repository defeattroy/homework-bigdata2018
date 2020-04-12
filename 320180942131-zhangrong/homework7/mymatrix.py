import copy
class Matrix:
    """Matrix class supports operations bellow:
    1 getitem: get the value
    2 setitem: set the value 
    3 equal or ''=': To judge whether or not
    4 add or '+':a.add(b) or a+b
    5 sub or '-':a.sub(b) or a-b
    6 mul or '*':a.mul(b) or a*b
    ...
    
    """
    # Create a matrix class
    def __init__(self, row, column, fill=0.0):
        self.shape = (row, column)
        self.row = row
        self.column = column
        self._matrix = [[fill]*column for i in range(row)]
        
        
    # Gets a tuple in the matrix：
    def __getitem__(self, index):
        if isinstance(index, int):
            return self._matrix[index-1]
        elif isinstance(index, tuple):#When is a tuple, it is equivalent to finding a specific value
            return self._matrix[index[0]-1][index[1]-1]

    # Sets the value of an element：
    def __setitem__(self, index, value):
        if isinstance(index, int):
            self._matrix[index-1] = copy.deepcopy(value)#深拷贝value（列表）里的元素
        elif isinstance(index, tuple):
            self._matrix[index[0]-1][index[1]-1] = value
    #  if these two matrices are equal
    def __eq__(self, N):
        # A == B
        assert isinstance(N, Matrix), "类型不同，不能比较"#断言是否同类型
        return N.shape == self.shape  
    # Addition of two matrices
    def __add__(self, N):
        # A + B
        assert N.shape == self.shape, "维度不同，不能相加"
        M = Matrix(self.row, self.column)
        for r in range(self.row):
            for c in range(self.column):
                M[r, c] = self[r, c] + N[r, c]
        return M
    
    #  Subtract two matrices 
    def __sub__(self, N):
        assert N.shape == self.shape, "维度同，不能相减"
        M = Matrix(self.row, self.column)
        for r in range(self.row):
            for c in range(self.column):
                M[r, c] = self[r, c] - N[r, c]
        return M
    
    # Multiplication of two matrices
    def __mul__(self, N):
        if isinstance(N, int) or isinstance(N,float):
            M = Matrix(self.row, self.column)
            for r in range(self.row):
                for c in range(self.column):
                    M[r, c] = self[r, c]*N
        else:
            assert N.row == self.column, "维度不同，不能相乘"
            M = Matrix(self.row, N.column)
            for r in range(self.row):
                for c in range(N.column):
                    sum = 0
                    for k in range(self.column):
                        sum += self[r, k] * N[k, r]
                    M[r, c] = sum
        return M
    
    #  finding the inverse matrix
    def invert(self):
        assert self.row == self.column, "不是方阵"
        M = Matrix(self.row, self.column*2)#Double column
        I = self.identity() #  unit matrix
        
        I.show() 
        
        # joint
        for r in range(1,M.row+1):
            temp = self[r]
            temp.extend(I[r])
            M[r] = copy.deepcopy(temp)
        M.show()
        
        # elementary row transformation primary row transformation
        for r in range(1, M.row+1):
            # If the first element of the line (M[r, r]) is 0, 
            #then the row with the nearest non-zero element is swapped downwards
            if M[r, r] == 0:
                for rr in range(r+1, M.row+1):
                    if M[rr, r] != 0:
                        M[r],M[rr] = M[rr],M[r] # Exchange of two lines
                    break

            assert M[r, r] != 0, '矩阵不可逆'
            
            # The first element of the line (M[r, r]) is reduced to 1
            temp = M[r,r]  # 缓存
            for c in range(r, M.column+1):
                M[r, c] /= temp
                print("M[{0}, {1}] /=  {2}".format(r,c,temp))
            M.show()
                
            # All the elements above and below this column become 0
            for rr in range(1, M.row+1):
                temp = M[rr, r] # 缓存
                for c in range(r, M.column+1):
                    if rr == r:
                        continue
                    M[rr, c] -= temp * M[r, c]
                    print("M[{0}, {1}] -= {2} * M[{3}, {1}]".format(rr, c, temp,r))
                M.show()    
            
        # Take the inverse matrix
        N = Matrix(self.row,self.column)
        for r in range(1,self.row+1):
            N[r] = M[r][self.row:]
        return N
    
    # transpose of a matrix
    def transpose(self):
        M = Matrix(self.column, self.row)
        for r in range(self.column):
            for c in range(self.row):
                M[r, c] = self[c, r]
        return M
    
    #Initialize the zero matrix
    def zeros(self):
        M = Matrix(self.column, self.row, fill=0.0)
        return M
    #Initialize the 1 matrix
    def ones(self):
        M = Matrix(self.column, self.row, fill=1.0)
        return M
    
    # Find the identity matrix
    def identity(self):
        assert self.row == self.column, "不是方阵，无单位矩阵"
        M = Matrix(self.column, self.row)
        for r in range(self.row):
            for c in range(self.column):
                M[r, c] = 1.0 if r == c else 0.0
        return M
    # print the matrix
    def show(self):
        for r in range(self.row):
            for c in range(self.column):
                print(self[r+1, c+1],end='  ')
            print()

import Matrix as ma
if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)
    b=ma.Matrix(3,4,fill=1)
    b.show()
    a=ma.Matrix(3,4,fill=2)
    a.show()
    (a+b).show()
    c=ma.Matrix(4,4,fill=3)
    (a*c).show()
    c[1]=[1,2,9,4]
    c.show()
    c.transpose().show()
    c.identity().show()
    c.invert()
