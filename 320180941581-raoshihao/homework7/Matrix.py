class Matrix:
    def __init__(self, element: list):
        row_number = [len(item) for item in element]
        #empty matrix
        if set(row_number) == {0}:
            self.element = []
            self.row = 0
            self.col = 0
        #nomal matrix
        elif len(set(row_number)) == 1:
            self.element = element
            self.row = len(element)
            self.col = len(element[0])
        #not a matrix
        else :
            raise ValueError("That's not a matrix!")
    '''
    Let's test various situations of create a matrix.
    
    >>> a = Matrix([[1,2,3],[4,5,6,],[7,8,9,]])
    >>> a
    [1, 2, 3]
    [4, 5, 6]
    [7, 8, 9]

    >>> l = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    >>> Matrix(l)
    [0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9, 10, 11]
    
    >>> l = [[]]
    >>> m = [[0]]
    >>> a = Matrix(l)
    >>> b = Matrix(m)
    >>> l,m
    ([[]], [[0]])
    >>> a.col,b.row
    (0, 1)
    '''
    
    def __repr__(self):
        m = ""
        for item in self.element:
            m += str(item)
            m += '\n'
        return m

    def __iter__(self):
        return iter(self.element)

    def __add__(self, other):
        if (self.row != other.row) or (self.col != other.col):
            raise TypeError("Column and row should be same！")
        else:
            C = []
            for A, B in zip(self.element, other.element):
                c = [ a + b for a,b in zip(A, B)]
                C.append(c)
            return Matrix(C)
        
    def __sub__(self, other):
        if (self.row != other.row) or (self.col != other.col):
            raise TypeError("column and row should be same！")
        else:
            C = []
            for A, B in zip(self.element, other.element):
                c = [ a - b for a,b in zip(A, B)]
                C.append(c)
            return Matrix(C)

    def __mul__(self, other):
        if self.col != other.row:
            raise TypeError("The column of the first matrix should be same as the second's row!")
        else:
            # create matrix
            C = [[0] * other.col for i in range(self.row)]
            for i in range(self.row):
                for j in range(other.col):
                    for k in range(self.col):
                        C[i][j] += self.element[i][k] * other.element[k][j]
            return Matrix(C)


    #get a negative matrix
    def __neg__(self):
        new = [[-i for i in item] for item in self.element]
        return Matrix(new)

     #Matrix exponentiation
    def __pow__(self, power, modulo=None):
        if not self.is_square():
            raise TypeError("只有方阵才能求幂！")
        else:
            new = identity(self.row)
            A = Matrix(self.element)
            for i in range(power):
                new = new * A
            return new

    #Matrix number division and multiplication
    def div(self, k: int):
        for i in range(self.row):
            for j in range(self.col):
                self.element[i][j] /= k
    
    def mul(self, k: int):
        for i in range(self.row):
            for j in range(self.col):
                self.element[i][j] *= k

    '''
    Matrix calculation test:
    >>> a = Matrix([[1,2,3],[4,5,6,],[7,8,9,]])
    >>> c = Matrix([[3,2,1],[4,5,6,],[9,8,7]])
    >>> b = Matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
    >>> a + c
    [4, 4, 4]
    [8, 10, 12]
    [16, 16, 16]
    
    >>> a - c
    [-2, 0, 2]
    [0, 0, 0]
    [-2, 0, 2]

    >>> a*b
    [32, 38, 44, 50]
    [68, 83, 98, 113]
    [104, 128, 152, 176]

    >>> a**3
    [468, 576, 684]
    [1062, 1305, 1548]
    [1656, 2034, 2412]

    >>> -a
    [-1, -2, -3]
    [-4, -5, -6]
    [-7, -8, -9]

    >>> a.div(2)
    >>> a
    [0.5, 1.0, 1.5]
    [2.0, 2.5, 3.0]
    [3.5, 4.0, 4.5]
    
    >>> a.mul(2)
    >>> a
    [1.0, 2.0, 3.0]
    [4.0, 5.0, 6.0]
    [7.0, 8.0, 9.0]

    '''
                
    def shape(self):
        #Get the shape of the matrix
        print((self.row, self.col))

    def is_square(self):
        #Determine if it's a square matrix
        return self.row == self.col
    
    '''
    >>> l = [[]]
    >>> m = [[1]]
    >>> n = [[1,2],[3,4],[5,6]]
    >>> a = Matrix(l)
    >>> b = Matrix(m)
    >>> c = Matrix(n)
    >>> a.shape()
    (0, 0)
    >>> b.shape()
    (1, 1)
    >>> c.shape()
    (3, 2)
    >>> a.is_square()
    True
    >>> b.is_square()
    True
    >>> c.is_square()
    False
    >>> d = Matrix([[1,0],[0,1]])
    >>> d.is_square()
    True
    '''
    
    def getitem(self, i: int, j: int):
        i -= 1
        j -= 1
        if (0 <= i <= self.row) and (0 <= j <= self.col):
            return self.element[i][j]
        else:
            raise IndexError("Index error！")
        
    def setitem(self, i: int, j: int, value) -> None:
        i -= 1
        j -= 1
        if (0 <= i <= self.row) and (0 <= j <= self.col):
            self.element[i][j] = value
        else:
            raise IndexError("Index error！")
    '''
    >>> A = Matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    >>> A.setitem(1,1,0)
    >>> A
    [0, 2, 3, 4]
    [5, 6, 7, 8]
    [9, 10, 11, 12]
    [13, 14, 15, 16]

    >>> A.getitem(4,4)
    16
    '''
    # Matrix Translation
    def T(self):
        AT = [[0] * self.row for i in range(self.col)]
        for i in range(self.row):
            for j in range(self.col):
                AT[j][i] = self.element[i][j]
        return Matrix(AT)
    '''
    >>> A = Matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    >>> A.T()
    [1, 5, 9, 13]
    [2, 6, 10, 14]
    [3, 7, 11, 15]
    [4, 8, 12, 16]


    '''
    # Matrix determinant
    def det(self):
        assert self.is_square()
        return determinant(self.element)

    '''
    >>> A=Matrix([[1,2],[3,4]])
    >>> A.element
    [[1, 2], [3, 4]]
    >>> A.det()
    -2
    >>> B = Matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    >>> B.det()
    0

    '''
    
def determinant(m):
    if len(m) <= 0:
        return None
    if len(m) == 1:
        return m[0][0]
    else:
        s = 0
        for i in range(len(m)):
            n = [[row[a] for a in range(len(m)) if a != i] for row in m[1:]]
            if i % 2 == 0:
                s += m[0][i] * determinant(n)
            else:
                s -= m[0][i] * determinant(n)
        return s

#zeros matrix and identity matrix
def zeros(row: int, col: int) -> Matrix:
    new = [[0] * col for i in range(row)]
    return Matrix(new)

def identity(n: int) -> Matrix:
    new = [[0] * n for i in range(n)]
    for i in range(n):
        new[i][i] = 1
    return Matrix(new)

'''
>>> zeros(2,3)
[0, 0, 0]
[0, 0, 0]

>>> identity(5)
[1, 0, 0, 0, 0]
[0, 1, 0, 0, 0]
[0, 0, 1, 0, 0]
[0, 0, 0, 1, 0]
[0, 0, 0, 0, 1]

>>> 
'''
if __name__ == '__main__':
    import doctest
    doctest.testmod()
