"""The matrix class supports the following methods:
  1, Matrix.T #矩阵的转置
  2，Matrix.is_square #判断矩阵是不是一个方阵
  3，Matrix.determinant #简单计算1x1 2x2 3x3 方阵的值
  4，Matrix.__add__ #计算同型矩阵的加法
  5，Matrix.__sub__#计算同型矩阵的加法
  6, Matrix.__mul__#计算两个矩阵的相乘，生成一个新的矩阵
  7, Matix.__repr__#打印矩阵的标准写法
  注:(编译环境，windows10,IDLE，pyhon3.7)
    
"""
class Matrix():
    def __init__(self,matrix):
        self.m=matrix
        self.r=len(matrix)
        self.c=len(matrix[0])

    def T(self):

        """Transpose matrix,exchange its rows and columns of a matrix

        >>> matrix = Matrix([[1,2,3], [4,5,6], [7,8,9]])
        >>> matrix.T()
        [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        """
        matrix_transpose = []
        for j in range(self.c):
            new_row = []
            for i in range(self.r):
                new_row.append(self.m[i][j])
            matrix_transpose.append(new_row)          
        return matrix_transpose

    
    def is_square(self):
        """Judge whether a matrix is a square matrix

        >>> matrix = Matrix([[1,2,3], [4,5,6], [7,8,9]])
        >>> matrix.is_square()
        True
        >>> matrix = Matrix([[1,2], [4,5], [7,8]])
        >>> matrix.is_square()
        False
        """
        return self.r == self.c

    def determinant(self):
        """This method is used to calculate the first and second order determinants

        >>> matrix = Matrix([[1,2], [4,5]])
        >>> matrix.determinant()
        -3
    
        >>> matrix = Matrix([[1]])
        >>> matrix.determinant()
        1
        >>> matrix = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        >>> matrix.determinant()
        0
        """
        if self.c!=self.r:
            raise(ValueError, "the matrix isn't a determinant,can't use this method")
        else:
            if self.c>3:
                raise(ValueError,"the matrix is lager than 2x2,can't calculate")
            if self.c == 2:
                return (self.m[0][0]*self.m[1][1]-self.m[0][1]*self.m[1][0])
            if self.c == 3:
                return (self.m[0][0]*(self.m[1][1]*self.m[2][2]-self.m[1][2]*self.m[2][1])+
                        self.m[0][1]*(self.m[2][1]*self.m[0][2]-self.m[0][1]*self.m[2][2])+
                        self.m[0][2]*(self.m[0][1]*self.m[1][2]-self.m[0][2]*self.m[1][1]))
            if self.c ==1:
                return self.m[0][0]

    def __add__(self,other):
        """this method is used to add up two matrixs of the same dimension
        >>> a = Matrix([[1,2,3],[4,5,6]])
        >>> b = Matrix([[9,10,11],[12,13,14]])
        >>> print(a+b)
        [[10, 12, 14], [16, 18, 20]]
        """
        if self.c == other.c and self.r == other.r:
            matrix=[]
            for i in range(self.r):
                new_row=[]
                for j in range(self.c):
                    add = self.m[i][j] + other.m[i][j]
                    new_row.append(add)
                matrix.append(new_row)
            return matrix
        else:
            raise(ValueError, "Matrices can only be added if the dimensions are the same")


    def __sub__(self,other):
        """For subtraction of two matrices which have the same dimension
        >>> a = Matrix([[1,2,3],[4,5,6]])
        >>> b = Matrix([[9,10,11],[12,13,14]])
        >>> print(b-a)
        [[8, 8, 8], [8, 8, 8]]
        """
        if self.c == other.c and self.r == other.r:
            matrix=[]
            for i in range(self.r):
                new_row=[]
                for j in range(self.c):
                    add = self.m[i][j] - other.m[i][j]
                    new_row.append(add)
                matrix.append(new_row)
            return matrix
        else:
            raise(ValueError, "Matrices can only be added if the dimensions are the same")


    def __mul__(self,other):
        """In this method, two matricesthat can be multiplied are first
         decomposed into vectors, and the row vectors of the first matrix are
         multiplied by the column vectors of the second matrix to obtain
         the matrix elements after multiplication
        >>> a=Matrix([[1,2],[3,4]])
        >>> b=Matrix([[1,2],[3,4]])
        >>> print(a*b)
        [[7, 10], [15, 22]]
        """
        if self.c == other.r:
            def dot_product(vector1, vector2):#Calculate the point multiplication of two vectors
                dot_product = 0
                for i in range(len(vector1)):
                    dot_product += vector1[i] * vector2[i]
                return dot_product


            def get_row(matrix, row):#Take the row out as vectors
               return matrix[row]

            def get_column(matrix, col):
                column = []
                for i in range(len(matrix)): 
                    column.append(matrix[i][col])
                return column

            result = []
            for i in range(self.r):
                row_result = []
                for j in range(self.c):
                    vector1 = get_row(self.m, i)
                    vector2 = get_column(other.m, j)
                    new_element = dot_product(vector1, vector2)
                    row_result.append(new_element)
                result.append(row_result)

            return result
        else:
            raise(ValueError, "It must be satisfied that the columns of the first matrix and the rows of the second matrix are equal")



    def __repr__(self):
        """This method is to print the matrix neatly
        >>> a = Matrix([[1,2],[3,4]])
        >>> print(a)
        1  2 
        3  4 
        """
        s = ""
        i=0
        for row in self.m:
            s += " ".join(["{} ".format(x) for x in row])
            if i!=self.r-1:
               s += "\n"
            i=i+1
        return s
        
            

if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)
    
