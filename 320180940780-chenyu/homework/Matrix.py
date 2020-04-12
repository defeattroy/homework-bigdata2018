class Matrix:
    mat=[[]]
    def __init__(self,rows,cols):
        """
        >>> A=Matrix(2,2)
        >>> A.rows
        2
        >>> A.cols
        2
        >>> A.mat
        [[0, 0], [0, 0]]
        """
        self.rows=rows
        self.cols=cols
        self.mat=[[0 for i in range(cols)] for i in range(rows)]
        #Initialize the zero matrix with a given size


    def setMatrix(self,value):#Bulid a matrix and assign values
        """
        >>> value=[[1, 2], [3, 4]]
        >>> A=Matrix(len(value),len(value[0]))
        >>> A.setMatrix(value)
        >>> A.mat
        [[1, 2], [3, 4]]
        """
        i=0
        for row in value:
            j=0
            for col in row:
                self.mat[i][j]=col
                j+=1
            i+=1

            

    def __add__(self,other):
        """
        >>> value1=[[1, 2, 3], [3, 4, 5]]
        >>> A=Matrix(len(value1),len(value1[0]))
        >>> A.setMatrix(value1)
        >>> value2=[[1, 1, 1], [2, 2, 2]]
        >>> B=Matrix(len(value2),len(value2[0]))
        >>> B.setMatrix(value2)
        >>> A.mat
        [[1, 2, 3], [3, 4, 5]]
        >>> B.mat
        [[1, 1, 1], [2, 2, 2]]
        >>> (A+B).mat
        [[2, 3, 4], [5, 6, 7]]
        >>> (A+1).mat
        [[2, 3, 4], [4, 5, 6]]
        >>> value3=[[1, 2], [1, 3]]
        >>> C=Matrix(len(value3),len(value3[0]))
        >>> C.setMatrix(value3)
        >>> (A+C).mat
        Traceback (most recent call last):
           ...
        Exception: Can not be calculated!
        """
        if(type(other)==int):
            #Matrix plus a constant
            M=Matrix(self.rows,self.cols)
            for row in range(self.rows):
                for col in range(self.cols):
                    M.mat[row][col]=self.mat[row][col]+other      
        else:
            #The rows and columns of matrix cannot be calculated if they are not equal
            if(self.rows!=other.rows)or(self.cols!=other.cols):
                raise Exception('Can not be calculated!')
             #The matrix plus the matrix
            M=Matrix(self.rows,self.cols)
            for row in range(self.rows):
                for col in range(self.cols):
                    M.mat[row][col]=self.mat[row][col]+other.mat[row][col]
        return M


    def __sub__(self,other):
        """
        >>> value1=[[1, 2, 3], [3, 4, 5]]
        >>> A=Matrix(len(value1),len(value1[0]))
        >>> A.setMatrix(value1)
        >>> value2=[[1, 1, 1], [2, 2, 2]]
        >>> B=Matrix(len(value2),len(value2[0]))
        >>> B.setMatrix(value2)
        >>> A.mat
        [[1, 2, 3], [3, 4, 5]]
        >>> B.mat
        [[1, 1, 1], [2, 2, 2]]
        >>> (A-B).mat
        [[0, 1, 2], [1, 2, 3]]
        >>> (A-1).mat
        [[0, 1, 2], [2, 3, 4]]
        >>> value3=[[1, 2], [1, 3]]
        >>> C=Matrix(len(value3),len(value3[0]))
        >>> C.setMatrix(value3)
        >>> (A-C).mat
        Traceback (most recent call last):
           ...
        Exception: Can not be calculated!
        """
        if(type(other)==int):
            #Matrix minus a constant
            M=Matrix(self.rows,self.cols)
            for row in range(self.rows):
                for col in range(self.cols):
                    M.mat[row][col]=self.mat[row][col]-other     
        else:
            #The rows and columns of matrix cannot be calculated if they are not equal
            if(self.rows!=other.rows)or(self.cols!=other.cols):
                raise Exception('Can not be calculated!')
            #Matrix minus matrix
            M=Matrix(self.rows,self.cols)
            for row in range(self.rows):
                for col in range(self.cols):
                    M.mat[row][col]=self.mat[row][col]-other.mat[row][col]
        return M


    def dot(self,other):
        """
        >>> value1=[[1, 2, 3], [3, 4, 5]]
        >>> A=Matrix(len(value1),len(value1[0]))
        >>> A.setMatrix(value1)
        >>> value2=[[1, 1, 1], [2, 2, 2]]
        >>> B=Matrix(len(value2),len(value2[0]))
        >>> B.setMatrix(value2)
        >>> (A.dot(B)).mat
        [[1, 2, 3], [6, 8, 10]]
        >>> (A.dot(3)).mat
        [[3, 6, 9], [9, 12, 15]]
        >>> value3=[[1, 2], [1, 3]]
        >>> C=Matrix(len(value3),len(value3[0]))
        >>> C.setMatrix(value3)
        >>> (A.dot(C)).mat
        Traceback (most recent call last):
           ...
        Exception: Can not be calculated!
        """
        if(type(other)==int):
            #Matrix dot a constant
            M=Matrix(self.rows,self.cols)
            for row in range(self.rows):
                for col in range(self.cols):
                    M.mat[row][col]=self.mat[row][col]*other     
        else:
            #The rows and columns of matrix cannot be calculated if they are not equal
            if(self.rows!=other.rows)or(self.cols!=other.cols):
                raise Exception('Can not be calculated!')
            #Matrix dot matrix
            M=Matrix(self.rows,self.cols)
            for row in range(self.rows):
                for col in range(self.cols):
                    M.mat[row][col]=self.mat[row][col]*other.mat[row][col]
        return M

    def __mul__(self,other):
        """
        >>> value1=[[1, 2, 3], [3, 4, 5]]
        >>> A=Matrix(len(value1),len(value1[0]))
        >>> A.setMatrix(value1)
        >>> value2=[[1, 1, 1], [2, 2, 2],[3, 3, 3]]
        >>> B=Matrix(len(value2),len(value2[0]))
        >>> B.setMatrix(value2)
        >>> (A*B).mat
        [[14, 14, 14], [26, 26, 26]]
        >>> value3=[[1, 2], [1, 3]]
        >>> C=Matrix(len(value3),len(value3[0]))
        >>> C.setMatrix(value3)
        >>> (A*C).mat
        Traceback (most recent call last):
           ...
        Exception: Can not be calculated!
        """
        #The cols of matrix-A and the rows of matrix-B cannot be calculated if they are not equal
        if(self.cols!=other.rows):
            raise Exception('Can not be calculated!')
        #Matrix multiplicate matrix
        M=Matrix(self.rows,other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    M.mat[i][j]+=self.mat[i][k]*other.mat[k][j]
        return M

    def transpose(self):
        """
        >>> value1=[[1, 2, 3], [3, 4, 5]]
        >>> A=Matrix(len(value1),len(value1[0]))
        >>> A.setMatrix(value1)
        >>> (A.transpose()).mat
        [[1, 3], [2, 4], [3, 5]]
        """
        #The transpose of the matrix
        M=Matrix(self.cols,self.rows)
        for row in range(self.rows):
            for col in range(self.cols):
                M.mat[col][row]=self.mat[row][col]
        return M

if __name__=='__main__':
    import doctest
    doctest.testmod()
