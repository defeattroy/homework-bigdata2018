#I used list realize class matrix
class Matrix:
    def __init__(self,lis):
        self.__value = lis
        self.shape = (len(lis),len(lis[0]))
        self.rows = self.shape[0]
        self.columns = self.shape[1]

    @property
    def value(self):
        return self.__value
    @value.setter
    def value(self,lis):
        assert isinstance(lis,list),"must input a list."
        self.__value = lis

    def add(self,other):
        """"
        Addition of matrices
        >>> f = Matrix([[1,2,3],[4,5,6]])
        >>> g = Matrix([[7,8,9],[4,1,2]])
        >>> Matrix.add(f,g)
        [[8, 10, 12], [8, 6, 8]]
        """
        assert isinstance(other,Matrix),"the other must be a matrix."
        final_mat = self.zeros(self.shape)
        try:
            for i in range(self.rows):
                for j in range(self.columns):
                    final_mat[i][j] = self.value[i][j] + other.value[i][j]
        except Exception:
            print("Two matrices are not matched!")
        return final_mat

    def sub(self,other):
        """
        Matrix subtraction
        >>> f = Matrix([[1,2,3],[4,5,6]])
        >>> g = Matrix([[7,8,9],[4,1,2]])
        >>> Matrix.sub(f,g)
        [[-6, -6, -6], [0, 4, 4]]
        """
        for i in range(other.rows):
            for j in range(other.columns):
                other.value[i][j] = -other.value[i][j]
        return self.add(other)

    def sca(self,k):
        """
        get matrix scalar-multiplication
        >>> f = Matrix([[1,2,3],[4,5,6]])
        >>> f.sca(3)
        [[3, 6, 9], [12, 15, 18]]
        """
        sca_mul = [[r[c]*k for c in range(self.columns)] for r in self.value] #try a pythonic way
        return  sca_mul
    def tran(self):
        """
        transpose of a matrix
        >>> f = Matrix([[1,2,3],[4,5,6]])
        >>> f.tran()
        [[1, 4], [2, 5], [3, 6]]
        """
        trans_mat = [[r[c] for r in self.value] for c in range(self.columns)] #try a pythonic way too
        return trans_mat

    def identity(self,shape):
        """
        get identity matrix

        >>> f = Matrix([[1,2,3],[4,5,6]])
        >>> f.identity((3,3))
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        """
        assert isinstance(shape,tuple),"shape must be a tuple"
        identity_matrix = []
        for i in range(shape[0]):
            identity_matrix.append([])
            for j in range(shape[1]):
                identity_matrix[i].append(1) if i == j else identity_matrix[i].append(0)
        return identity_matrix

    def zeros(self,shape):
        """
        get null matrix
        >>> f = Matrix([[1,2,3],[4,5,6]])
        >>> f.zeros((3,3))
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        """
        assert isinstance(shape,tuple),"shape must be a tuple"
        zero_matrix = []
        for i in range(shape[0]):
            zero_matrix.append([])
            for j in range(shape[1]):
                zero_matrix[i].append(0)
        return zero_matrix

    def dot(self,other):
        """
        >>> f = Matrix([[1,2],[4,5]])
        >>> g = Matrix([[3,6],[3,7]])
        >>> f.dot(g)
        [[9, 16], [15, 22]]
        """
        assert isinstance(other, Matrix), "the other must be a matrix."
        shape = (self.rows,other.columns)
        final_mat = self.zeros(shape)
        try:
            for i in range(self.rows):
                for j in range(other.columns):
                    num = 0
                    for k in range(self.columns):
                        num += self.value[i][k] + other.value[k][j]
                    final_mat[i][j] = num
            return final_mat
        except Exception:
            print("Two matrices are not matched!")

    def multi(self,other):
        """
        get Element-wise product
        >>> f = Matrix([[1,2,3],[4,5,6]])
        >>> g = Matrix([[7,8,9],[4,1,2]])
        >>> f.multi(g)
        [[7, 16, 27], [16, 5, 12]]
        """
        assert isinstance(other, Matrix), "the other must be a matrix."
        final_mat = self.zeros(self.shape)
        try:
            for i in range(self.rows):
                for j in range(self.columns):
                    final_mat[i][j] = self.value[i][j] * other.value[i][j]
        except Exception:
            print("Two matrices are not matched!")
        return final_mat

    def elimination(self,choice,k):
        """delete a row or column of the matrix
        >>> f = Matrix([[1,2,3],[4,5,6]])
        >>> print(f.elimination('c',1))
        [[1, 3], [4, 6]]
        """
        if choice == 'c':
            for item in self.value:
                item.pop(k)
        elif choice == 'r':
            self.value.pop(k)
        else:
            print("please input correct choice")
        return self.value
if __name__ == '__main__':
    import doctest
    doctest.testmod()

