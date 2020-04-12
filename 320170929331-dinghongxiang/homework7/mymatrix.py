class Matrix(object):
    """
    the matrix class having methods below:
        add: R = A+B, R += A
        sub: R = A-B, R -= A
        mul: R = λ*A, R = A*λ
        div: R = A/B
        dot: R = A.dot(B), returns A·B
        element: R = A.element(B), returns A*B
        inverse: R = A.inverse()
        identity: A.identity()
        transpose: R = A.transpose()
    """

    def __init__(self, *args, **kwargs):
        """initialize the matrix, using a 2-dimension list(or no parameter)

        >>> Matrix([[1, 2, 3], [4, 5, 6]]).size()
        (2, 3)
        
        """
        if args == ():
            self.M = self.N = 0
            self.mtx = []
        else:
            self.M = len(args[0])
            if type(args[0][0]) != int:
                self.N = len(args[0][0])
            else:
                self.N = self.M
                self.M = 1
            self.mtx = args[0]
            
    def __str__(self):
        """turn the matrix into string
        
        >>> Matrix([1, 2, 3])
        1 2 3 
        
        """
        st = ''
        if not self.mtx == []:
            for line in self.mtx:
                if self.M == 1:
                    st += str(line) + ' '
                else:
                    for item in line:
                        st += str(item) + ' '
                    st += '\n'
        return st
    
    __repr__ = __str__

    def size(self):
        """return the size of the matrix
        
        >>> Matrix([[1], [2], [3]]).size()
        (3, 1)
        
        """
        return (self.M, self.N)

    def __eq__(self, another):
        """compare this matrix with another, return the result
        
        >>> Matrix([1, 2, 3]) == Matrix([1, 2, 3]) * 1
        True
        >>> Matrix([[1, 2], [3, 4]]) == Matrix([1, 2, 3, 4])
        False

        """
        if self.M != another.M or self.N != another.N:
            return False
        else:
            if self.mtx == another.mtx:
                return True
            else:
                return False

    def __add__(self, another):
        """add the matrix with another one, return the result
        
        >>> str(Matrix([1, 2, 3]) + Matrix([1, 2, 3]))
        '2 4 6 '
        >>> Matrix([[1, 2], [3, 4]]) + Matrix([[2, 4], [6, 8]]) == Matrix([[3, 6], [9, 12]])
        True
        >>> Matrix([1, 2]) + Matrix([1])
        False

        """
        if self.M == another.M and self.N == another.N:
            t_mtx = self.mtx
            if self.M == 1:
                for i in range(self.N):
                    t_mtx[i] += another.mtx[i]
            else:
                for iline in range(self.M):
                    for i in range(self.N):
                        t_mtx[iline][i] += another.mtx[iline][i]
            return Matrix(t_mtx)
        else:
            return False
    
    def __iadd__(self, another):
        """add the matrix with another one, change the matrix itself
        
        >>> mtx1 = Matrix([[1, 1, 1], [2, 2, 2]])
        >>> mtx1 += Matrix([[1, 2, 3], [4, 5, 6]])
        >>> str(mtx1)
        '2 3 4 \\n6 7 8 \\n'

        """
        if self.M == another.M and self.N == another.N:
            if self.M == 1:
                for i in range(self.N):
                   self.mtx[i] += another.mtx[i]
            else:
                for iline in range(self.M):
                    for i in range(self.N):
                        self.mtx[iline][i] += another.mtx[iline][i]
            return self
        else:
            return False
    
    def __sub__(self, another):
        """subtract the matrix with another one, return the result
        
        >>> str(Matrix([1, 2, 3]) - Matrix([3, 2 ,1]))
        '-2 0 2 '

        """
        if self.M == another.M and self.N == another.N:
            t_mtx = self.mtx
            if self.M == 1:
                for i in range(self.N):
                    t_mtx[i] -= another.mtx[i]
            else:
                for iline in range(self.M):
                    for i in range(self.N):
                        t_mtx[iline][i] -= another.mtx[iline][i]
            return Matrix(t_mtx)
        else:
            return False
    
    def __isub__(self, another):
        """subtract the matrix with another one, change the matrix itself
        
        >>> mtx1 = Matrix([[1, 1, 1], [2, 2, 2]])
        >>> mtx1 -= Matrix([[1, 2, 3], [4, 5, 6]])
        >>> str(mtx1)
        '0 -1 -2 \\n-2 -3 -4 \\n'
        
        """
        if self.M == another.M and self.N == another.N:
            if self.M == 1:
                for i in range(self.N):
                   self.mtx[i] -= another.mtx[i]
            else:
                for iline in range(self.M):
                    for i in range(self.N):
                        self.mtx[iline][i] -= another.mtx[iline][i]
            return self
        else:
            return False
    
    def __mul__(self, val):
        """multiply(scalar) with a constant, return the result
        
        >>> str(Matrix([1, 2, 3]) * 2)
        '2 4 6 '
        >>> str(Matrix([[1, 2 ,3] ,[4, 5, 6]]) * 0.5)
        '0.5 1.0 1.5 \\n2.0 2.5 3.0 \\n'

        """
        if self.M == 1:
            t_mtx = [0 for i in range(self.N)]
            for i in range(self.N):
                t_mtx[i] = self.mtx[i] * val
        else:
            t_mtx = [[0 for i in range(self.N)] for j in range(self.M)]
            for iline in range(self.M):
                for i in range(self.N):
                    t_mtx[iline][i] = val * self.mtx[iline][i]
        return Matrix(t_mtx)

    def __rmul__(self, val):
        """multiply(scalar) with a constant, return the result
        
        str(2 * Matrix([1, 2, 3]))
        '2 4 6 '
        >>> str(0.5 * Matrix([[1, 2 ,3] ,[4, 5, 6]]))
        '0.5 1.0 1.5 \\n2.0 2.5 3.0 \\n'
        
        """
        if self.M == 1:
            t_mtx = [0 for i in range(self.N)]
            for i in range(self.N):
                t_mtx[i] = self.mtx[i] * val
        else:
            t_mtx = [[0 for i in range(self.N)] for j in range(self.M)]
            for iline in range(self.M):
                for i in range(self.N):
                    t_mtx[iline][i] = val * self.mtx[iline][i]
        return Matrix(t_mtx)

    def __truediv__(self, another):
        """divide this matrix with another, return the result
        
        >>> str(Matrix([[1, 2], [3, 4]]) / Matrix([[1, 2,], [3, 4]]))
        '1.0 0.0 \\n0.0 1.0 \\n'
        
        """
        return self.dot(another.inverse())
    
    def transpose(self):
        """
        transpose the Matrix and return the result

        >>> str(Matrix([1, 2, 3]).transpose())
        '1 \\n2 \\n3 \\n'
        >>> str(Matrix([[1], [2], [3]]).transpose())
        '1 2 3 '

        """
        if self.M == 1:
            if self.N == 1:
                return self
            else:
                t_mtx = [[0] for i in range(self.N)]
                for i in range(self.N):
                    t_mtx[i][0] = self.mtx[i]
        elif self.N == 1:
            t_mtx = []
            for i in range(self.M):
                t_mtx.append(self.mtx[i][0])
        else:
            t_mtx = [[0 for i in range(self.M)] for j in range(self.N)]
            for iline in range(self.N):
                for i in range(self.M):
                    t_mtx[iline][i] = self.mtx[i][iline]
        return Matrix(t_mtx)
    
    def identity(self, n:int):
        """make the Matrix a identity matrix with n order (Attention: not generate but replace the matrix itself)
        
        >>> str(Matrix().identity(3))
        '1 0 0 \\n0 1 0 \\n0 0 1 \\n'
        
        """
        if n > 1:
            self.M = self.N = n
            t_mtx = []
            for iline in range(n):
                t_mtx.append([])
                for i in range(n):
                    t_mtx[iline].append(0)
                t_mtx[iline][iline] = 1
            self.mtx = t_mtx
            return Matrix(t_mtx)
        elif n == 1:
            self.M = self.N = 1
            self.mtx = [1]
    
    def dot(self, another):
        """dot product with another matrix, return the result

        >>> str(Matrix([[1, 0], [2, 1]]).dot(Matrix([[1, 0], [-2, 1]])))
        '1 0 \\n0 1 \\n'

        """
        if self.N == another.M:
            if self.M == 1:
                rs_mtx = []
                if another.M == 1:
                    rs_mtx.append(self.mtx[0] * another.mtx[0])
                else:
                    
                    for i in range(another.N):
                        k = 0
                        for l in range(self.N):
                            k += self.mtx[l] * another[l][i]
                        rs_mtx.append(k)
            else:
                rs_mtx = [[0 for i in range(self.N)] for j in range(self.M)]
                for iline in range(self.M):
                    for i in range(another.N):
                        for p in range(self.N):
                            rs_mtx[iline][i] += (self.mtx[iline][p] * another.mtx[p][i])
            return Matrix(rs_mtx)
        else:
            return False
    
    def element(self, another):
        """element-wise product with another matrix, return the result

        >>> str(Matrix([[1, 2], [1, 0]]).element(Matrix([[1, 0.5], [-1, 2]])))
        '1 1.0 \\n-1 0 \\n'

        """
        if self.M == another.M and self.N == another.N:
            rs_mtx = self.mtx
            if self.M == 1:
                for i in range(self.N):
                    rs_mtx[i] *= another.mtx[i]
            else:
                for iline in range(self.M):
                    for i in range(self.N):
                        rs_mtx[iline][i] *= another.mtx[iline][i]
            return Matrix(rs_mtx)
        else:
            return False


    def swap_rows(self, r1, r2):
        tmp = self.mtx[r1]
        self.mtx[r1] = self.mtx[r2]
        self.mtx[r2] = tmp
    
    def scale_row(self, r, scalar):
        for i in range(self.N):
            self.mtx[r][i] *= scalar

    def shear_row(self, r1, r2, scalar):
        for i in range(self.N):
            self.mtx[r1][i] += scalar * self.mtx[r2][i]

    def inverse(self):
        """inverse the matrix, return the result
        
        >>> str(Matrix([[1, 0], [2, 1]]).inverse())
        '1.0 0.0 \\n-2.0 1.0 \\n'
        
        """
        n = self.M
        backup = Matrix(self.mtx)
        output = Matrix().identity(n)

        for i in range(n):
            if backup.mtx[i][i] == 0.0:
                for r in range(i+1, n+1):
                    if r == n:
                        return False
                    if backup.mtx[r][i] != 0.0:
                        backup.swap_rows(i, r)
                        output.swap_rows(i, r)
                        break
            scalar = 1.0 / backup.mtx[i][i]
            backup.scale_row(i, scalar)
            output.scale_row(i, scalar)

            for j in range(n):
                if i == j:
                    continue
                shear_needed = -backup.mtx[j][i]
                backup.shear_row(j, i, shear_needed)
                output.shear_row(j, i, shear_needed)
        return output        


if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)
