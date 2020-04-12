from functools import lru_cache
from itertools import product
import random

class Matrix:
    def __init__(self,
                 height: int,
                 width: int,
                 fill_value=[]) -> None:
        """
        Matrix initialization,
        the length and width of the matrix, and the value.
        If no value is passed, the default zero matrix.
        If there are too many elements in the list passed in, stop taking as many as needed.
        If not enough, set it to zero.
        >>> a = [1,2,3,4,5,6,7,8,9]
        >>> b = [1,2,3,4,5,6,7,8,9,10]
        >>> c = [1,2,3,4,5,6,7]
        >>> my_matrix = Matrix(3,3,a)
        >>> print(my_matrix)
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        >>> print(Matrix(3,3,b))
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        >>> print(Matrix(3,3,c))
        [1, 2, 3]
        [4, 5, 6]
        [7, 0, 0]
        >>> print(Matrix(3,3))
        [0, 0, 0]
        [0, 0, 0]
        [0, 0, 0]
        """

        self.height = height
        self.width = width
        self.rows = [[0] * width for _ in range(height)] #Build the framework first

        indices = product(range(self.height), range(self.width))

        values = iter(fill_value)

        for (i, j), value in zip(indices, values):
            self.rows[i][j] = value

    def __len__(self) -> int:
        """
        return the total number of elements
        >>> my_matrix = Matrix(3,3)
        >>> print(len(my_matrix))
        9
        """

        return self.height * self.width

    def __str__(self) -> str:
        """
        >>> print(Matrix(3,4))
        [0, 0, 0, 0]
        [0, 0, 0, 0]
        [0, 0, 0, 0]
        """

        return "\n".join(map(str, self.rows))

    def __repr__(self) -> str:
        """
        Returns matrix name and size
        >>> Matrix(3,4)
        Matrix(3, 4)
        """

        return f'{self.__class__.__name__}({self.height}, {self.width})'

    def __eq__(self, matrix2):
        """
        Judge equal
        >>> m1 = Matrix(3,4)
        >>> m2 = Matrix(3,4)
        >>> m3 = Matrix(3,3)
        >>> print((m1 == m2 , m2 == m3))
        (True, False)
        """

        return (self.rows == matrix2.rows)

    def __add__(self, matrix2):
        """
        Matrix addition
        Doesn't modifycthe current matrix
        >>> a = [1,2,3,4,5,6,7,8,9]
        >>> b = [1,2,3,4,5,6,7,8,9]
        >>> m1 = Matrix(3,3,a)
        >>> m2 = Matrix(3,3,b)
        >>> print(m1 + m2)
        [2, 4, 6]
        [8, 10, 12]
        [14, 16, 18]
        """

        if (self.height == matrix2.height and self.width == matrix2.width):
            return_matrix = Matrix(self.height, self.width)
        else:
            print("Both matrix formats should be the same!")
            return

        indices = product(range(self.height), range(self.width))
        for (i, j) in indices:
            return_matrix.rows[i][j] = self.rows[i][j] + matrix2.rows[i][j]
        return return_matrix

    def __sub__(self, matrix2):
        """
        Matrix subtraction
        Doesn't modify the current matrix
        >>> a = [9,8,7,6,5,4,3,2,1]
        >>> b = [1,2,3,4,5,6,7,8,9]
        >>> m1 = Matrix(3,3,a)
        >>> m2 = Matrix(3,3,b)
        >>> print(m1 - m2)
        [8, 6, 4]
        [2, 0, -2]
        [-4, -6, -8]
        """
        if (self.height == matrix2.height and self.width == matrix2.width):
            return_matrix = Matrix(self.height, self.width)
        else:
            print("Both matrix formats should be the same!")
            return

        indices = product(range(self.height), range(self.width))
        for (i, j) in indices:
            return_matrix.rows[i][j] = self.rows[i][j] - matrix2.rows[i][j]
        return return_matrix

    def __mul__(self, matrix2):
        """
        Matrix multiplication
        Doesn't modify the current matrix
        >>> a = [9,8,7,6,5,4,3,2,1]
        >>> b = [1,2,3,4,5,6,7,8,9]
        >>> m1 = Matrix(3,3,a)
        >>> m2 = Matrix(3,3,b)
        >>> print(m1*m2)
        [90, 114, 138]
        [54, 69, 84]
        [18, 24, 30]
        """
        if (self.height == matrix2.width):
            return_matrix = Matrix(self.height, self.width)
        else:
            print("The length of matrix one should be equal to the width of matrix two!")
            return

        for i in range(self.height):
            for j in range(matrix2.width):
                for k in range(matrix2.height):
                    return_matrix.rows[i][j] += self.rows[i][k] * matrix2.rows[k][j]
        return return_matrix

    @classmethod
    def makeRandom(cls,
                   height: int,
                   width: int,
                   low=1,
                   high=100):
        """
        Make a random matrix with elements in range (low-high)
        default low = 1  , high = 100
        #print(Matrix.makeRandom(3,5)
        [26, 51, 76, 39, 55]
        [84, 46, 32, 97, 77]
        [79, 9, 76, 44, 53]
        #print(Matrix.makeRandom(3,5,100,200))
        [133, 167, 102, 145, 112]
        [102, 181, 146, 121, 190]
        [115, 140, 168, 180, 138]
        """

        fill_value = []
        for _ in range(height * width):
            fill_value.append(random.randrange(low, high))
        return_matrix = Matrix(height, width, fill_value)
        return return_matrix

    @classmethod
    def Identity(cls, size: int):
        """
        Enter the side length of the identity matrix,
        and then build the identity matrix
        >>> print(Matrix.Identity(6))
        [1, 0, 0, 0, 0, 0]
        [0, 1, 0, 0, 0, 0]
        [0, 0, 1, 0, 0, 0]
        [0, 0, 0, 1, 0, 0]
        [0, 0, 0, 0, 1, 0]
        [0, 0, 0, 0, 0, 1]
        """

        return_matrix = Matrix(size, size)
        for _ in range(size):
            return_matrix.rows[_][_] = 1
        return return_matrix

    def replace(self,
                row_index: int,
                column_index: int,
                value) -> None:
        """
        Replaces a value to  a new one by given indices.
        Modify the current matrix
        >>> m1 = Matrix(4,4)
        >>> m1.replace(2,2,8)
        >>> print(m1)
        [0, 0, 0, 0]
        [0, 8, 0, 0]
        [0, 0, 0, 0]
        [0, 0, 0, 0]
        """

        self.rows[row_index-1][column_index-1] = value

    def Transpose(self):
        """
        Transpose matrix
        Doesn't modify the current matrix
        >>> m1 = Matrix(3,3,[1,2,3,4,5,6,7,8,9])
        >>> print(m1.Transpose())
        [1, 4, 7]
        [2, 5, 8]
        [3, 6, 9]
        """
        return_matrix = Matrix(self.height, self.width)

        indices = product(range(self.height), range(self.width))

        for (i, j) in indices:
            return_matrix.rows[j][i] = self.rows[i][j]

        return return_matrix

    def element_wise(self, matrix2):
        """
        Multiply elements one by one
        Doesn't modify the current matrix
        >>> m1 = Matrix(3,3,[1,2,3,4,5,6,7,8,9])
        >>> m2 = Matrix(3,3,[1,2,3,4,5,6,7,8,9])
        >>> print(m1.element_wise(m2))
        [1, 4, 9]
        [16, 25, 36]
        [49, 64, 81]
        """

        if (self.height == matrix2.height and self.width == matrix2.width):
            return_matrix = Matrix(self.height, self.width)
        else:
            print("Error!")
            return
        indices = product(range(self.height), range(self.width))

        for (i, j) in indices:
            return_matrix.rows[i][j] += self.rows[i][j] * matrix2.rows[i][j]
        return return_matrix

    def Scalar(self, num):
        """
        Multiply matrix by input number
        Doesn't modify the current matrix
        >>> m1 = Matrix(3,3,[1,2,3,4,5,6,7,8,9])
        >>> print(m1.Scalar(10))
        [10, 20, 30]
        [40, 50, 60]
        [70, 80, 90]
        """

        return_matrix = Matrix(self.height, self.width)
        indices = product(range(self.height), range(self.width))
        for (i, j) in indices:
            return_matrix.rows[i][j] += self.rows[i][j] * num
        return return_matrix

    def Permutation(self,
                    operation: str,
                    num1: int,
                    num2: int) -> None:
        """
        Swap rows or columns based on input selection
        Modify the current matrix
        >>> m1 = Matrix(3,3,[1,2,3,4,5,6,7,8,9])
        >>> m1.Permutation('col',1,3)
        >>> print(m1)
        [3, 2, 1]
        [6, 5, 4]
        [9, 8, 7]
        >>> m1.Permutation('row',1,3)
        >>> print(m1)
        [9, 8, 7]
        [6, 5, 4]
        [3, 2, 1]
        """
        if (operation == 'row'):
            self.rows[num1 - 1], self.rows[num2 - 1] = self.rows[num2 - 1], self.rows[num1 - 1]
        elif (operation == 'col'):
            for _ in range(self.height):
                self.rows[_][num1 - 1], self.rows[_][num2 - 1] = self.rows[_][num2 - 1], self.rows[_][num1 - 1]

    def Elimination(self):
        """
        The scientific calculation package sympy can use the method of A.rref () to directly find the matrix A's RREF
        A bit complicated, I don't know much
        """
        raise NotImplementedError

    def Inverse(self):
        """
        Use the numpy package function np.linalg.inv (A) to solve the inverse of matrix A
        A bit complicated, I don't know much
        """
        raise NotImplementedError


if __name__ == "__main__":
    import doctest

    doctest.testmod()