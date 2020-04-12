from typing import List
from typing import Tuple
from typing import Callable
import numpy as np

t_vector = List[float]  # define the data type of Vector
t_matrix = List[t_vector]  # define the data type of Matrix


class Matrix:

    def __init__(self, matrix: t_matrix):
        """
        In python, we could represent vectors as lists of numbers that says a vector is just a list of floats.
        A matrix is a two-dimensional collection of numbers. We will represent matrices as lists of Vector.
        If a matrix has n rows and k columns, we will refer to it as an n × k matrix. We can think of each row
        of an n × k matrix as a vector of length k, and each column as a vector of length n.
        :param matrix: parameter to create matrix object
        >>> A = Matrix([])
        The given matrix is invalid!
        >>> A = Matrix([[2, 3, 3],[6, 6, 6]])
        >>> print(A)
        My matrix: [[2, 3, 3], [6, 6, 6]]
        """
        try:
            assert matrix  # if the given matrix is invalid such as '[]', then it will raise exception
            self.__matrix = matrix
            self.__row = len(self.__matrix)
            self.__col = len(self.__matrix[0])
        except Exception:
            print('The given matrix is invalid!')

    def shape(self) -> Tuple:
        """
        :return: return a tuple of the shape of the self.__matrix
        >>> A = Matrix([[1, 2], [2, 4]])
        >>> A.shape()
        (2, 2)
        """
        return self.__row, self.__col

    def get_row(self, i: int) -> t_vector:
        """
        Returns the i-th row of self.__matrix (as a Vector)
        >>> A = Matrix([[1,2,3],[4,5,6]])
        >>> A.get_row(2)
        [4, 5, 6]
        >>> A.get_row(3)
        i is too big!
        []
        """
        if i > self.__row:
            print("i is too big!")
            return []
        elif i <= 0:
            print('Input error! i must be bigger than zero!')
            return []
        else:
            return self.__matrix[i - 1]  # self.__matrix[i-1] is already the ith row

    def get_column(self, j: int) -> t_vector:
        """
        Returns the j-th column of self.__matrix (as a Vector)
        >>> A = Matrix([[1,2,3],[4,5,6]])
        >>> A.get_column(2)
        [2, 5]
        >>> A.get_column(0)
        Input error! j must be bigger than zero!
        []
        """
        if j > self.__col:
            print('j is too big!')
            return []
        elif j <= 0:
            print('Input error! j must be bigger than zero!')
            return []
        else:
            return [temp_row[j - 1]  # jth element of row temp_row
                    for temp_row in self.__matrix]  # for each row temp_row

    def add(self, matrix: t_matrix) -> t_matrix:
        """
        Add corresponding elements of two matrices. Don't change the self.__matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> A.add([[7, 8, 9], [1, 2, 3]])
        [[8, 10, 12], [5, 7, 9]]
        >>> A.add([[12, 3], [4, 5]])
        Two matrices must be same shape!
        """
        try:
            assert self.__row == len(matrix)
            assert self.__col == len(matrix[0])
            return [[self_i + matrix_i for self_i, matrix_i in zip(i, j)]
                    for i, j in zip(self.__matrix, matrix)]  # add corresponding elements
        except Exception:
            print("Two matrices must be same shape!")

    def subtract(self, matrix: t_matrix) -> t_matrix:
        """
        Subtract corresponding elements of two matrices. Don't change the self.__matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> A.subtract([[2, 3, 3], [6, 6, 6]])
        [[-1, -1, 0], [-2, -1, 0]]
        """
        try:
            assert self.__row == len(matrix)
            assert self.__col == len(matrix[0])
            return [[self_i - matrix_i for self_i, matrix_i in zip(i, j)]
                    for i, j in zip(self.__matrix, matrix)]  # subtract corresponding elements

        except Exception:
            print("Two matrices must be same shape!")

    def scalar_multiply(self, k: float) -> t_matrix:
        """
        Multiplies every element by k. Don't change the self.__matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> A.scalar_multiply(3)
        [[3, 6, 9], [12, 15, 18]]
        """
        return [[j * k for j in i] for i in self.__matrix]

    def dot_product(self, matrix: t_matrix) -> t_matrix:
        """
        Don't change the self.__matrix
        :param matrix: another matrix need be multiplied
        :return: the dot product about two matrices
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> A.dot_product([[5, 6], [7, 8]])
        [[19, 22], [43, 50]]
        >>> A.dot_product([[5, 6], [7, 8], [9, 10]])   # this matrix can't not to be dot product, will raise error.
        Input error! self.__matrix's column number must be equal to matrix's row number!
        """
        try:
            assert self.__col == len(matrix)
            temp_row = []
            temp_col = []
            temp_matrix = []
            # use every row in self.__matrix to multiply matrix's corresponding column and sum them to make a new matrix
            for row_a in self.__matrix:
                for i in range(len(matrix[0])):
                    for row_b in matrix:
                        temp_col.append(row_b[i])
                    # sum self.__matrix's row elements multiply corresponding elements of matrix
                    temp_row.append(sum([i * j for i, j in zip(row_a, temp_col)]))
                    temp_col = []
                temp_matrix.append(temp_row)
                temp_row = []
            return temp_matrix
        except Exception:
            print("Input error! self.__matrix's column number must be equal to matrix's row number!")

    def element_wise_product(self, matrix: t_matrix) -> t_matrix:
        """
        Each element of two same dimensional matrices is multiplied separately and returned. Don't' change self.__matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> A.element_wise_product([[7, 8, 9], [1, 2, 3]])
        [[7, 16, 27], [4, 10, 18]]
        >>> A.element_wise_product([[4, 5, 6, 6],[8, 8, 9, 8]])
        Two matrices must be same shape!
        """
        try:
            assert self.__row == len(matrix)
            assert self.__col == len(matrix[0])
            return [[self_i * matrix_i for self_i, matrix_i in zip(i, j)]
                    for i, j in zip(self.__matrix, matrix)]  # Multiplies corresponding elements
        except Exception:
            print("Two matrices must be same shape!")

    def transpose(self) -> t_matrix:
        """
        :return: return the transpose of the self.__matrix, don't change the self.__matrix
        >>> A = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        >>> A.transpose()
        [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        """
        return [self.get_column(i + 1) for i in range(self.__col)]

    @staticmethod
    def make_matrix(num_rows: int, num_cols: int, func: Callable) -> t_matrix:
        """
        Returns a num_rows x num_cols matrix
        whose (i,j)-th entry is func(i, j)
        >>> Matrix.make_matrix(2, 3, lambda i, j: 1)
        [[1, 1, 1], [1, 1, 1]]
        """
        if num_rows > 0 and num_cols > 0:
            return [[func(i, j)  # given i, create a list
                     for j in range(num_cols)]  # [func(i,0),func(i,1)……]
                    for i in range(num_rows)]  # create on list for each i
        else:
            print('Input error! i and j must both be bigger than zero.')
            return []

    @staticmethod
    def identity_matrix(n: int) -> t_matrix:
        """
        Returns the n x n identity matrix
        >>> Matrix.identity_matrix(3)
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        """
        if n > 0:
            return Matrix.make_matrix(n, n, lambda i, j: 1 if i == j else 0)
        else:
            print("Input error! n must be bigger than zero!")

    def add_row(self, i: int, n: int, k: float) -> t_matrix:
        """
        Row i multiplied by k, added to row n, change row n in self.__matrix
        >>> A = Matrix([[1, 2], [1, 2], [1, 2]])
        >>> print(A.add_row(0,1,3.5))
        [[1, 2], [4.5, 9.0], [1, 2]]
        """
        if k == 0:
            return self.__matrix
        for index in range(self.__col):
            self.__matrix[n][index] = self.__matrix[n][index] + self.__matrix[i][index] * k
        return self.__matrix

    def swap_row(self, i: int, n: int) -> t_matrix:
        """
        Swap i row and n row, change the self.__matrix
        >>> A = Matrix([[1,2],[3,4]])
        >>> print(A.swap_row(0,1))
        [[3, 4], [1, 2]]
        """
        temp = self.__matrix[i]
        self.__matrix[i] = self.__matrix[n]
        self.__matrix[n] = temp
        return self.__matrix

    def elimination_matrix(self) -> t_matrix:
        """
        In my opinion, elimination matrix is self.__matrix's row-reduced echelon matrix.So this method
        is use linear algebra knowledge to get self.__matrix's elimination matrix, change the self.__matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6] ,[6, 8, 8]])
        >>> A.elimination_matrix()
        [[1, 2, 3], [0.0, -3.0, -6.0], [0.0, 0.0, -2.0]]
        """
        j = 0  # i, j is the coordinate of the data on the diagonal, plus 1 at the same time
        for i in range(self.__row):
            # If the first element is found to be 0, look for the row with the first element not 0 to exchange with it
            if self.__matrix[i][j] == 0:
                k = i + 1
                while True:
                    """If the row whose first element is not 0 cannot be found, the corresponding columns of the 
                    matrix are all 0, ending the function"""
                    if k >= self.__row:
                        return self.__matrix
                    if self.__matrix[k][j] != 0:
                        """After finding the row with the unsimplified column header element of 0, exchange the 
                        corresponding two rows """
                        self.swap_row(k, i)
                        break
                    k += 1
            # r_i records the next row, starting from the next row, and the first element of each row becomes 0
            r_i = i + 1
            c_j = i  # c_j records number of traversal columns
            # Since I done the next line traversing, after operation all the column first elements into 0
            for t_row in range(r_i, self.__row):
                # Calculate the factor that you need to multiply to make the first element of the line 0
                reason = (-1) * self.__matrix[t_row][c_j] / self.__matrix[i][c_j]
                self.add_row(i, t_row, reason)  # The corresponding column head element is 0
            j += 1
        return self.__matrix

    def inverse_matrix(self) -> t_matrix:
        """
        Use numpy to calculate the inverse matrix of self.__matrix
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> A.inverse_matrix()
        matrix([[-2. ,  1. ],
                [ 1.5, -0.5]])
        >>> A = Matrix([[1, 2], [3, 4],[5, 6]])
        >>> A.inverse_matrix()
        The matrix must be n x n type!
        """
        try:
            assert self.__row == self.__col
            matrix_i = np.matrix(self.__matrix).I
            return matrix_i
        except Exception:
            print("The matrix must be n x n type!")

    def permutation_matrix(self) -> t_matrix:
        """
        I'm sorry that I don't solve this problem.
        I think there are many permutation matrices for self.__matrix, I don't understand to find which one.
        """
        pass

    def reset_matrix(self, matrix: t_matrix):
        """
        Reset self.__matrix, change the self.__matrix
        :param matrix: new matrix that is given.
        :return: the result of operation
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> A.reset_matrix([[2, 3, 4], [7, 8, 9]])
        'Reset matrix success! new matrix: [[2, 3, 4], [7, 8, 9]]'
        >>> print(A)
        My matrix: [[2, 3, 4], [7, 8, 9]]
        """
        try:
            assert matrix  # if the given matrix is empty such as '[]', then it will raise exception
            self.__matrix = matrix
            self.__row = len(self.__matrix)
            self.__col = len(self.__matrix[0])
            return 'Reset matrix success! new matrix: {}'.format(self.__matrix)
        except Exception:
            print('The given matrix is invalid!')

    def __str__(self):
        """
        :return: return the string type of self.__matrix
        >>> A = Matrix([[1,2,3],[4,5,6]])
        >>> print(A)
        My matrix: [[1, 2, 3], [4, 5, 6]]
        """
        return 'My matrix: {}'.format(self.__matrix)

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
