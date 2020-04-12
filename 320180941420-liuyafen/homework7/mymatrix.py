import doctest

class Matrix(object):
    '''
    >>> list_1 = [[1, 1], [2, 2]]
    >>> list_2 = [[2, 2], [3, 3]]
    >>> list_3 = [[1, 1, 1], [2, 2, 2]]
    >>> mat1 = Matrix(list_1)
    >>> mat2 = Matrix(list_2)
    >>> mat3 = Matrix(list_3)
    >>> mat1.matrix
    [[1, 1], [2, 2]]
    >>> mat2.matrix
    [[2, 2], [3, 3]]
    >>> mat3.matrix
    [[1, 1, 1], [2, 2, 2]]
    >>> mat4 = mat1.matrix_addition(mat2)
    >>> mat4.matrix
    [[3, 3], [5, 5]]
    >>> mat5 = mat2.matrix_subtraction(mat1)
    >>> mat5.matrix
    [[1, 1], [1, 1]]
    >>> mat6 = mat1.matrix_multiplication(mat3)
    >>> mat6.matrix
    [[3, 3, 3], [6, 6, 6]]
    >>> mat1.matrix_equal(mat2)
    False
    >>> mat7=mat1.matrix_transpose()
    >>> mat7.matrix
    [[1, 2], [1, 2]]
    '''

    def __init__(self, list_a):    #list_a is list
        self.matrix = list_a       #make list_a become matrix
        self.shape = (len(list_a), len(list_a[0]))  #calculate list_a's row and column
        self.row = self.shape[0]
        self.column = self.shape[1]

    def build_zero_value_matrix(self, shape):    #set a 0 matrix, shape is tuple
        zero_value_mat = []
        for i in range(shape[0]):
            zero_value_mat.append([])
            for j in range(shape[1]):
                zero_value_mat[i].append(0)
        zero_value_matrix = Matrix(zero_value_mat)
        return zero_value_matrix

    def matrix_addition(self, the_second_mat):   #addition of matrix
        if the_second_mat.shape == self.shape:   #they must have same shape
            result_mat = self.build_zero_value_matrix(self.shape)  #use rusule_mat deposit result
            for i in range(self.row):
                for j in range(self.column):
                    result_mat.matrix[i][j] = self.matrix[i][j] + the_second_mat.matrix[i][j]
            return result_mat

    def matrix_subtraction(self, the_second_mat):  #subtraction of matrix
        if the_second_mat.shape == self.shape:     #they must have same shape
            result_mat = self.build_zero_value_matrix(self.shape)  #use rusule_mat deposit result
            for i in range(self.row):
                for j in range(self.column):
                    result_mat.matrix[i][j] = self.matrix[i][j] - the_second_mat.matrix[i][j]
            return result_mat

    def matrix_multiplication(self, the_second_mat): #multiplication of matrix
        if self.shape[1] == the_second_mat.shape[0]:
        #The number of columns of the first matrix does not match the number of rows of the second matrix and cannot be multiplied
            shape = (self.shape[0], the_second_mat.shape[1])
            #columns of new shape is columns of the first matrix
            #rows of new shape is rows of the second matrix
            result_mat = self.build_zero_value_matrix(shape) #use rusule_mat deposit result
            for i in range(self.shape[0]):
                for j in range(the_second_mat.shape[1]):
                    number = 0
                    for k in range(self.shape[1]):
                        number += self.matrix[i][k] * the_second_mat.matrix[k][j]
                    result_mat.matrix[i][j] = number
            return result_mat

    def matrix_equal(self, the_second_mat):  #Compare whether two matrices are the same
        if the_second_mat.shape == self.shape:  #they must have same shape
            for i in range(self.row):
                for j in range(self.column):
                    if self.matrix[i][j] != the_second_mat.matrix[i][j]:
                        return False
            return True
        else:
            return False

    def matrix_transpose(self):   #transpose of matrix
        transpose_mat = self.build_zero_value_matrix(self.shape)  #use transpose_mat deposit result
        for i in range(self.row):
            for j in range(self.column):
                transpose_mat.matrix[i][j] = self.matrix[j][i]
        return transpose_mat


if __name__=='__main__':
    doctest.testmod(verbose=True)
