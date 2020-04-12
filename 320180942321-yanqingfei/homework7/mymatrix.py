class Matrix:
    def __init__(self, matrix: list):
        check = [len(item) for item in matrix]     # To judge whether a matrix can be constructed
        if set(check) == {0}:
            self.matrix = []
            self.row = 0
            self.col = 0
            self.shape = (self.row, self.col)
        elif len(set(check)) == 1:
            self.matrix = matrix
            self.row = len(matrix)
            self.col = len(matrix[0])
            self.shape = (self.row, self.col)
        elif len(set(check)) > 1:
            raise ValueError("It can't be a Matrix")

    def __repr__(self):              # Play the list as a Matrix
        mat = ""
        for item in self.matrix:
            mat += str(item)
            mat += '\t\n'
        return str(mat)

    def __add__(self, other):
        """
        >>> Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) + Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) #doctest: +NORMALIZE_WHITESPACE
        [2, 4, 6]
        [8, 10, 12]
        [14, 16, 18]
        <BLANKLINE>
        """
        new_matrix = [[0] * other.col for i in range(self.row)]   # constructed a new list to set the result
        if (self.row == other.row) and (self.col == other.col):
            for i in range(self.row):
                for j in range(self.col):
                    new_matrix[i][j] =self.matrix[i][j] + other.matrix[i][j]
        else:
            raise TypeError("The two Matrix's shape isn't equal")
        return Matrix(new_matrix)

    def __sub__(self, other):
        """
        >>> Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) - Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) #doctest: +NORMALIZE_WHITESPACE
        [0, 0, 0]
        [0, 0, 0]
        [0, 0, 0]
        <BLANKLINE>
        """
        new_matrix = [[0] * other.col for i in range(self.row)]
        if (self.row == other.row) and (self.col == other.col):
            for i in range(self.row):
                for j in range(self.col):
                    new_matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
        else:
            raise TypeError("The two Matrix's shape isn't equal")
        return Matrix(new_matrix)

    def __mul__(self, other):
        """
        >>> Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) #doctest: +NORMALIZE_WHITESPACE
        [30, 36, 42]
        [66, 81, 96]
        [102, 126, 150]
        <BLANKLINE>
        """
        new_matrix = [[0] * other.col for i in range(self.row)]
        if self.col == other.row:
            result = 0
            for i in range(self.row):
                for j in range(other.col):
                    result = 0
                    for k in range(self.col):
                        result += self.matrix[i][k] * other.matrix[k][j]
                    new_matrix[i][j] = result
        else:
            raise TypeError("These two Matrix can't multiply")
        return Matrix(new_matrix)


def transpose(self):
    """
    >>> transpose(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))    #doctest: +NORMALIZE_WHITESPACE
    [1, 4, 7]
    [2, 5, 8]
    [3, 6, 9]
    """
    new_matrix = [[0] * self.row for i in range(self.col)]
    for i in range(self.row):
        for j in range(self.col):
            new_matrix[i][j] = self.matrix[j][i]
    return Matrix(new_matrix)


def unit_matrix(n: int):   # constructed a unit matrix
    """
    >>> unit_matrix(3) #doctest: +NORMALIZE_WHITESPACE
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    <BLANKLINE>
    """
    new_matrix = [[0] * n for i in range(n)]
    for i in range(n):
        new_matrix[i][i] = 1
    return Matrix(new_matrix)


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)