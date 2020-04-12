class Matrix:
    """This class implements matrix and partial operations of matrix
 """

    def __init__(self, M: list):
        """constructor"""
        self.M = M
        self.row = len(self.M)
        self.column = len(self.M[0])

    def __add__(self, other):
        """Matrix addition

        >>> A = Matrix([[0,0,0],[0,0,0]])
        >>> B = Matrix([[1,1,1],[1,1,1]])
        >>> C = Matrix([[2,2],[2,2],[2,2]])
        >>> A+B
        [[1, 1, 1], [1, 1, 1]]
        >>> B+C
        Can't add.
        >>> A+C
        Can't add.
        """
        if self.row == other.row and self.column == other.column:
            added = []
            for i in range(0, self.row):
                ad = []
                for j in range(0, self.column):
                    ad.append(self.M[i][j] + other.M[i][j])
                added.append(ad)
            return added
        else:
            print("Can't add.")
            return None

    def __sub__(self, other):
        """Matrix subtraction

        >>> A = Matrix([[0,0,0],[0,0,0]])
        >>> B = Matrix([[1,1,1],[1,1,1]])
        >>> C = Matrix([[2,2],[2,2],[2,2]])
        >>> A-B
        [[-1, -1, -1], [-1, -1, -1]]
        >>> B-A
        [[1, 1, 1], [1, 1, 1]]
        >>> A-C
        Can't sub.
        """
        if self.row == other.row and self.column == other.column:
            subed = []
            for i in range(0, self.row):
                su = []
                for j in range(0, self.column):
                    su.append(self.M[i][j] - other.M[i][j])
                subed.append(su)
            return subed
        else:
            print("Can't sub.")
            return None

    def __mul__(self, other):
        """Matrix scalar multiplication

        >>> A = Matrix([[0,0,0],[0,0,0]])
        >>> B = Matrix([[1,1,1],[1,1,1]])
        >>> C = Matrix([[2,2],[2,2],[2,2]])
        >>> B*3
        [[3, 3, 3], [3, 3, 3]]
        >>> C*2
        [[4, 4], [4, 4], [4, 4]]
        """
        muled = []
        for i in range(0, self.row):
            mu = []
            for j in range(0, self.column):
                mu.append(other * self.M[i][j])
            muled.append(mu)
        return muled

    def __truediv__(self, other):
        """Matrix transpose

        >>> B = Matrix([[1,1,1],[1,1,1]])
        >>> B/2
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        >>> B/0
        Can't truediv.
        """
        if other == 0:
            print("Can't truediv.")
            return None
        truedived = []
        for i in range(0, self.row):
            tr = []
            for j in range(0, self.column):
                tr.append(self.M[i][j] / other)
            truedived.append(tr)
        return truedived

    def __pow__(self, x):
        """Matrix power

        """
        import copy
        assert self.row == self.column
        po = copy.deepcopy(self)
        for i in range(x):
            po = po * self
        return po

    def transpose(self):
        """Matrix transpose

        >>> A = Matrix([[0,0,0],[0,0,0]])
        >>> A.transpose()
        [[0, 0], [0, 0], [0, 0]]
        """
        transposed = []
        for i in range(self.column):
            item = []
            for index in range(self.row):
                item.append(self.M[index][i])
            transposed.append(item)
        return transposed

    def __eq__(self, other):
        """Determine whether matrices are equal

        >>> A = Matrix([[0,0,0],[0,0,0]])
        >>> B = Matrix([[1,1,1],[1,1,1]])
        >>> C = Matrix([[2,2],[2,2],[2,2]])
        >>> D = Matrix([[2,2],[2,2],[2,2]])
        >>> A==B
        False
        >>> C==D
        True
        """
        return self.M == other.M

    def __repr__(self):              # Play the list as a Matrix
        """Determine whether matrices are equal

         >>> A = Matrix([[0,0,0],[0,0,0]])
         >>> B = Matrix([[1,1,1],[1,1,1]])
         >>> C = Matrix([[2,2],[2,2],[2,2]])
         >>> D = Matrix([[2,2],[2,2],[2,2]])
         >>> A
        [0, 0, 0]
        [0, 0, 0]
         """
        mat = ""
        for item in self.M:
            mat += str(item)
            mat += '\t\n'
        return str(mat)

if __name__ == '__main__':
    import doctest

    doctest.testmod()