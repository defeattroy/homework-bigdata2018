from typing import Callable
from typing import List
from typing import Tuple

Matrix = List[List[float]]
Vector = List[float]


class Matrix1(object):
    """
    the class has three attributes: num_rows, num_cols,entry_fn,a
    a is the list of list(matrix)
    """

    def __init__(self, num_rows: int,
                 num_cols: int,
                 entry_fn: Callable[[int, int], float]):
        """
        :param num_rows: the numbers of row
        :param num_cols: the numbers of column
        :param entry_fn: a name of function which is used to produce value of matrix
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.entry_fn = entry_fn
        self.a = self._make_matrix()                       # a is the list of list(matrix)

    def _make_matrix(self) -> Matrix:
        """
        :return:  return a matrix(list of list)
        """
        return [[self.entry_fn(i, j)
                 for j in range(self.num_cols)]
                for i in range(self.num_rows)]

    def shape(self) -> Tuple[int, int]:
        """ return a tuple which displays the size of matrix
        >>> from Matrix import Matrix1
        >>> m = Matrix1(2,2,lambda x, y: 0)
        >>> m.shape()
        (2, 2)
        >>> m1 = Matrix1(3,6,lambda x, y: 0)
        >>> m1.shape()
        (3, 6)
        """
        return self.num_rows, self.num_cols

    def get_row(self, i: int) -> Vector:
        """ return the row vector of matrix
        >>> m = Matrix1(2,2,lambda i, j: 1 if i==j else 0)
        >>> m.get_row(0)
        [1, 0]
        >>> m.get_row(1)
        [0, 1]
        """
        return self.a[i]

    def get_column(self, j: int) -> Vector:
        """ return the column vector of matrix
        >>> m = Matrix1(2,2,lambda i, j: 1 if i==j else 0)
        >>> m.get_column(0)
        [1, 0]
        >>> m.get_column(1)
        [0, 1]
        >>> m1 = Matrix1(3,3,lambda i, j: 1 if i==j else 0)
        >>> m1.get_column(2)
        [0, 0, 1]
        >>> m1.get_column(1)
        [0, 1, 0]
        """
        return [a_i[j] for a_i in self.a]

    def get_item(self, i: int, j: int) -> float:
        """  return the value of matrix, (i,j) is the index
        >>> m = Matrix1(2,2,lambda i, j: 1 if i==j else 0)
        >>> m.get_item(0,1)
        0
        >>> m.get_item(0,0)
        1
        >>> m.get_item(1,1)
        1
        >>> m1 = Matrix1(3,3,lambda i, j: 1 if i==j else 0)
        >>> m1.get_item(0,2)
        0
        >>> m1.get_item(0,0)
        1
        >>> m1.get_item(2,2)
        1
        >>> m1.get_item(1,1)
        1
        """
        return self.a[i][j]

    def set_item(self, x: float, i: int, j: int):
        """  reset the value of matrix, (i,j) is the index
         >>> m1 = Matrix1(3,3,lambda i, j: 1 if i==j else 0)
         >>> m1.set_item(3.0,0,0)
         >>> m1.set_item(3.0,0,0)
         >>> m1.set_item(3.0,0,0)
         >>> m1.get_item(0,0)
         3.0
         >>> m1.set_item(3.0,1,0)
         >>> m1.get_item(1,0)
         3.0
        """
        self.a[i][j] = x


if __name__ == "__main__":
    import doctest
    doctest.testmod()
