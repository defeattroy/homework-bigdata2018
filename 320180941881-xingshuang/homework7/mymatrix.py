"""
This module provides two classes: Vector and Matrix
Vector class includes 11 functions:
__add__,__sub__,lenth,dot_product,scalar ...
Matrix class includes 17 functions:
add,__add__,sub,mul,transpose,dot_product ...

>>> a=Vector(3,4)
>>> b=Vector(2,2)
>>> a,b
(Vector(3,4), Vector(2,2))
>>> a==b
False
>>> a+b
Vector(5,6)
>>> a-b
Vector(1,2)
>>> a.lenth
5.0
>>> a.dot_product(b)
14
>>> a.scalar(2)
Vector(6,8)
"""

import math
class Vector:
    """
    Vector class is a two-dimensional vector class.
    """
    def __init__(self,x=0,y=0):
        self.x=x
        self.y=y

    def __repr__(self):
        """
        >>> c=eval("Vector(5,5)")
        >>> c
        Vector(5,5)
        """
        return "Vector({0.x!r},{0.y!r})".format(self)

    def __str__(self):
        """
        >>> d=Vector(3,7)
        >>> print(d)
        (3,7)
        """
        return "({0.x!r},{0.y!r})".format(self)

    def __eq__(self, other):
        return self.x==other.x and self.y==other.y

    def __add__(self, other):
        return Vector(self.x+other.x,self.y+other.y)

    def __iadd__(self, other):
        """
        >>> a=Vector(3,3)
        >>> b=Vector(2,2)
        >>> a+=b
        >>> a
        Vector(5,5)
        """
        self.x=self.x+other.x
        self.y=self.y+other.y
        return self

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __isub__(self, other):
        """
        >>> a=Vector(3,3)
        >>> b=Vector(2,2)
        >>> a-=b
        >>> a
        Vector(1,1)
        """
        self.x=self.x-other.x
        self.y=self.y-other.y
        return self

    @property
    def lenth(self):
        """
        Calculating the length of a vector
        """
        return math.sqrt(self.x*self.x+self.y*self.y)

    def dot_product(self,other):
        """
        Calculating the dot product of two vectors
        """
        assert isinstance(other,Vector)
        return self.x * other.x + self.y * other.y

    def scalar(self,num:int):
        """
        Calculating the multiplication of a integer and a vector
        :param num:
        :return:
        """
        return Vector(self.x*num,self.y*num)


class Matrix(object):
    """
    Matrix class provides a few common functions
    >>> a=Matrix([[5,2,4],[3,8,2],[6,0,4],[0,1,6]])
    >>> b=Matrix([[2,4],[1,3],[3,2]])
    >>> c=Matrix([[3,3,3],[3,3,3],[3,3,3]])
    >>> d=Matrix([[2,2,2],[2,2,2],[2,2,2]])
    >>> e=Matrix([[2,2,2],[2,2,2],[2,2,2]])
    >>> print(c+d) # doctest: +NORMALIZE_WHITESPACE
    5  5  5
    5  5  5
    5  5  5
    >>> print(c-d) # doctest: +NORMALIZE_WHITESPACE
    1  1  1
    1  1  1
    1  1  1
    >>> print(a.mul(b))# doctest: +NORMALIZE_WHITESPACE
    24  34
    20  40
    24  32
    19  15
    >>> a==b
    False
    >>> d==e
    True
    """

    def __init__(self,List):
        assert isinstance(List,list)

        self.matrix=List
        self.shape=(len(List),len(List[0]))
        self.row=self.shape[0]
        self.column=self.shape[1]


    def __str__(self):
        Str = ''
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                Str += str(self.matrix[i][j])
                Str +='  '
            Str += '\n'
        return Str

    def getShape(self):
        """
        >>> a=Matrix([[1,2,3],[4,5,6]])
        >>> a.getShape()
        (2, 3)
        """
        return self.shape

    def __eq__(self, other):
        if self.shape == other.shape:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if __name__ == '__main__':
                        if not self.matrix[i][j]==other.matrix[i][j]:
                            return False
            return True
        else:
            return False

    def __add__(self, other):
        assert isinstance(other, Matrix)
        assert self.shape == other.shape

        result = []
        for i in range(self.shape[0]):
            result.append([])
            for j in range(self.shape[1]):
                result[i].append(self.matrix[i][j] + other.matrix[i][j])
        return Matrix(result)

    def add(self,other):
        """
        >>> a=Matrix([[1,1],[2,2]])
        >>> b=Matrix([[2,2],[1,1]])
        >>> print( a.add(b) )# doctest: +NORMALIZE_WHITESPACE
        3  3
        3  3
        """
        assert isinstance(other,Matrix)
        assert self.shape==other.shape

        result=[]
        for i in range(self.shape[0]):
            result.append([])
            for j in range(self.shape[1]):
                result[i].append(self.matrix[i][j] + other.matrix[i][j])
        return Matrix(result)

    def __iadd__(self, other):
        """
        >>> a=Matrix([[1,1],[2,2]])
        >>> b=Matrix([[2,2],[1,1]])
        >>> a+=b
        >>> print( a )# doctest: +NORMALIZE_WHITESPACE
        3  3
        3  3
        """
        assert isinstance(other, Matrix)
        assert self.shape == other.shape

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.matrix[i][j] += other.matrix[i][j]
        return self

    def __sub__(self, other):
        assert isinstance(other, Matrix)
        assert self.shape == other.shape

        result = []
        for i in range(self.shape[0]):
            result.append([])
            for j in range(self.shape[1]):
                result[i].append(self.matrix[i][j] - other.matrix[i][j])
        return Matrix(result)

    def sub(self,other):
        """
        >>> a=Matrix([[2,2],[2,2]])
        >>> b=Matrix([[1,1],[1,1]])
        >>> print( a.sub(b) )# doctest: +NORMALIZE_WHITESPACE
        1  1
        1  1
        """
        assert isinstance(other,Matrix)
        assert self.shape==other.shape

        result=[]
        for i in range(self.shape[0]):
            result.append([])
            for j in range(self.shape[1]):
                result[i].append(self.matrix[i][j] - other.matrix[i][j])
        return Matrix(result)

    def __isub__(self, other):
        """
        >>> a=Matrix([[2,2],[2,2]])
        >>> b=Matrix([[1,1],[1,1]])
        >>> a-=b
        >>> print( a )# doctest: +NORMALIZE_WHITESPACE
        1  1
        1  1
        """
        assert isinstance(other, Matrix)
        assert self.shape == other.shape

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.matrix[i][j] -= other.matrix[i][j]
        return self

    def mul(self, other):
        assert isinstance(other, Matrix)
        assert self.shape[1] == other.shape[0]

        result = []
        for i in range(self.shape[0]):
            result.append([])
            for k in range(other.shape[1]):
                aaa = 0
                for m in range(self.shape[1]):
                    aaa += self.matrix[i][m] * other.matrix[m][k]
                result[i].append(aaa)
        return Matrix(result)

    def scalar(self,num:int):
        """
        >>> a=Matrix([[1,1],[1,1]])
        >>> print(a.scalar(5)) # doctest: +NORMALIZE_WHITESPACE
        5  5
        5  5
        """
        result=[]
        for i in range(self.shape[0]):
            result.append([])
            for j in range(self.shape[1]):
                result[i].append(self.matrix[i][j]*num)
        return Matrix(result)

    def transpose(self):
        """
        >>> a=Matrix([[1,2,3],[4,5,6]])
        >>> print(a) # doctest: +NORMALIZE_WHITESPACE
        1  2  3
        4  5  6
        >>> print(a.transpose()) # doctest: +NORMALIZE_WHITESPACE
        1  4
        2  5
        3  6
        """
        result=[]
        for j in range(self.shape[1]):
            result.append([])
            for i in range(self.shape[0]):
                result[j].append(self.matrix[i][j])
        return Matrix(result)

    @classmethod
    def eye(cls,n=1):
        """
        >>> print( Matrix.eye(5) ) # doctest: +NORMALIZE_WHITESPACE
        1  0  0  0  0
        0  1  0  0  0
        0  0  1  0  0
        0  0  0  1  0
        0  0  0  0  1
        """
        ans=[]
        for i in range(n):
            ans.append([])
            for j in range(n):
                if i==j:
                    ans[i].append(1)
                else:
                    ans[i].append(0)
        return Matrix(ans)

    def dot_product(self,other):
        """
        >>> a=Matrix([[2,3],[7,5]])
        >>> b=Matrix([[1,4],[9,5]])
        >>> print(a.dot_product(b))  # doctest: +NORMALIZE_WHITESPACE
        2  12
        63  25
        """
        assert self.shape == other.shape

        result=[]
        for i in range(self.shape[0]):
            result.append([])
            for j in range(self.shape[1]):
                result[i].append(self.matrix[i][j]*other.matrix[i][j])
        return Matrix(result)

    def exchange_row(self,a,b):
        """
        >>> a=Matrix([[1,1,1],[2,2,2],[3,3,3]])
        >>> print(a) # doctest: +NORMALIZE_WHITESPACE
        1  1  1
        2  2  2
        3  3  3
        >>> print(a.exchange_row(0,2)) # doctest: +NORMALIZE_WHITESPACE
        3  3  3
        2  2  2
        1  1  1
        """
        for j in range(self.shape[1]):
            tmp = self.matrix[a][j]
            self.matrix[a][j] = self.matrix[b][j]
            self.matrix[b][j] = tmp
        return self

    def exchange_column(self,a,b):
        """
        >>> a=Matrix([[1,2,3],[1,2,3],[1,2,3]])
        >>> print(a) # doctest: +NORMALIZE_WHITESPACE
        1  2  3
        1  2  3
        1  2  3
        >>> print(a.exchange_column(0,2)) # doctest: +NORMALIZE_WHITESPACE
        3  2  1
        3  2  1
        3  2  1
        """
        for i in range(self.shape[0]):
            tmp = self.matrix[i][a]
            self.matrix[i][a] = self.matrix[i][b]
            self.matrix[i][b] = tmp
        return self


if __name__ =="__main__":
    import doctest
    doctest.testmod()
