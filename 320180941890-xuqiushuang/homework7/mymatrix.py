import copy
import doctest

#copy.deepcopy(x):The copy module's deepcopy method makes a full copy of the parent object and its children.That is,
#a new composite object is created and all the children are recursively copied. The new composite object has no relation to the original object.
class Matrix:
  # Define the matrix class
  def __init__(self, row, column, fill=0.0):#Initialize
    '''
    >>> matrix=Matrix(3,3,1.0)
    >>> matrix.show()
    1.0 1.0 1.0 
    1.0 1.0 1.0 
    1.0 1.0 1.0 
    >>> matrix.shape
    (3, 3)
    >>> matrix.row
    3
    >>> matrix.column
    3
    >>> matrix._matrix
    []
    
    '''
    self.shape = (row, column)
    self.row = row
    self.column = column
    self._matrix = [[fill]*column for i in range(row)]
    
  # Returns the value of the element m(I, j) : m[I, j] 
  def getitem(self,index):
    '''
    >>> matrix=Matrix(3,3,1.0)
    >>> matrix.__getitem__(1)
    [1.0, 1.0, 1.0]
    >>> matrix.__getitem__(2.0)
    >>>
    >>> matrix.__getitem__((1,2))
    1.0
    
    '''
    return self.__getitem__(index)
  
  def __getitem__(self, index):
    if isinstance(index, int):#The isinstance() function determines whether an object is a known type
      return self._matrix[index-1]
    elif isinstance(index, tuple):
      return self._matrix[index[0]-1][index[1]-1]
    
  # Set the value of the element m(I,j) to s: m[I,j] = s
  def setitem(self,index,value):
    '''
    >>> s=matrix[1,2]
    >>> s
    1.0
    
    '''
    return self.__setitem__(index,value)
 
  def __setitem__(self, index, value):
    if isinstance(index, int):
      self._matrix[index-1] = copy.deepcopy(value)
    elif isinstance(index, tuple):
      self._matrix[index[0]-1][index[1]-1] = value

  def eq(self,N):
    '''
    >>> matrix=Matrix(3,3,1.0)
    >>> matrix.eq(1)
    Traceback (most recent call last):
    File "<pyshell#8>", line 1, in <module>
    matrix.eq(1)
    File "H:\m.py", line 52, in eq
    return self.__eq__(N)
    File "H:\m.py", line 56, in __eq__
    assert isinstance(N, Matrix), "类型不匹配，不能比较"
    AssertionError: 类型不匹配，不能比较

    >>> matrix2=Matrix(3,3,1.0)
    >>> matrix.eq(matrix2)
    True
    
    '''
    return self.__eq__(N)
  
  def __eq__(self, N):
    # A == B
    assert isinstance(N, Matrix), "类型不匹配，不能比较"
    return N.shape == self.shape # Comparison of dimensions

  def add(self,N):
    '''
    >>> m1=Matrix(3,3,1.0)
    >>> m2=Matrix(3,3,2.0)
    >>> m1.show()
    1.0 1.0 1.0 
    1.0 1.0 1.0 
    1.0 1.0 1.0
    >>> m2.show()
    2.0 2.0 2.0 
    2.0 2.0 2.0 
    2.0 2.0 2.0
    >>> m3=m1.add(m2)
    >>> m3.show()
    3.0 3.0 3.0 
    3.0 3.0 3.0 
    3.0 3.0 3.0
    >>> m4=Matrix(1,1,1.0)
    >>> m5=m1.add(m4)
    Traceback (most recent call last):
    File "<pyshell#15>", line 1, in <module>
    m5=m1.add(m4)
    File "H:\m.py", line 77, in add
    return self.__add__(N)
    File "H:\m.py", line 82, in __add__
    assert N.shape == self.shape, "维度不匹配，不能相加"
    AssertionError: 维度不匹配，不能相加
    >>> 

    '''
    return self.__add__(N)

  def __add__(self, N):

    # A + B
    #Assert is used to determine an expression that fires an exception if the expression condition is false.
    assert N.shape == self.shape, "维度不匹配，不能相加"
    M = Matrix(self.row, self.column)
    for r in range(self.row):
      for c in range(self.column):
        M[r, c] = self[r, c] + N[r, c]
    return M
  
  def sub(self,N):
    '''
    >>> m6=m1.sub(m2)
    >>> m6
    <__main__.Matrix object at 0x044DBFB0>
    >>> m6.show()
    -1.0 -1.0 -1.0 
    -1.0 -1.0 -1.0 
    -1.0 -1.0 -1.0 
    >>>
    '''
    return self.__sub__(N)
  def __sub__(self, N):

    # A - B
    assert N.shape == self.shape, "维度不匹配，不能相减"
    M = Matrix(self.row, self.column)
    for r in range(self.row):
      for c in range(self.column):
        M[r, c] = self[r, c] - N[r, c]
    return M

  def mul(self,N):
    '''
    >>> m7=m1.mul(m2)
    >>> m7.show()
    6.0 6.0 6.0 
    6.0 6.0 6.0 
    6.0 6.0 6.0
    >>> m8=m1*2
    >>> m8.show()
    2.0 2.0 2.0 
    2.0 2.0 2.0 
    2.0 2.0 2.0
    >>> m9=m1.mul(m4)
    Traceback (most recent call last):
    File "<pyshell#28>", line 1, in <module>
    m9=m1.mul(m4)
    File "H:\m.py", line 102, in mul
    return self.__mul__(N)
    File "H:\m.py", line 112, in __mul__
    assert N.row == self.column, "维度不匹配，不能相乘"
    AssertionError: 维度不匹配，不能相乘
    
    '''
    return self.__mul__(N)
  def __mul__(self, N):

    # A * B (or：A * 2.0),Supports operations between matrices and matrices and between matrices and Numbers
    if isinstance(N, int) or isinstance(N,float):
      M = Matrix(self.row, self.column)
      for r in range(self.row):
        for c in range(self.column):
          M[r, c] = self[r, c]*N
    else:
      assert N.row == self.column, "维度不匹配，不能相乘"
      M = Matrix(self.row, N.column)
      for r in range(self.row):
        for c in range(N.column):
          sum = 0
          for k in range(self.column):
            sum += self[r, k] * N[k, r]
          M[r, c] = sum
    return M

  def pow(self,k):
    
    '''
    >>> ma=m1.pow(2)
    >>> ma.show()
    9.0 9.0 9.0
    9.0 9.0 9.0 
    9.0 9.0 9.0
    
    '''
    return self.__pow__(k)
  
  def __pow__(self, k):

    # A**k,power
    assert self.row == self.column, "不是方阵，不能乘方"
    M = copy.deepcopy(self)
    for i in range(k):
      M = M * self
    return M
  
  def transpose(self):
    
    '''
    >>> m1=Matrix(3,3,1.0)
    >>> m1.show()
    1.0 1.0 1.0 
    1.0 1.0 1.0 
    1.0 1.0 1.0 
    >>> m2=m1.transpose()
    >>> m2.show()
    1.0 1.0 1.0 
    1.0 1.0 1.0 
    1.0 1.0 1.0
    
    '''
    return self.__transpose__()
  
  def transpose(self):
    # transposition
    M = Matrix(self.column, self.row)
    for r in range(self.column):
      for c in range(self.row):
        M[r, c] = self[c, r]
    return M
  def zeros(self):
    # All zero matrices
    
    '''
    >>> m3=m1.zeros()
    >>> m3.show()
    0.0 0.0 0.0 
    0.0 0.0 0.0 
    0.0 0.0 0.0
    
    '''
    M = Matrix(self.column, self.row, fill=0.0)
    return M
  
  def identity(self):
    # Unit matrix
    '''
    >>> m2=Matrix(3,3,2.0)
    >>> m2.show()
    2.0 2.0 2.0 
    2.0 2.0 2.0 
    2.0 2.0 2.0 
    >>> m4=m2.identity()
    >>> m4.show()
    1.0 0.0 0.0 
    0.0 1.0 0.0 
    0.0 0.0 1.0
    
    '''
    assert self.row == self.column, "非n*n矩阵，无单位矩阵"
    M = Matrix(self.column, self.row)
    for r in range(self.row):
      for c in range(self.column):
        M[r, c] = 1.0 if r == c else 0.0
    return M
  def show(self):
    # show the matrix
    for r in range(self.row):
      for c in range(self.column):
        print(self[r+1, c+1],end=' ')
      print()

if __name__=='__main__':
  doctest.testmod(verbose=True)
