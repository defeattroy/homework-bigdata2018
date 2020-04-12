import copy
class Matrix:
  '''矩阵类'''
  def __init__(self, row, column, f=0.0):
    self.shape = (row, column)
    self.row = row
    self.column = column
    self._matrix = [[f]*column for i in range(row)]
  # 返回元素m(i, j)的值: m[i, j]
  def __getitem__(self, index):
    if isinstance(index, int):
      return self._matrix[index-1]
    elif isinstance(index, tuple):
      return self._matrix[index[0]-1][index[1]-1]
  # 设置元素m(i,j)的值为s: m[i, j] = s
  def __setitem__(self, index, value):
    if isinstance(index, int):
      self._matrix[index-1] = copy.deepcopy(value)
    elif isinstance(index, tuple):
      self._matrix[index[0]-1][index[1]-1] = value
  def __eq__(self, N):
    '''相等'''
    # A == B
    assert isinstance(N, Matrix), "类型不匹配，不能比较"
    return N.shape == self.shape # 比较维度，可以修改为别的
  def __add__(self, N):
    '''加法'''
    # A + B
    assert N.shape == self.shape, "维度不匹配，不能相加"
    M = Matrix(self.row, self.column)
    for r in range(self.row):
      for c in range(self.column):
        M[r, c] = self[r, c] + N[r, c]
    return M
  def __sub__(self, N):
    '''减法'''
    # A - B
    assert N.shape == self.shape, "维度不匹配，不能相减"
    M = Matrix(self.row, self.column)
    for r in range(self.row):
      for c in range(self.column):
        M[r, c] = self[r, c] - N[r, c]
    return M
  def __mul__(self, N):
    '''乘法'''
    # A * B (或：A * 2.0)
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
  def __div__(self, N):
    '''除法'''
    # A / B
    pass
  def __pow__(self, k):
    '''乘方'''
    # A**k
    assert self.row == self.column, "不是方阵，不能乘方"
    M = copy.deepcopy(self)
    for i in range(k):
      M = M * self
    return M
  def invert(self):
    '''逆矩阵'''
    assert self.row == self.column, "不是方阵"
    M = Matrix(self.row, self.column*2)
    I = self.identity() # 单位矩阵
    I.show()#############################
    # 拼接
    for r in range(1,M.row+1):
      temp = self[r]
      temp.extend(I[r])
      M[r] = copy.deepcopy(temp)
    M.show()#############################
    # 初等行变换
    for r in range(1, M.row+1):
      # 本行首元素(M[r, r])若为 0，则向下交换最近的当前列元素非零的行
      if M[r, r] == 0:
        for rr in range(r+1, M.row+1):
          if M[rr, r] != 0:
            M[r],M[rr] = M[rr],M[r] # 交换两行
          break
      assert M[r, r] != 0, '矩阵不可逆'
      # 本行首元素(M[r, r])化为 1
      temp = M[r,r] # 缓存
      for c in range(r, M.column+1):
        M[r, c] /= temp
        print("M[{0}, {1}] /= {2}".format(r,c,temp))
      M.show()
      # 本列上、下方的所有元素化为 0
      for rr in range(1, M.row+1):
        temp = M[rr, r] # 缓存
        for c in range(r, M.column+1):
          if rr == r:
            continue
          M[rr, c] -= temp * M[r, c]
          print("M[{0}, {1}] -= {2} * M[{3}, {1}]".format(rr, c, temp,r))
        M.show()
    # 截取逆矩阵
    N = Matrix(self.row,self.column)
    for r in range(1,self.row+1):
      N[r] = M[r][self.row:]
    return N
  def transpose(self):
    '''转置'''
    M = Matrix(self.column, self.row)
    for r in range(self.column):
      for c in range(self.row):
        M[r, c] = self[c, r]
    return M
  def cofactor(self, row, column):
    '''代数余子式（用于行列式展开）'''
    assert self.row == self.column, "不是方阵，无法计算代数余子式"
    assert self.row >= 3, "至少是3*3阶方阵"
    assert row <= self.row and column <= self.column, "下标超出范围"
    M = Matrix(self.column-1, self.row-1)
    for r in range(self.row):
      if r == row:
        continue
      for c in range(self.column):
        if c == column:
          continue
        rr = r-1 if r > row else r
        cc = c-1 if c > column else c
        M[rr, cc] = self[r, c]
    return M
  def det(self):
    '''计算行列式(determinant)'''
    assert self.row == self.column,"非行列式，不能计算"
    if self.shape == (2,2):
      return self[1,1]*self[2,2]-self[1,2]*self[2,1]
    else:
      sum = 0.0
      for c in range(self.column+1):
        sum += (-1)**(c+1)*self[1,c]*self.cofactor(1,c).det()
      return sum
  def zeros(self):
    '''全零矩阵'''
    M = Matrix(self.column, self.row, f=0.0)
    return M
  def ones(self):
    '''全1矩阵'''
    M = Matrix(self.column, self.row, f=1.0)
    return M
  def identity(self):
    '''单位矩阵'''
    assert self.row == self.column, "非n*n矩阵，无单位矩阵"
    M = Matrix(self.column, self.row)
    for r in range(self.row):
      for c in range(self.column):
        M[r, c] = 1.0 if r == c else 0.0
    return M
  def show(self):
    '''打印矩阵'''
    for r in range(self.row):
      for c in range(self.column):
        print(self[r+1, c+1],end=' ')
      print()
if __name__ == '__main__':
  m = Matrix(3,3,f=2.0)
  n = Matrix(3,3,f=3.5)
  m[1] = [1.,1.,2.]
  m[2] = [1.,2.,1.]
  m[3] = [2.,1.,1.]
  p = m * n
  q = m*2.1
  r = m**3
  #r.show()
  #q.show()
  #print(p[1,1])
  #r = m.invert()
  #s = r*m
  p.show()
  print()
  m.show()
  print()
  r.show()
  print()
  print(m.det())
