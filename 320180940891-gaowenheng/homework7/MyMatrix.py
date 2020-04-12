#!/usr/bin/env python
# coding: utf-8

# In[80]:


import copy
class Matrix:
    
  def __init__(A, row, column, fill=0.0):
    A.shape = (row, column)
    A.row = row
    A.column = column
    A._matrix = [[fill]*column for i in range(row)]
  def __getitem__(A, index):
    if isinstance(index, int):
      return A._matrix[index-1]
    elif isinstance(index, tuple):
      return A._matrix[index[0]-1][index[1]-1]
  def __setitem__(A, index, value):
    if isinstance(index, int):
      A._matrix[index-1] = copy.deepcopy(value)
    elif isinstance(index, tuple):
      A._matrix[index[0]-1][index[1]-1] = value
    
  def __add__(A, B):
    #加法
    assert B.shape == A.shape, "判断维度"
    M = Matrix(A.row, A.column)
    for r in range(A.row):
      for c in range(A.column):
        M[r, c] = A[r, c] + B[r, c]
    return M
  def __sub__(A, B):
    #减法
    # A - B
    assert B.shape == A.shape, "判断维度"
    M = Matrix(A.row, A.column)
    for r in range(A.row):
      for c in range(A.column):
        M[r, c] = A[r, c] - B[r, c]
    return M
  def __mul__(A, B):
    #乘法
    # A * B 
    if isinstance(B, int) or isinstance(B,float):
      M = Matrix(A.row, A.column)
      for r in range(A.row):
        for c in range(A.column):
          M[r, c] = A[r, c]*B
    else:
      assert B.row == A.column, "判断维度"
      M = Matrix(A.row, B.column)
      for r in range(A.row):
        for c in range(B.column):
          sum = 0
          for k in range(A.column):
            sum += A[r, k] * B[k, r]
          M[r, c] = sum
    return M
  def __pow__(A, k):
    #乘方
    # A**k
    assert A.row == A.column, "判断方阵"
    M = copy.deepcopy(A)
    for i in range(k):
      M = M * A
    return M
  def show(A):
    '''打印矩阵'''
    for r in range(A.row):
      for c in range(A.column):
        print(A[r+1, c+1],end=' ')
      print()
if __name__ == '__main__':
  m = Matrix(3,3,fill=2.0)
  n = Matrix(3,3,fill=3.5)
  m[1] = [1.,1.,1.]
  m[2] = [1.,2.,3.]
  m[3] = [3.,2.,1.]
  
  q = m*2.33
  p = m * n
  r = m**3
  print("m**3：")
  r.show()
  print()
  print("数乘2.33：")
  q.show()
  s = r*m
  print()
  print("m:")
  m.show()
  print()
  print("r*m：")
  s.show()
  print()
