from Matrix import Matrix1

"""
 this module has some functions of Matrix1:
 matrix_add: add the two matrix
 matrix_sub:  subtract the two matrix
 matrix_multi: multiply the two matrix
 matrix_transpose: transpose the matrix
"""


def matrix_add(m1: Matrix1, m2: Matrix1) -> Matrix1:
    """  add two matrix  and their dimension must be the same
    >>> m1 = Matrix1(3,3,lambda i ,j: 1 if i==j else 0)      # 生成两个相同的对角矩阵
    >>> m2 =  Matrix1(3,3,lambda i, j: 1 if i==j else 0)
    >>> print(m1.a)
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> print(m2.a)
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> t = matrix_add(m1,m2)              # 将两个矩阵相加后得到新的矩阵，为之前的矩阵之和
    >>> t.a                                # 输出之后的矩阵
    [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    >>> m3 = Matrix1(1,1,lambda i, j: 1)
    >>> m3 = Matrix1(1,1,lambda i, j: 1)
    >>> matrix_add(m1,m3)                   # 两个维度不同的矩阵要相加时
    please input the same dimensions!
    """
    if m1.shape() != m2.shape():
        print('please input the same dimensions!')
    else:
        m = Matrix1(m1.num_rows, m1.num_cols, lambda i, j: 0)
        m.a = [[m1.get_item(i,j) + m2.get_item(i, j)
                for j in range(m1.num_cols)]
               for i in range(m1.num_rows)]
        return m


def matrix_sub(m1: Matrix1, m2: Matrix1) -> Matrix1:
    """  subtract two matrix and the dimension must be the same
     >>> m1 = Matrix1(3,3,lambda i ,j: 1 if i==j else 0)      # 生成两个相同的对角矩阵
    >>> m2 =  Matrix1(3,3,lambda i, j: 1 if i==j else 0)
    >>> print(m1.a)
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> print(m2.a)
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> t = matrix_sub(m1,m2)              # 将两个矩阵相加后得到新的矩阵，为之前的矩阵之和
    >>> t.a                                # 输出之后的矩阵
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    >>> m3 = Matrix1(1,1,lambda i, j: 1)
    >>> m3 = Matrix1(1,1,lambda i, j: 1)
    >>> matrix_sub(m1,m3)                   # 两个维度不同的矩阵要相加时
    please input the same dimensions!
    """
    if m1.shape() != m2.shape():
        print('please input the same dimensions!')
    else:
        m = Matrix1(m1.num_rows, m1.num_cols, lambda i, j: 0)
        m.a = [[m1.get_item(i, j) - m2.get_item(i, j)
                for j in range(m1.num_cols)]
               for i in range(m1.num_rows)]
        return m


def matrix_multi(m1: Matrix1, m2: Matrix1) -> Matrix1:
    """  multiply the two matrix
    >>> m1 = Matrix1(3,4,lambda i ,j: 1 if i==j else 0)
    >>> m2 = Matrix1(4,2,lambda i, j: 1 if i==j else 0)
    >>> m3 = matrix_multi(m1,m2)       # 两个矩阵相乘
    >>> m1.a
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    >>> m2.a
    [[1, 0], [0, 1], [0, 0], [0, 0]]
    >>> m3.a                     # 相乘后的矩阵的输出结果
    [[1, 0], [0, 1], [0, 0]]
    >>> m4 = Matrix1(2,2,lambda i, j: 0)
    >>> matrix_multi(m1,m4)
    please reset the two matrix!
    """
    if m1.num_cols != m2.num_rows:
        print('please reset the two matrix!')
    else:
        m = Matrix1(m1.num_rows, m2.num_cols, lambda i, j: 0)
        m.a = [[sum(m1.get_item(i1, j) * m2.get_item(j, i2)
                for j in range(m1.num_cols))
                for i2 in range(m2.num_cols)]
               for i1 in range(m1.num_rows)]
        return m


def matrix_transpose(m1: Matrix1) -> Matrix1:
    """   transposing the matrix
    >>> m1 = Matrix1(3,4,lambda i ,j: 1 if i==j else 0)
    >>> m1.a
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    >>> m = matrix_transpose(m1)              # 将m1转置
    >>> m.a
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
   """
    m = Matrix1(m1.num_cols, m1.num_rows, lambda i1, j1: 0)
    for i in range(m1.num_rows):
        for j in range(m1.num_cols):
            m.a[j][i] = m1.a[i][j]
    return m


if __name__ == "__main__":
    import doctest
    doctest.testmod()
