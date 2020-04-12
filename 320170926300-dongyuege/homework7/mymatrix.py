import copy
class Matrix:
    '''Matrix is a two-dimensional of numbers.
    For matrix A[m][n], A has m rows and n columns while A[i][j] is the element in the ith row and the jth colunn.
    Matrix add/transpose/subtract/scalar
   '''
    def __init__(self, row,col,value=0):
        self.shape=(row,col)
        self.row=row
        self.col=col
        self._matrix=[[value for _ in range(row)] for _ in range(row)]
      
    def __getitem__(self,index):
        if isinstance(index,int):
            return self._matrix[index]
        elif isinstance(index,tuple):
            return self._matrix[index[0]][index[1]]

    def show(self):
        for r in range(self.row):
            for c in range(self.col):
                print(self[r+1,c+1],end=' ')

    def __setitem__(self,index,value):
        if isinstance(index,int):
            self._matrix[index]=copy.deepcopy(value)
        elif isinstance(index,tuple):
            self._matrix[index[0]][index[1]]=value
    
    def shape(self):
        i_length = len(self)
        j_length = len(self[0])
        return (i_length, j_length)
    
 
# 转置
def transpose(M):
    _tmp = zip(*M)
    return Matrix([list(_l) for _l in _tmp])
 
# 阵乘
def multiplyMatrix(M1, M2):
    M1 = Matrix(M1)
    M2 = Matrix(M2)
    m, x1 = M1.shape()
    x2, n = M2.shape()
    if x1 != x2:
        raise ValueError('Impossibal Multiplication for M(%s,%s) * M(%s,%s)'%(m,x1,x2,n))
    
    productF = lambda item: item[0]*item[1]
    M2 = transpose(M2)
    returnMatrix = []
    for l_list in M1:
        _tmpList = []
        returnMatrix.append(_tmpList)
        for r_list in M2:
            value = sum([ productF(item) for item in zip(l_list, r_list)])
            if abs(round(value) - value) < 0.00001: value = int(round(value))
            _tmpList.append(value)
    return Matrix(returnMatrix)
 
# 数乘
def multiplyNumber(N, M):
    _M = copy.deepcopy(M)
    _N = float(N)
    _M = [
        [ value*_N for value in _l ]
        for _l in _M
    ]
    return Matrix(_M)
 
# 上三角阵
def getUpperTriangularMatrix(M):
    # 杜尔里特算法（Doolittle algorithm）
    M = Matrix(M)
    m, n = M.shape()
    j_indexer = itertools.cycle(range(n))
    minusF = lambda rate, item: item[0] + (rate * item[1])
    for i in range(m):
        j = j_indexer.next()
        base = float(M[i][j])
        # 逢对角线元素为0，将本行压入最底
        _count = 0
        while base == 0 and _count<(m-i):
            _count += 1
            M.append(M.pop(i))
            base = float(M[i][j])
        if base == 0: continue
        # 初等行变化，将下三角设为0
        _i = i
        _j = j
        while _i < m-1:
            _i += 1
            _base = float(M[_i][_j])
            if _base == 0: continue
            rate = -(_base/base)
            M[_i] = [minusF(rate, item) for item in zip(M[_i], M[i])]
    return M
 
# 行列式
def getDeterminant(M):
    # 杜尔里特算法（Doolittle algorithm）
    M = Matrix(M)
    m, n = M.shape()
    if m != n:
        raise ValueError('Matrix(%s*%s) %s has NO Det.'%(m, n, M))
    j_indexer = itertools.cycle(range(n))
    minusF = lambda rate, item: item[0] + (rate * item[1])
    product = 1
    for i in range(m):
        j = j_indexer.next()
        base = float(M[i][j])
        _count = 0
        here_to_tail_span = m-i-1
        while base == 0 and _count<here_to_tail_span:
            product *= (-1)**here_to_tail_span
            _count += 1
            M.append(M.pop(i))
            base = float(M[i][j])
        if base == 0: return 0
        product *= base
        _i = i
        _j = j
        while _i < m-1:
            _i += 1
            _base = float(M[_i][_j])
            if _base == 0: continue
            rate = -(_base/base)
            M[_i] = [minusF(rate, item) for item in zip(M[_i], M[i])]
    return product
 
# 伴随矩阵
def getAdjugateMatrix(M):
    length = len(M)
    if length == 1: return [[1]]
    _returnM = []
    for i in range(length):
        _tmpList = []
        _returnM.append(_tmpList)
        for j in range(length):
            _M = copy.deepcopy(M)
            _M = [
                _l for _l in _M
                if _M.index(_l) != i
            ]
            [ _l.pop(j) for _l in _M ]
            _Determinant = getDeterminant(_M)
            _tmpList.append(((-1)**(i+j))*_Determinant)
    return Matrix(transpose(_returnM))
 
# 逆矩阵
def getInversedMatrix(M):
    # A* / |A|
    _Determinant = getDeterminant(M)
    if _Determinant == 0:
        raise ZeroDivisionError('%s is NOT Non-Singular, has NO InversedMatrix'%str(M))
    k = 1.0/_Determinant
    _AdjugateMatrix = getAdjugateMatrix(M)
    _returnM = multiplyNumber(k, _AdjugateMatrix)
    return _returnM

if __name__=="__main__":
    m=Matrix(3,3,0)
    n=Matrix(3,3,0)
    print(m[1,1])


         
