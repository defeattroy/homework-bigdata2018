import copy

class Matrix:
    '''matrix class'''
    def __init__(self, row, column, fill =0):
        self.shape = (row, column)
        self.row = row
        self.column = column
        if isinstance(fill,float) or isinstance(fill,int):
            self._matrix = [[fill]*column for i in range(row)]
        elif isinstance(fill,list):
            self._matrix = fill
              
    # 返回元素m(i,j)的值：m[i,j]
    def __getitem__(self,index):
        if isinstance(index,int):
            return self._matrix[index]
        elif isinstance(index,tuple):
            return self._matrix[index[0]][index[1]]
           
    #设置元素m(i,j)的值为s：m[i,j] = s
    def __setitem__(self,index,value):
        if isinstance(index,int):
            self._matrix[index] = copy.deepcopy(value)
        elif isinstance(index,tuple):
            self._matrix[index[0]][index[1]] = value
         
    def __eq__(self,N):
        assert isinstance(N,Matrix), "类型不匹配，不能比较"
        return N.shape == self.shape
        
    def __add__(self,N):
        assert N.shape == self.shape, "维度不匹配，不能相加"
        M = Matrix(self.row,self.column)
        for r in range(self.row):
            for c in range(self.column):
                M[r,c] = self[r,c]+N[r,c]
        return M
        
    def __sub__(self,N):
        assert N.shape == self.shape, "维度不匹配，不能相减"
        M = Matrix(self.row,self.column)
        for r in range(self.row):
            for c in range(self.column):
                M[r,c] = self[r,c]-N[r,c]
        return M
        
    def __mul__(self,N):
        if isinstance(N,int) or isinstance(N,float):
            M = Matrix(self.row,self.column)
            for r in range(self.row):
                for c in range(self.column):
                    M[r,c] = self[r,c]*N
        else:
            assert N.row == self.column, "维度不匹配，不能相乘"
            M = Matrix(self.row,N.column)
            for r in range(self.row):
                for c in range(N.column):
                    sum = 0
                    for k in range(self.column):
                        sum +=self[r,k]*N[k,c]
                    M[r,c] = sum
        return M
        
    def __pow__(self,k):
        assert self.row == self.column,"不是方阵，不能乘方"
        M = copy.deepcopy(self)
        for i in range(k):
            M = M*self
        return M
        
    #矩阵转置
    def transpose(self):
        M = Matrix(self.column,self.row)
        for r in range(self.row):
            for c in range(self.column):
                M[c,r] = self[r,c]
        return M  
        
    #逆矩阵
    def invert(self):
        assert self.row == self.column,"不是方阵"
        M = Matrix(self.row,self.column*2)
        I = identity(self.row)
        #I.show()
        #原有矩阵拼接一个单位矩阵
        for r in range(M.row):
            temp = self[r]
            temp.extend(I[r])
            M[r] = copy.deepcopy(temp)
        # M.show()
        
        #初等行变换
        for r in range(M.row):
            # 本行首元素(M[r, r])若为 0，则向下交换最近的当前列元素非零的行
            if M[r,r] == 0:
                for rr in range(r+1,M.row):
                    if M[rr,r] != 0:
                        M[r],M[rr] = M[rr],M[r] #交换两行
                    break
            assert M[r,r] != 0,"矩阵不可逆"
            
            # 本行首元素化为1
            temp = M[r,r]
            for c in range(r,M.column):
                M[r,c] /=temp
            # M.show()
            
            # 本列上、下方所有元素的处理
            for rr in range(M.row):
                temp = M[rr,r]
                for c in range(M.column):
                    if rr == r:
                        continue
                    M[rr,c] -=temp*M[r,c]
                # M.show()
                
        #截取逆矩阵
        N = Matrix(self.row,self.column)
        for r in range(self.row):
            N[r] = M[r][self.row:]
        return N
        
    
    def show(self):
        for r in range(self.row):
            for c in range(self.column):
                print self[r,c] + ' '
            print ''   
        
       
#生成单位矩阵
def identity(num):
    M = Matrix(num,num)
    for r in range(num):
        for c in range(num):
            M[r,c] = 1 if r==c else 0
    return M  
    
#将数组转换为矩阵
def mat(arr):
    assert isinstance(arr,list),"参数不是数组，不能转换"
    if isinstance(arr[0],list):
        row = len(arr)
        column = len(arr[0])
        return Matrix(row,column,arr)
    else:
        row = 1
        column = len(arr)
        m = Matrix(row,column,0)
        for r in range(m.row):
            for c in range(m.column):
                m[r,c] = arr[c]
        return m
        
#计算矩阵的行列式(m为二维数组)
def det(m):
    if len(m) <= 0:
        return None
    elif len(m) == 1:
        return m[0][0]
    else:
        s = 0
        for i in range(len(m)):
            # 这里生成余子式
            n = [[row[a] for a in range(len(m)) if a != i] for row in m[1:]]
            s += m[0][i] * det(n) * (-1) ** (i % 2)
        return s
