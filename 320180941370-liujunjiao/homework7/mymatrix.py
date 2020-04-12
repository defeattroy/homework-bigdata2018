class Matrix:
    '''
    Define a class named "Matrix"
    "x" is the number of rows
    "y" is the number of columns
    "matrix" is list of this Matrix

    functions:
    __str__:
    __repr__:
    __getitem__:
    __setitem__:
    __add__:
    __sub__:
    __mul__:
    transpose:
    reshape:
    '''
    def __init__(self,args):#parameter is list
        '''
        >>> a=Matrix([[3,3,3],[3,5,5]])
        >>> a.x,a.y,a.matrix
        (2, 3, [[3, 3, 3], [3, 5, 5]])

        >>> b=Matrix([[1,1,2]])
        >>> b.x,b.y,b.matrix
        (1, 3, [[1, 1, 2]])
        '''
        self.x=0#Number of rows
        self.y=0#Number of columns
        self.matrix=args
        tem=[]#recover every columns of rows
        for i in args:#calculate the self.x
            self.x=self.x+1
            te=0
            for j in i:
                te=te+1
            tem.append(te)
        for x in tem:
            if x!=tem[0]:
                raise Exception("参数有误")
        self.y=te
    
    def __str__(self):
        return str(self.matrix)

    def __repr__(self):
        return self.__str__()
    
    
    def __getitem__(self,key):
        '''
        >>> a=Matrix([[3,3,3],[3,5,5]])
        >>> a[0]
        [3, 3, 3]
        '''
        return self.matrix[key]
    
    
    def __setitem__(self,key,value):
        '''
        >>> a=Matrix([[3,3,3],[3,5,5]]) 
        >>> a[0]=[1,1,1]
        >>> a
        [[1, 1, 1], [3, 5, 5]]
        '''
        self.matrix[key]=value

    def __add__(self,other):#rewrite "add"
        '''
        >>> a=Matrix([[3,3,3],[3,5,5]])
        >>> b=Matrix([[3,3,5],[8,8,8]])
        >>> a+b
        [[6, 6, 8], [11, 13, 13]]
        '''
        try:
            if self.x!=other.x & self.y!=other.y:#Only homomorphic matrix can be added
                raise Exception
            c=[]
            for i in range(self.x):
                d=[]
                for x,y in zip(self.matrix[i],other.matrix[i]):
                    sum=x+y
                    d.append(sum)
                c.append(d)
            r=Matrix(c)
            return r
        except:
            print("相加的矩阵不是同型矩阵！")
    
    def __sub__(self,other):#rewrite "sub"
        '''
        >>> a=Matrix([[3,3,3],[3,5,5]])
        >>> b=Matrix([[3,3,5],[8,8,8]])
        >>> a-b
        [[0, 0, -2], [-5, -3, -3]]
        '''
        try:
            if self.x!=other.x & self.y!=other.y:#Only homomorphic matrix can be subtracted
                raise Exception
            c=[]
            for i in range(self.x):
                d=[]
                for x,y in zip(self.matrix[i],other.matrix[i]):
                    sum=x-y
                    d.append(sum)
                c.append(d)
            r=Matrix(c)
            return r
        except:
            print("相减的矩阵不是同型矩阵！")
    
    def transpose(self):#matrix transpose
        '''
        >>> r=Matrix([[0,0,-2],[-5,-3,-3]])
        >>> r.transpose()
        [[0, -5], [0, -3], [-2, -3]]
        '''
        c=[]
        for y in range(self.y):
            d=[]
            for x in range(self.x):
                d.append(self.matrix[x][y])
            c.append(d)
        r=Matrix(c)
        return r

    def __mul__(self,other): #Multiplication of matrix,rewrite /brtweeen matrix and matrix
        '''
        >>> p=Matrix([[1,0,3,-1],[2,1,0,2]])
        >>> q=Matrix([[4,1,0],[-1,1,3],[2,0,1],[1,3,4]])
        >>> p*q
        [[9, -2, -1], [9, 9, 11]]
        '''
        try:
            if self.y!=other.x:
                raise Exception
            c=[]
            for x in range(self.x):
                d=[]
                for y in range(other.y):
                    t=0
                    for i in range(other.x):
                        t=t+self.matrix[x][i]*other.matrix[i][y]
                    d.append(t)
                c.append(d)
            r=Matrix(c)
            return(r)
        except:
            print("此两个矩阵不能相乘！")
    
    def reshape(self,new_x,new_y):#matrix changing shape
        '''
        >>> p=Matrix([[1,0,3,-1],[2,1,0,2]])
        >>> p.reshape(1,8)
        [[1, 0, 3, -1, 2, 1, 0, 2]]
        '''
        try:
            if self.x*self.y !=new_x*new_y:
                raise Exception
            c=[]
            d=[]
            tem=0
            for x in range(self.x):
                for y in range(self.y):
                    if tem<new_y:
                        d.append(self.matrix[x][y])
                        tem=tem+1
                    else:
                        c.append(d)
                        d=[]
                        d.append(self.matrix[x][y])
                        tem=1
            c.append(d)
            r=Matrix(c)
            return(r)
        except:
            print("无法改变！")

if __name__ == "__main__":  #
      import doctest
      doctest.testmod(verbose=True)
