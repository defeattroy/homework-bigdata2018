class matrix():

    def add(self,A,B):     #矩阵加法
        row=len(A)
        col=len(B[0])
        cos=len(B)
        cor=len(A[0])
        C=[[0]*col for i in range(row)]     #对新的矩阵进行初始化，行列均化为0
        if row==cos and col==cor:
            for i in range(row):
                for j in range(cor):
                    aft=A[i][j]+B[i][j]
                    C[i][j]+=aft
            print (C)
        else:
            print("A与B不能相加")

    def sub(self,A,B):     #矩阵减法
        row=len(A)
        col=len(B[0])
        cos=len(B)
        cor=len(A[0])
        C=[[0]*col for i in range(row)]
        if row==cos and col==cor:
            for i in range(row):
                for j in range(col):
                    aft=A[i][j]-B[i][j]
                    C[i][j]+=aft
            print (C)
        else:
            print("A与B不能相减")

    def sca(self,A,B):    #矩阵的乘法
        row=len(A)
        col=len(B[0])
        cos=len(B)
        cor=len(A[0])
        C=[[0]*row for i in range(col)]
        if row==col and cos==cor:     #先判断两个矩阵是否具备相乘的条件
            for i in range(row):
                for j in range(col):
                    for k in range(cos):
                        aft=A[i][k]*B[k][j]
                        C[i][j]+=aft
            print (C)
        else:
            print("AB不能相乘")

    def tran(self,A): #矩阵转置
        row=len(A)
        cor=len(A[0])
        C=[[0]*row for i in range(cor)]
        for i in range(row):
            for j in range(cor):
                aft = A[i][j]
                C[j][i] += aft
        print(C)

    def dot(self,A,B): #矩阵点积
        row=len(A)
        col=len(B[0])
        cos=len(B)
        cor=len(A[0])
        c=0
        if row==1 and col==1:     #先判断矩阵是否具备计算点积的条件
            if cos==cor:
                for i in range(row):
                    for j in range(cos):
                        c=c+A[i][j]*B[j][i]
                print(c)
            else:
                print("AB无法求点积")
        else:
            print("AB无法求点积")

    def det(self,A):   #矩阵求行列式的值
        row=len(A)
        cor=len(A[0])
        C=A[0][0]
        if row==cor and row>1:
            for i in range(1,row):
                times=A[i][0]/A[0][0]     #计算出前一行比下一行大的倍数
                for j in range(cor):
                    A[i][j]=A[i][j]-times*A[0][j]    #将矩阵部分行化为0
            t=[[0]*(row-1)for i in range(cor-1)]     #创建新的用于递归的矩阵
            for i in range(1,row):
                for j in range(1,cor):
                    t[i-1][j-1]=A[i][j]
            C*=self.det(t)     #运用递归计算值
        elif row!=cor:
            print("无法求行列式的值")
        return C


def main():
    A = [[1, 1, 1], [2, 0, 2]]
    B = [[1, 1, 1], [1, 5, 2],[1, 2, 1]]
    m=matrix()
    m.tran(A)   #矩阵的转置运算
    print(m.det(B))    #矩阵的求行列式值


    A = [[1, 1, 1], [2, 0, 2]]
    B = [[2, 3, 6], [5, 6, 8]]
    m=matrix()
    m.add(A,B)    #矩阵相加
    m.sub(A,B)    #矩阵相减


    A = [[2, 5, 6]]
    B = [[2], [4], [1]]
    m=matrix()
    m.dot(A,B)    #矩阵求点积运算


    A=[[1, 1, 1], [2, 3, 4]]
    B=[[2, 5],[1, 2],[3, 6]]
    m=matrix()
    m.sca(A,B)    #矩阵求乘积运算

if __name__=='__main__':
    main()