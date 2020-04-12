class Matrix(object):
    # mat_shape = (row, col) is a tuple matrixs'size
    def __init__(self, mat_shape):
        self.cube = []
        self.row = mat_shape[0]
        self.col = mat_shape[1]
 
    def add_value(self):#create matrix
        temp = []
        for _ in range(self.row):
            for _ in range(self.col):
                temp.append(int(input('按行输入矩阵: ')))
 
            self.cube.append(temp)
            temp = []
 
    def show(self): #print matrix 
        print("the cube is :", self.cube)
 
    def multiply(self, other):#matrix multiply
        if self.col != other.row:
            print("矩阵1的行数与矩阵2的列数不相等，不能进行运算！")
        else:
            temp_result = []
            cube_result = []
            temp = []  #create empty list
            for a in range(self.row):
                for b in range(self.col):
                    for x in range(other.row):#user a there-layer loop
                        temp_result.append(self.cube[a][x] * other.cube[x][b])#Multiply the row elements of matrix one and the column elements of matrix other
                    temp.append(sum(temp_result))#Add the calculated results to get a new element
                    temp_result = []
                cube_result.append(temp)
                temp = []
            print(cube_result)


    def addition(self, other):#addition 
        if self.col != other.col&self.row != other.row:
            print("矩阵维度不相等，不能运算！")#Matrix dimensions must be equal to add or subtract
        else:
            k=[]#Create a new list to store the result matrix
            for i in range (self.row):
                for j in range(other.col):
                    k.append(self.cube[i][j] + other.cube[i][j])#Add the corresponding elements in rows and columns
            print (k)


    def subtraction(self, other):#Subtraction is the same as addition
        if self.col != other.col&self.row != other.row:
            print("矩阵维度不相等，不能运算！")
        else:
            k=[]
            for i in range (self.row):
                for j in range(other.col):
                    k.append(self.cube[i][j] - other.cube[i][j])
            print (k)

    def transpose(self) :#Turn the corresponding row element into a column element
        res = [] #col * []
        for i in range(self.row):
            temp = []#Create a new matrix to hold the transposed elements
            for j in range(self.col):
                temp.append(self.cube[j][i])
                res.append(temp)
        return res




if __name__=='__main__':
    import doctest
    doctest.testmod()

#Test code
mat1 = Matrix(mat_shape=(3, 3))
mat2 = Matrix(mat_shape=(3, 3))
print('矩阵1;')
mat1.add_value()
print('矩阵2:')
mat2.add_value()
mat1.show()
mat2.show()
print('结果为：')
mat1.multiply(mat2)
mat1.addition(mat2)
mat1.subtraction(mat2)
mat1.transpose()
