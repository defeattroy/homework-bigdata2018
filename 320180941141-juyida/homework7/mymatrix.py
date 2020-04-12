import sys
print("初始化第一个矩阵")
long=int(input("请输入矩阵的列"))
wide=int(input("请输入矩阵的行"))
space=long*wide
fz1=1
a = [[0]*long for i in range(wide)]
print(a)
print(a[1][1])
choose1=int(input("请做出选择：\n1:矩阵由代码默认生成\n2:自行输入\n"))
if choose1==1:#确定矩阵大小并赋值
    for i in range(wide):
        for j in range(long):
            a[i][j]=int(fz1)
            fz1+=1
    print(a)

else:
    #a = np.empty([long,wide], dtype=int)#确定矩阵类型
    for i in range(wide):#通过循环自行赋值
        for j in range(long):
            a[i][j]=int(input())
            print(a)

print("初始化第二个矩阵")
long2=int(input("请输入矩阵的列"))
wide2=int(input("请输入矩阵的行"))
space=long2*wide2
fz1=1
b = [[0]*long2 for i in range(wide2)]
print(b)
print(b[1][1])
choose1=int(input("请做出选择：\n1:矩阵由代码默认生成\n2:自行输入\n"))
if choose1==1:#确定矩阵大小并赋值
    for i in range(wide2):
        for j in range(long2):
            b[i][j]=int(fz1)
            fz1+=1
    print(b)

else:
    #a = np.empty([long,wide], dtype=int)#确定矩阵类型
    for i in range(wide2):#通过循环自行赋值
        for j in range(long2):
            b[i][j]=int(input())
            print(b)

print("请选择矩阵间的运算关系")


choose2=int(input("1:加法运算\n2:减法运算\n3:数乘运算\n4:点乘运算\n5:矩阵转置"))

if choose2==1:
    c = [[0] * long for i in range(wide)]
    for i in range(long):
        for j in range(wide):
            c[j][i]=a[j][i]+b[j][i]
    print("注意：应满足加法基本规则")
    print(c)
if choose2==2:
    c = [[0] * long for i in range(wide)]
    choose3=int(input("1：a-b\n2:b-a"))
    if choose3==1:
        for i in range(long):
            for j in range(wide):
                c[j][i]=a[j][i]-b[j][i]
    else:
        for i in range(long):
            for j in range(wide):
                c[j][i]=b[j][i]-a[j][i]
    print(c)
if choose2==3:
    a3=int(input("请输入数乘的数字（数乘的对象默认为第一个）"))
    print(a3*a)
if choose2==4:
    if wide != long2:
        print("不满足矩阵的乘法条件")
    else:
        c = [[0] * long2 for i in range(long2)]
        i=j=k=l=0
        fz2=0
        for i in range(long2):
            for j in range(long2):
                for k in range(long):
                    #for l in range(wide2):
                    fz2+=int(a[i][k])*int(b[k][j])
                    print(a[i][k],b[k][j],fz2)
                c[i][j] =fz2
                print(c)
                fz2=0
    print('矩阵间的乘积为',c)
if choose2==5:
    c = [[0] * wide for i in range(long)]
    d = [[0] * wide2 for i in range(long2)]
    for i in range(long):
        for j in range(wide):
            c[i][j]=a[j][i]
    for i in range(long2):
        for j in range(wide2):
            d[i][j]=b[j][i]
    print(c)
    print(d)



if __name__ =='__main__':
    import doctest
    doctest.testmod()