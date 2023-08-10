# https://blog.csdn.net/baidu_38127162/article/details/82380914
import time
import numpy as np
from matplotlib import pyplot as plt



#其他函数部分
# 权函数
def W_fun(d,x,X):
    s=abs(x-X)/d
    if (s<=0.5):
        return (2/3)-4*s**2+4*s**3
    elif(s<=1):
        return (4/3)-4*s+4*s**2-(4/3)*s**3
    else:
        return 0
# 权函数记号(pm,pn)的计算
def pm_pn(w,x,p,m,n):
    # x为数据点，w为支撑域的权重，M为数据点个数 p1,p2为传入的数值
    pmn=0
    M=len(x)
    # i代表数据点,m n代表(pm,pn)的下标
    for i in range(M):
        pmn=pmn+w[i]*p[i][m]*p[i][n]
    return float(pmn)
# B矩阵的建立
def fun_B(u,w,p):
    pumI=0
    M=len(u) #数据点个数
    m=p.shape[1] # 基函数个数
    B=[]
    for j in range(m):
        for i in range(M):
            pumI=pumI+w[i]*p[i][j]*u[i]
        B.append(float(pumI))
    return B
 # A矩阵的建立
def fun_A(x,w,p):
    M=len(x)
    m=p.shape[1]
    A=[]
    for mm in range(m):
        matA=[]
        for nn in range(m):
            pmn=pm_pn(w,x,p,mm,nn)
            matA.append(pmn)
        A.append(matA)
    return A



#主题部分
x=np.arange(-0.9,0.9,0.05)
# 数据点x个数
M=len(x)
# 基函数个数
N=2
p=np.zeros((M,2))
Y=[]
for XX in X:
    w = np.zeros((M,1))
    d=0.1 # 影响区域的半径
    for i in range(0,M):
        w[i]=W_fun(d, x[i], XX)
        p[i][0]=1
        p[i][1]=x[i]
    A=fun_A(x,w,p)
    B=fun_B(y,w,p)
    a=np.linalg.solve(A,B)
    Y.append(a[0]+a[1]*XX)

