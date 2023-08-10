# https://www.jianshu.com/p/f0ed7d3a9900
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:39:26 2019

@author: yxh
"""
# Python实现MLS曲线拟合
import time
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    M=len(u) # 数据点个数
    m=p.shape[1] # 基函数个数
    B=[]
    for j in range(m):
        for i in range(M):
            pumI=pumI+w[i]*p[i][j]*u[i]
        B.append(float(pumI))
    return B

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



# path = './testMLS3.txt'
# pos = np.loadtxt(path)
# pos_rgb = pos[:, -1]
# idxs = np.where(pos_rgb == 1)
# lane = pos[idxs]
# x = lane[:, 0]
# y = lane[:, 1]
# z = lane[:, 2]
# lane = lane[lane[:, 0].argsort()]
# nums = 0
# num = np.uint8(lane.shape[0]/50) + 1
# lane_nums = np.zeros((num, 3))
# for i in range(lane.shape[0]):
#     if(i==0 or i%50==0):
#         lane_nums[nums, :] = lane[i, 0:3]
#         nums += 1
#
# X = lane_nums[:, 0]
# #Y = lane_nums[:, 1]
# Z = lane_nums[:, 2]
X=np.arange(np.min(x, axis = 0), np.max(x, axis = 0), 1.0)
# 数据点x个数
M=len(x)
# 基函数个数
N=6
p=np.zeros((M,N))
Y=[]
time_begin = time.time()
ids = 0
for XX in X:
    w = np.zeros((M,1))
    d=4.0 # 影响区域的半径
    for i in range(0,M):
        w[i]=W_fun(d, x[i], XX)
        p[i][0]=1
        p[i][1]=x[i]
        p[i][2]=z[i]
        p[i][3]=pow(x[i], 2)
        p[i][4]=x[i]*z[i]
        p[i][5]=pow(z[i], 2)
    A=fun_A(x,w,p)
    B=fun_B(y,w,p)
    print("len(A):",len(A), " len(B):", len(B))
    a=np.linalg.solve(A,B)
    Y.append(a[0]+a[1]*XX+a[2]*Z[ids]+a[3]*pow(XX, 2)+a[4]*XX*Z[ids]+a[5]*pow(Z[ids], 2))
    ids += 1
time_end = time.time()
print('time_cost:', time_end-time_begin)
Z = np.zeros((X.shape))
print(Z, Z.shape)
f1 = np.polyfit(X, Y, 2)   # Least squares polynomial fit. use 4 can get best fitting
p1 = np.poly1d(f1)
print( 'xishu:', f1)
print( 'fangcheng:', p1)
yval = p1(X)
final_pos = np.zeros((len(Y), 3))
final_pos[:, 0] = X
final_pos[:, 1] = yval
final_pos[:, 2] = np.mean(z, axis = 0)
#Z = final_pos[:, 2]
print( final_pos)
print( ">>>>>>>>>>>>>>>>>final_pos[0]:", final_pos[0])
print( ">>>>>>>>>>>>>>>>>final_pos[-1]:", final_pos[-1])
print( final_pos.shape)
"""
plot1 = plt.plot(X, Y, '-s', label='original values')
plot2 = plt.plot(X, yval, 'r', label='ployfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.title('ployfitting')
plt.show()
"""

fig = plt.Figure()
ax = plt.gca(projection='3d')
#ax3 = plt.axes(projection='3d')
ax.set_title('3D-curve')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
figure1 = ax.plot(X, Y, Z, c = 'r')  #red
figure2 = ax.plot(X, yval, Z)   #blue
plt.show()

#xx, yy = np.meshgrid(X, Y)
#print xx.shape

#zz = np.zeros((xx.shape))
#print zz.shape

#ax3.plot_surface(xx, yy, zz, cmap='rainbow')
#plt.show()