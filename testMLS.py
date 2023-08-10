# from https://blog.csdn.net/dreliveam/article/details/111666235

import numpy as np
import random
from matplotlib import pyplot as plt

Consistency = 3  #阶数
d = 0.9	#紧支域半径
# dotNumber = 36	#散点数量
dotNumber = 5	#散点数量
# dotDinstance = 0.05	#散点间距
dotDinstance = 0.2	#散点间距

test = 0	# 0 为曲线，1 为直线

x = []
v = []
Vp = []
p = np.zeros((dotNumber, Consistency)) # 36,3:阶数(1,x,x^2)
# M = []


def w(xi,xj,d): # 参考 https://blog.csdn.net/dreliveam/article/details/111666235 三次样条曲线
    # 权函数 https://wenku.baidu.com/view/fe7a74976f1aff00bed51eb1.html?_wkts_=1691573136260
    # 三次样条函数
    s = abs(xi-xj)/d
    # print("w s:", s)
    if(s <= 0.5):
        return (2/3)-4*s+4*s**2+4*s**3
    elif(s<=1):
        return (4/3)-4*s+4*s**2-(4/3)*s**3
    else:
        return 0

def b_compute(i, x, v, d):#i:点的次序,x是散点横坐标，v是散点纵坐标, d = 0.9紧支域半径
    b = []
    for c in range(Consistency):#Consistency = 3  #阶数
        t = 0
        for j in range(dotNumber):
            # if abs(x[i] - x[j]) < d: # 超过范围的，w是0，所以这里不是必须的
                t +=  p[j][c] * w(x[i], x[j], d) * v[j] # 基函数矩阵*权重*散点纵坐标
        b.append(float(t))
    return b
        # for j in range(0, Consistency):
        #     w()


def A_compute(i, x, d):## i:点的次序,x是散点横坐标，v是散点纵坐标, d = 0.9紧支域半径
    A = []
    for ci in range(Consistency):#Consistency = 3  #阶数
        vec = []
        for cj in range(Consistency):#Consistency = 3  #阶数
            t = 0
            for j in range(dotNumber):#36
                w1 = w(x[i], x[j], d) #0.6666666666666666, i 是某个点的index
                pcj = p[j][cj]#1.0 p:1.00000,-0.90000,0.81000
                pci = p[j][ci]#1.0
                t +=  w1 * pcj * pci# 权重*基函数矩阵？？？
            vec.append(float(t))
        A.append(vec)
    return A



def polySet(p,x): # 基函数矩阵，横坐标
    for i in range(dotNumber):#36
        for j in range(0, Consistency):#3
            p[i][j] = x[i]**(j)
            print("polySet i:",i, "j:",j, " x[i]:",x[i], " j:", j, " pij:",p[i][j]);


def Vcontruct(a,x,i):#a:{3},x:横坐标,i:横坐标index之一
    value = 0
    cnt = len(a) # 3
    print("Vcontruct a:",a, " x:", x, " i:", i)
    for c in range(cnt): # 3
        value +=  a[c] * x[i]** c
        print("Vcontruct c:", c , " i:", i, " x[i]:",x[i], " x[i]**c:" ,x[i]**c, " value:", value)
    return value

def main():
    x = np.arange(-0.5*dotNumber*dotDinstance, 0.5*dotNumber*dotDinstance, dotDinstance) # 横坐标

    for i in range(0, dotNumber):#36
        if test == 0: # 曲线
            # v.append(i + random.randint(-50,50)/10)
            v.append(i + random.randint(-50,50)/50)
            # v.append(random.randint(-50, 50) / 10)
        else: # 直线
            v.append(i + 0 / 10)

    polySet(p,x) # p 36*3
    print("x:",x)#
    print("v:",v)# 散点纵坐标
    print("p:",p)# 矩阵，多项式系数,二阶; 线性基函数,k(k=2)阶完备的多项式

    for i in range(dotNumber):# 36
        b = b_compute(i, x, v, d)# {list:3}可以理解为一个点对应的多个邻坐标权重平均后的纵坐标 [2.1856195701874563, 0.7183847736625579, -1.559917981252861]

        A = A_compute(i, x, d)# {list:3}{list:3}[[1.8703703703703678, -1.4077846364883386, 1.1442438271604927], [-1.4077846364883386, 1.1442438271604929, -0.9639741941015081], [1.1442438271604927, -0.9639741941015081, 0.8257673996913575]]
        a = np.linalg.solve(A, b)# a是Ax=b中的x {ndarray:(3,)} [ 14.1233183   -4.50527104 -26.71866532] 系数向量
        print("linalg.solve: A:",A)
        print("linalg.solve: b:",b)
        print("linalg.solve: a:",a)
        Vp.append(Vcontruct(a, x, i))# 拟合曲线纵坐标
        # V.append(a[0]+a[1]*x[i])
        print("拟合曲线参数----------->>>>>>>>>>>")
        print("i:", i)
        print("b:", b)
        print("A:", A)
        print("a:", a)

    plt.title("MLS")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x,v,color = "green", s= 10)
    plt.plot(x, Vp)
    print("Vp:", Vp)
    plt.show()

if __name__ =='__main__':
    main()


# numpy.linalg.lstsq() curve_fit()
# numpy.polyfit https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html

# https://blog.csdn.net/dreliveam/article/details/111666235
# https://wenku.baidu.com/view/fe7a74976f1aff00bed51eb1.html?_wkts_=1691573136260
# http://www.nealen.com/projects/mls/asapmls.pdf
# https://neuron.eng.wayne.edu/auth/ece512/lecture18.pdf
# https://math.mit.edu/icg/resources/teaching/18.085-spring2015/LeastSquares.pdf
# https://zhuanlan.zhihu.com/p/353282073
# https://blog.csdn.net/liumangmao1314/article/details/54179526
# https://www.jianshu.com/p/af0a4f71c05a

# 这篇最好
#https://blog.csdn.net/dreliveam/article/details/111666235
#n<=M? 对于一维，二阶，m=3,即n<=3？,不对,没有必要