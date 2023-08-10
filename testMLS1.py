#from https://www.cnblogs.com/zzk0/p/10468502.html
import numpy as np
import random
from matplotlib import pyplot as plt

x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y = [0, 4, 5, 14, 15, 14.5, 14, 12, 10, 5, 4]

# 使用投影的方法来求解 一阶
def test1():
    A = np.empty((0, 2))
    B = np.empty((0, 1))
    for (a, b) in zip(x, y):
        row1 = np.array([a, 1])
        row2 = np.array([b])
        A = np.vstack([A, row1])
        B = np.vstack([B, row2])
        # print("a:", a, " b:", b,)
        # print("row1:", row1)
        # print("row2:", row2)
    print("A:", A)
    print("B:", B)
    # x′=(AT∗A)−1∗AT∗b https://www.cnblogs.com/zzk0/p/10468502.html
    ans = np.linalg.inv(np.mat(A.transpose()) * np.mat(A)) * np.mat(A.transpose()) * np.mat(B)
    xx = np.arange(0, 1, 0.01)
    yy = ans[0, 0] * xx + ans[1, 0]
    print("ans:", ans, ans[0, 0], ans[1, 0])
    plt.plot(xx, yy)
    plt.scatter(x, y, c='r')
    plt.show()

# 使用投影的方法来求解 二阶
def test2():
    A = np.empty((0, 3))
    B = np.empty((0, 1))
    for (a, b) in zip(x, y):
        row1 = np.array([a*a, a, 1])
        row2 = np.array([b])
        A = np.vstack([A, row1])
        B = np.vstack([B, row2])
    print("A:", A)
    print("B:", B)
    ans = np.linalg.inv(np.mat(A.transpose()) * np.mat(A)) * np.mat(A.transpose()) * np.mat(B)
    print(ans)
    xx = np.arange(0, 1, 0.01)
    yy = ans[0, 0] * xx**2 + ans[1, 0] * xx + ans[2, 0]
    print("ans:", ans, ans[0, 0], ans[1, 0], ans[2, 0])
    plt.plot(xx, yy)
    plt.scatter(x, y, c='r')
    plt.show()

# 使用求偏导的方法来求解
def test3():
    sumx = sumxx = sumxxx = sumxxxx = sumf = sumxf = sumxxf = 0
    for (a, b) in zip(x, y):
        sumx += a
        sumxx += a ** 2
        sumxxx += a ** 3
        sumxxxx += a ** 4
        sumf += b
        sumxf += a * b
        sumxxf += a * a * b
    A = np.array([[len(x), sumx, sumxx],
                  [sumx, sumxx, sumxxx],
                  [sumxx, sumxxx, sumxxxx]])
    B = np.array([sumf, sumxf, sumxxf])
    ans = np.linalg.solve(A, B)
    print(ans)
    xx = np.arange(0, 1, 0.01)
    yy = ans[0] + ans[1] * xx + ans[2] * xx**2
    plt.plot(xx, yy)
    plt.scatter(x, y, c='r')
    plt.show()



# 加权最小二乘法
def w(dis):
    dis = dis / 0.3
    if dis < 0:
        return 0
    elif dis <= 0.5:
        return 2/3 - 4 * dis**2 + 4 * dis**3
    elif dis <= 1:
        return 4/3 - 4 * dis + 4 * dis**2 - 4/3 * dis**3
    else:
        return 0

def mls(x_):
    sumxx = sumx = sumxf = sumf = sumw = 0
    for (a, b) in zip(x, y):
        weight = w(abs(x_ - a))
        sumw += weight
        sumx += a * weight
        sumxx += a * a * weight
        sumf += b * weight
        sumxf += a * b * weight
    A = np.array([[sumw, sumx],
                  [sumx, sumxx]])
    B = np.array([sumf, sumxf])
    ans = np.linalg.solve(A, B) # A*ans = B
    return ans[0] + ans[1] * x_

# 加权最小二乘法
def test4():
    # x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # y = [0, 4, 5, 14, 15, 14.5, 14, 12, 10, 5, 4]
    xx = np.arange(0, 1, 0.01)
    yy = [mls(xi) for xi in xx]
    plt.plot(xx, yy)
    plt.scatter(x, y, c='r')
    plt.show()


test4()