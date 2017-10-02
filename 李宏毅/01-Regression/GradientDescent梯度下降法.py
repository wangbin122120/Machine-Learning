# 梯度下降法的实现
# 2017-10-1 13:10:24 ，王斌 ，
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl  # 支持绘图时的中文显示

mpl.rcParams['font.sans-serif'] = ['SimHei']


def gradient_descent(LearningRate, x, y, ep=0.0001, max_iter=10000, LearningRate_SelfAdaption=None):
    x = np.column_stack((np.ones(x.shape[0]), x))  # 加上一列x0=1,方便矩阵计算。
    w = np.random.random(x.shape[1])  # 初始化w为随机数

    # 用矩阵方法计算的时候，要非常注意矩阵大小
    # print(x.shape, '\t', y.shape, '\t', w.shape)
    x = np.mat(x)
    y = np.mat(y).T  # np.mat(y) 得到的仅仅是(1,N)，转为(N,1)
    w = np.mat(w).T
    # print(x.shape, '\t', y.shape, '\t', w.shape)

    converged = False
    iter = 0
    loss = []
    if LearningRate_SelfAdaption == None:
        # 固定步长
        while not converged:
            iter += 1
            w = w - LearningRate * x.T * (x * w - y)  # 用矩阵计算，非常容易书写。
            if w.max() < ep: converged = True
            if iter == max_iter: converged = True
            loss.append(((x * w - y).T * (x * w - y) / 2).tolist()[0][0])
            if iter % 500 == 0:
                # print('#', iter, '\t,损失函数值为:', (x * w - y).T * (x * w - y) / 2, '\t,线性方程系数为：', w.T)
                pass

    elif LearningRate_SelfAdaption == 'Adagrad':
        # 这个 方法没有实现好，结果很一般，走得特别慢了。
        LSGD = np.zeros(x.shape[0])
        LearningRateR = np.tile(LearningRate, (x.shape[0], 1))
        while not converged:
            iter += 1
            GradientDescent = x.T * (x * w - y)

            # LSGD = [LSGD ** 2  + GD[0] ** 2 for LSGD, GD in zip(LSGD, GradientDescent.tolist())]
            # LearningRate = [LearningRate[i] / np.sqrt(LSGD[i] ) for i in range(len(LSGD))]
            LSGD = [LSGD ** 2 + GD[0] ** 2 for LSGD, GD in zip(LSGD, GradientDescent.tolist())]
            LearningRate = [LR / np.sqrt(LSGD) for LR, LSGD in zip(LearningRateR, LSGD)]
            # w = w - LearningRate * GradientDescent  # 用矩阵计算，非常容易书写。
            w = np.mat([w[0] - LR * GD[0] for w, LR, GD in zip(w.tolist(), LearningRate, GradientDescent.tolist())])

            if w.max() < ep: converged = True
            if iter == max_iter: converged = True
            loss.append(((x * w - y).T * (x * w - y) / 2).tolist()[0][0])
            if iter % 500 == 0:
                # print('#', iter, '\t,损失函数值为:', (x * w - y).T * (x * w - y) / 2, '\t,线性方程系数为：', w.T)
                pass
    else:
        # 最普通的自适应是除以一个迭代次数，使得步长越来越小
        while not converged:
            iter += 1
            LearningRate /= iter ** 0.5  # 最普通的自适应是除以一个迭代次数，使得步长越来越小
            w = w - LearningRate * x.T * (x * w - y)  # 用矩阵计算，非常容易书写。

            if w.max() < ep: converged = True
            if iter == max_iter: converged = True
            loss.append(((x * w - y).T * (x * w - y) / 2).tolist()[0][0])
            if iter % 500 == 0:
                # print('#', iter, '\t,损失函数值为:', (x * w - y).T * (x * w - y) / 2, '\t,线性方程系数为：', w.T)
                pass
    return w, loss


# y=2 * (x1) + (x2) + 3
LearningRate = 0.001
ep = 0.001
x_train = np.array([[1, 2], [2, 1], [2, 3], [3, 5], [1, 3], [4, 2], [7, 3], [4, 5], [11, 3], [8, 7]])
y_train = np.array([7, 8, 10, 14, 8, 13, 20, 16, 28, 26])
x_test = np.array([[1, 4], [2, 2], [2, 5], [5, 3], [1, 5], [4, 1]])


def func1():
    set_show_num = 50
    plt.axis([0, set_show_num, 0, 1500])  # 设置区间
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('梯度下降_不同学习步长的收敛情况')
    w, loss = gradient_descent(LearningRate, x_train, y_train, ep=0.0001, max_iter=5000)
    plt.plot(range(1, len(loss) + 1)[:set_show_num], loss[:set_show_num], label='LearningRate=%s' % str(LearningRate))

    # 对比不同的 LearningRate
    w, loss = gradient_descent(LearningRate / 10, x_train, y_train, ep=0.0001, max_iter=5000)
    plt.plot(range(1, len(loss) + 1)[:set_show_num], loss[:set_show_num],
             label='LearningRate=%s' % str(LearningRate / 10))

    w, loss = gradient_descent(LearningRate * 10, x_train, y_train, ep=0.0001, max_iter=5000)
    plt.plot(range(1, len(loss) + 1)[:set_show_num], loss[:set_show_num],
             label='LearningRate=%s' % str(LearningRate * 10))

    w, loss = gradient_descent(LearningRate * 5, x_train, y_train, ep=0.0001, max_iter=5000)
    plt.plot(range(1, len(loss) + 1)[:set_show_num], loss[:set_show_num],
             label='LearningRate=%s' % str(LearningRate * 5))

    plt.legend()  # 没有这句，label是不会显示的
    plt.show()
    '''输出结果： 和原来假设一致
    # 500 	,损失函数值为: [[ 0.7000368]] 	,线性方程系数为： [[ 2.12280763  2.03255968  1.17166714]]
    # 1000 	,损失函数值为: [[ 0.1214496]] 	,线性方程系数为： [[ 2.63463036  2.01356181  1.07150309]]
    # 1500 	,损失函数值为: [[ 0.02107033]] 	,线性方程系数为： [[ 2.84781562  2.00564879  1.02978259]]
    # 2000 	,损失函数值为: [[ 0.0036555]] 	,线性方程系数为： [[ 2.93661191  2.00235284  1.01240509]]
    # 2500 	,损失函数值为: [[ 0.00063419]] 	,线性方程系数为： [[ 2.97359748  2.00098001  1.00516699]]
    # 3000 	,损失函数值为: [[ 0.00011003]] 	,线性方程系数为： [[ 2.98900278  2.0004082   1.00215216]]
    # 3500 	,损失函数值为: [[  1.90885198e-05]] 	,线性方程系数为： [[ 2.99541942  2.00017002  1.00089642]]
    # 4000 	,损失函数值为: [[  3.31167327e-06]] 	,线性方程系数为： [[ 2.99809209  2.00007082  1.00037338]]
    # 4500 	,损失函数值为: [[  5.74543232e-07]] 	,线性方程系数为： [[ 2.99920531  2.0000295   1.00015552]]
    # 5000 	,损失函数值为: [[  9.96776852e-08]] 	,线性方程系数为： [[ 2.999669    2.00001229  1.00006478]]
    '''
    return w


def func2():
    set_show_num = 200
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('梯度下降_自动调整步长')
    w, loss = gradient_descent(LearningRate, x_train, y_train, ep=0.0001, max_iter=5000)
    plt.plot(range(1, len(loss) + 1)[:set_show_num], loss[:set_show_num], label='LearningRate=%s' % str(LearningRate))

    w, loss = gradient_descent(LearningRate, x_train, y_train, ep=0.0001, max_iter=5000, LearningRate_SelfAdaption=True)
    plt.plot(range(1, len(loss) + 1)[:set_show_num], loss[:set_show_num], label='LearningRate=SelfAdaption')

    w, loss = gradient_descent(LearningRate, x_train, y_train, ep=0.0001, max_iter=5000, LearningRate_SelfAdaption='Adagrad')
    plt.plot(range(1, len(loss) + 1)[:set_show_num], loss[:set_show_num], label='LearningRate=Adagrad')

    plt.legend()  # 没有这句，label是不会显示的
    plt.show()
    return w


def plot_data(x, y, x2=None, y2=None, marker1='o', marker2='^'):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], y.tolist(), marker=marker1)
    if x2 is not None and y2 is not None:
        ax.scatter(x2[:, 0], x2[:, 1], y2.tolist(), marker=marker2)
    plt.show()


def predict(w, x):
    x = np.column_stack((np.ones(x.shape[0]), x))  # 加上一列x0=1,方便矩阵计算。
    return x * w


def feature_scaling(x, norm=False):
    if norm == True:
        import math
        mean_ = np.mean(x)
        var_ = np.var(x)
        return (x - mean_) / var_
    else:
        min_ = np.min(x)
        range_ = np.max(x) - np.min(x)
        x = (x - min_) / range_
    return x


# ## 归一化
# x_train = feature_scaling(x_train, norm=True)
# y_train = feature_scaling(y_train, norm=True)

# func1()

w = func2()
y_predict = predict(w, x_test)
print(y_predict)

plot_data(x_test, y_predict, x_train, y_train)  # 把所有点的都放一起看看预测结果。
