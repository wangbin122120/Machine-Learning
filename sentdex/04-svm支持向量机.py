from time import time,sleep

'''
svm ( Support Vector Machine) The objective of the Support Vector Machine is to find the best splitting boundary between data 支持向量机的目标是寻找数据之间的最佳分裂边界。 在60-90年代都是最好的机器学习算法。

    # 分类数据是机器学习中的一项常见任务。 假设某些给定的数据点各自属于两个类之一，而目标是确定新数据点将在哪个类中。对于支持向量机来说，数据点被视为 p 维向量，而我们想知道是否可以用 (p-1) 维超平面来分开这些点。
    这就是所谓的线性分类器。可能有许多超平面可以把数据分类。最佳超平面的一个合理选择是以最大间隔把两个类分开的超平面。因此，我们要选择能够让到每边最近的数据点的距离最大化的超平面。如果存在这样的超平面，则称为最大间隔超平面，而其定义的线性分类器被称为最大间隔分类器，或者叫做最佳稳定性感知器。


#p20.关于SVM的概念以及简单的用法（调用sklearn.svm.svc()判断癌症的分类）
    https://pythonprogramming.net/support-vector-machine-intro-machine-learning-tutorial/

#p21.关于vector向量的概念，以及向量点乘的计算。
    https://pythonprogramming.net/vector-basics-machine-learning-tutorial/

    另外，《关于点乘和叉乘》
        点乘，也叫数量积。结果是一个向量在另一个向量方向上投影的长度，是一个标量。叉乘，也叫向量积。结果是一个和已有两个向量都垂直的向量。以我比较熟悉的图形学而言，一般点乘用来判断两个向量是否垂直，因为比较好算。也可以用来计算一个向量在某个方向上的投影长度，就像定义一样。
        叉乘更多的是判断某个平面的方向。从这个平面上选两个不共线的向量，叉乘的结果就是这个平面的法向量。两种不同的运算而已。
        更详细的介绍见：http://blog.csdn.net/dcrmg/article/details/52416832

#p22.Support Vector支持向量的概念和二维平面中支持向量的计算。    
https://pythonprogramming.net/support-vector-assertions-machine-learning-tutorial/ 


#p23.Support Vector Machine SVM的原理。 算法的原理是非常重要的，理解原理才能知道这个算法适用于什么场景。
https://pythonprogramming.net/support-vector-machine-fundamentals-machine-learning-tutorial/?completed=/support-vector-assertions-machine-learning-tutorial/

    关于svm更多的解释看wiki:https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA

#p24.Support Vector Machine Optimization SVM优化求解。
https://pythonprogramming.net/svm-constraint-optimization-machine-learning-tutorial/?completed=/support-vector-machine-fundamentals-machine-learning-tutorial/


#p25. 从头开始写SVM算法程序，Creating an SVM from scratch - Practical Machine Learning
https://pythonprogramming.net/svm-in-python-machine-learning-tutorial/?completed=/svm-constraint-optimization-machine-learning-tutorial/

#p26. 支持向量机的优化算法
https://pythonprogramming.net/svm-optimization-python-machine-learning-tutorial/?completed=/svm-in-python-machine-learning-tutorial/

附带资源： 

    Convex Optimization Book: 神作700页，凸优化，看一眼就两眼一黑，一亿点伤害，知道自己多么的low！ 
    https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
    
    SMO,Sequential Minimal Optimization book: 20多页论文，序列最小优化，懵逼^2！
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/smo-book.pdf
    
    还是SMO,20多页论文，懵逼^3 
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
    
    python的凸优化，CVXOPT (Convex Optimization Module for Python): 
    http://cvxopt.org/


#27. 支持向量机的优化算法2
https://pythonprogramming.net/svm-optimization-python-2-machine-learning-tutorial/?completed=/svm-optimization-python-machine-learning-tutorial/

#28.用自定义的svm实现可视化和预测。
Visualization and Predicting with our Custom SVM
https://pythonprogramming.net/predictions-svm-machine-learning-tutorial/?completed=/svm-optimization-python-2-machine-learning-tutorial/


最后，关于svm的应用（来源wiki=svm） 
    用于文本和超文本的分类，在归纳和直推方法中都可以显著减少所需要的有类标的样本数。
    用于图像分类。实验结果显示：在经过三到四轮相关反馈之后，比起传统的查询优化方案，支持向量机能够获取明显更高的搜索准确度。这同样也适用于图像分区系统，比如使用Vapnik所建议的使用特权方法的修改版本SVM的那些图像分区系统。[4][5]
    用于手写字体识别。
    用于医学中分类蛋白质，超过90%的化合物能够被正确分类。基于支持向量机权重的置换测试已被建议作为一种机制，用于解释的支持向量机模型。[6][7] 支持向量机权重也被用来解释过去的SVM模型。[8] 为识别模型用于进行预测的特征而对支持向量机模型做出事后解释是在生物科学中具有特殊意义的相对较新的研究领域。
        
'''

def sklearn_svm():
    import pandas as pd
    import numpy as np  # 用于将 get到的数据 序列化成 numpy.
    from sklearn import preprocessing  # 数据标准化，归一化 ，参考：http://blog.csdn.net/dream_angel_z/article/details/49406573
    # from sklearn import neighbors  # 最临近算法
    from sklearn import svm       # svm 算法  ，使用svm和 knn方法一样简单。
    from sklearn.model_selection import train_test_split  # 训练集和测试集的划分。


    df = pd.read_csv('data/classification/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)  # 第一列id是无用干扰信息，必须抛去，否则结果非常差。

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # clf = neighbors.KNeighborsClassifier()
    clf = svm.SVC()  # 用sklearn计算很方便，修改一下接口方法即可。

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)  # svm:0.95 , knn :0.978571428571 ，如果没有取出’id' ，结果只有0.5，完全就是抛硬币。所以在数据初始处理的时候，非常重要。
    # svm 算法速度比knn快几倍，效果也很好。

    # 数据预测
    example_measures = np.array([[5, 5, 1, 6, 3, 5, 3, 5, 5],
                                 [2, 1, 1, 1, 1, 2, 3, 1, 1]])
    print(clf.predict(example_measures))  # [4 2]


# 25 -26，
def self_svm():
    import matplotlib.pyplot as plt
    from matplotlib import style
    import numpy as np
    style.use('ggplot')

    class Support_Vector_Machine:
        def __init__(self, visualization=True):
            self.visualization = visualization
            self.colors = {1: 'r', -1: 'b'}
            if self.visualization:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(1, 1, 1)

        # train
        def fit(self, data):
            self.data = data
            # { ||w||: [w,b] }
            opt_dict = {}

            transforms = [[1, 1],
                          [-1, 1],
                          [-1, -1],
                          [1, -1]]

            all_data = []
            for yi in self.data:
                for feature_set in self.data[yi]:
                    for feature in feature_set:
                        all_data.append(feature)

            self.max_feature_value = max(all_data)
            self.min_feature_value = min(all_data)
            all_data = None

            # support vectors yi(xi.w+b) = 1


            step_sizes = [self.max_feature_value * 0.1,
                          self.max_feature_value * 0.01,
                          # point of expense:
                          self.max_feature_value * 0.001,
                          ]

            # extremely expensive
            b_range_multiple = 2
            # we dont need to take as small of steps
            # with b as we do w
            b_multiple = 5
            latest_optimum = self.max_feature_value * 10

            for step in step_sizes:
                w = np.array([latest_optimum, latest_optimum])
                # we can do this because convex
                optimized = False
                while not optimized:
                    for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                       self.max_feature_value * b_range_multiple,
                                       step * b_multiple):
                        for transformation in transforms:
                            w_t = w * transformation
                            found_option = True
                            # weakest link in the SVM fundamentally
                            # SMO attempts to fix this a bit
                            # yi(xi.w+b) >= 1
                            #
                            # #### add a break here later..
                            for i in self.data:
                                for xi in self.data[i]:
                                    yi = i
                                    if not yi * (np.dot(w_t, xi) + b) >= 1:
                                        found_option = False
                                        # print(xi,':',yi*(np.dot(w_t,xi)+b))

                            if found_option:
                                opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                    if w[0] < 0:
                        optimized = True
                        print('Optimized a step.')
                    else:
                        w = w - step

                norms = sorted([n for n in opt_dict])
                # ||w|| : [w,b]
                opt_choice = opt_dict[norms[0]]
                self.w = opt_choice[0]
                self.b = opt_choice[1]
                latest_optimum = opt_choice[0][0] + step * 2

            for i in self.data:
                for xi in self.data[i]:
                    yi = i
                    print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

        def predict(self, features):
            # sign( x.w+b )
            classification = np.sign(np.dot(np.array(features), self.w) + self.b)
            if classification != 0 and self.visualization:
                self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
            return classification

        def visualize(self):
            [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

            # hyperplane = x.w+b
            # v = x.w+b
            # psv = 1
            # nsv = -1
            # dec = 0
            def hyperplane(x, w, b, v):
                return (-w[0] * x - b + v) / w[1]

            datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
            hyp_x_min = datarange[0]
            hyp_x_max = datarange[1]

            # (w.x+b) = 1
            # positive support vector hyperplane
            psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
            psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
            self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

            # (w.x+b) = -1
            # negative support vector hyperplane
            nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
            nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
            self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

            # (w.x+b) = 0
            # positive support vector hyperplane
            db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
            db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
            self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

            plt.show()

    data_dict = {-1: np.array([[1, 7],
                               [2, 8],
                               [3, 8], ]),

                 1: np.array([[5, 1],
                              [6, -1],
                              [7, 3], ])}

    svm = Support_Vector_Machine()
    svm.fit(data=data_dict)

    predict_us = [[0, 10],
                  [1, 3],
                  [3, 4],
                  [3, 5],
                  [5, 5],
                  [5, 6],
                  [6, -5],
                  [5, 8]]

    for p in predict_us:
        svm.predict(p)

    svm.visualize()

# 用sklearn 自带方法 svm.svc() 进行训练和预测 癌症数据。
# sklearn_svm()

# 用自己写的程序进行训练和预测。
self_svm()