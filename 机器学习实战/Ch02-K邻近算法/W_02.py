import W_KNN


# 程序2：约会网站的配对 ，p24
def datingClassTest():
    datingDataMat, datingLabels = W_KNN.file2matrix('datingTestSet2.txt')
    print(datingDataMat, datingLabels)
    print(datingDataMat[0:20])

    normMat, ranges, minVals = W_KNN.autoNorm(datingDataMat)
    print('normMat', normMat, 'ranges', ranges, 'minVals', minVals)

    m = normMat.shape[0]
    hoRatio = 0.50
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = W_KNN.classify0(normMat[i, :],
                                           normMat[numTestVecs:m, :],
                                           datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total accuracy rate is: %f" % (1 - errorCount / float(numTestVecs)))


def W_datingClassTest():
    returnMat, classLabelVector = W_KNN.file2matrix('datingTestSet2.txt')
    normDataSet, ranges, minVals = W_KNN.autoNorm(returnMat)
    testNum = int(0.5 * len(normDataSet))
    err = 0
    for i in range(testNum):
        classResult = W_KNN.classify0(normDataSet[i],
                                      normDataSet[testNum:],
                                      classLabelVector[testNum:],
                                      k=3)
        if classLabelVector[i] != classResult:
            err += 1
    print('accuracy:', 1 - err / len(normDataSet))


def classifyPerson():
    pass


def W_classifyPerson():
    playGame = float(input('玩视频游戏所耗时间百分比'))
    flyM = float(input('毎年获得的飞行常客里程数'))
    icecream = float(input('毎周消费的冰淇淋公升数'))

    returnMat, classLabelVector = W_KNN.file2matrix('datingTestSet2.txt')
    normDataSet, ranges, minVals = W_KNN.autoNorm(returnMat)

    inX = ([playGame, flyM, icecream] - minVals) / ranges
    classResult = W_KNN.classify0(inX,
                                  normDataSet,
                                  classLabelVector,
                                  k=3)

    if classResult == 3:
        classResult = 'largeDoses'
    elif classResult == 2:
        classResult = 'smallDoses'
    elif classResult == 1:
        classResult = 'didntLike'
    else:
        print('error in classify')
    print('样本分类', classResult)


# #测试数据训练效果
# datingClassTest()
W_classifyPerson()

# # 数据集的可视化
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# # scatter 函数简介：http://blog.csdn.net/u013634684/article/details/49646311
# ax.scatter(datingDataMat[:, 1],  # 横坐标
#            datingDataMat[:, 2],  # 纵坐标
#            s=15.0 * np.array(datingLabels),  # 标量数据，可以理解为z轴，对应每个点对应数据,15表示点的粗细
#            c=15.0 * np.array(datingLabels),  # 颜色序列，以及依据分类数据
#            )
# # 设置标题
# ax.set_title('Scatter Plot')
# ax.axis([-2, 25, -0.2, 2.0])
# plt.xlabel('Percentage of Time Spent Playing Video Games')
# plt.ylabel('Liters of Ice Cream Consumed Per Week')
# plt.show()
