
from numpy import *  # numpy 科学计算
import operator  # 运算符操作
''' knn算法：
特征值归一化
计算距离L2
取距离最近的k个点
对这k个点按特征值分组
返回人数最多的组名作为分类结果
http://blog.csdn.net/lu597203933/article/details/37969799

'''

def createDataSet():
    group = array([[1, 1.1],
                   [1, 1],
                   [0, 0],
                   [0, 0.1],
                   ])
    labels = ['a', 'a', 'b', 'b']
    return group, labels


# 使用k邻近算法将每组数据划分到某个类中：
def classify0(inX, dataSet, labels, k):
    # 对未知类别属性的数据集中的每个点依次执行以下操作：
    # (1)计算已知类别数据集中的点与当前点之间的距离；
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # @ 通过tile复制矩阵，dataSet 的shape=(4,2), inX=(2,) --tile--> (4,2)
    '''def tile(A, reps): 复制A 为 reps份，比如：
        >>> c = np.array([1,2,3,4])
        >>> np.tile(c,(4,1))
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])
    经过tile的调整后，inX就能和每个数据快速的计算结果。应该底层会做并行优化。
    '''
    sqDiffMat = diffMat ** 2  # sqDiffMat.shape =(4, 2) 经过每个点与各自点计算差的平方得到的(4, 2)
    sqDistances = sqDiffMat.sum(axis=1)  # 得到的shape=(4,),sum对shape中的第1列汇总，结果自然就是每个点各自相加平方和相加
    distances = sqDistances ** 0.5  # 结果是每个点的距离

    # (2)按照距离递增次序排序；
    sortedDistIndicies = distances.argsort()  # @ argsort() 是numpy库中的函数，返回的是数组值从小到大的索引值

    # (3)选取与当前点距离最小的k个点；
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 统计每个排序后距离对应的分类标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # @ dict.get(key,0)对于字典没有的键值，返回0值。

    # print(classCount)  # {'b': 2, 'a': 1} --.items()--> dict_items([('b', 2), ('a', 1)]) 是列表list。
    # (4)确定前k个点所在类别的出现频率；
    sortedClassCount = sorted(classCount.items(),  # 输入待排序数据，items()转换成 dict_items([('b', 2), ('a', 1)]) 返回类型其实就成了列表list
                              key=operator.itemgetter(1),  # 排序关键字是 上述 dict_items的第二列 ，每个分类的个数
                              reverse=True)

    # (5)返回前女个点出现频率最高的类别作为当前点的预测分类。
    return sortedClassCount[0][0]


# 自己练习一遍 def classify0(inX, dataSet, labels, k):
def w_classify0(x, data, labels, k):
    x_ = tile(x, (data.shape[0], 1))
    dis = sum((x_ - data) ** 2, axis=1) ** 0.5

    dis_sorted_index = dis.argsort()

    clas = {}
    for i in range(k):
        labels_ = labels[dis_sorted_index[i]]
        clas[labels_] = clas.get(labels_, 0) + 1

    return sorted(clas.items(),
                  key=operator.itemgetter(1),
                  reverse=True)[0][0]


# 程序2：约会网站的配对 ，p24
# 将文本文件的数据转换到numpy
def file2matrix(filename):
    with open(filename) as fr:
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector


# 对每个维度都进行归一化，以防止单个特征值值域过大对其他特征值造成误差
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 特征值相除，如果是要用矩阵相除，用numpy.linalg.solve(matA,matB)
    return normDataSet, ranges, minVals
