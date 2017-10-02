# http://blog.csdn.net/lu597203933/article/details/38024239

from math import log
##### 计算香农熵
def calcShannonEnt(dataSet):
    # 首先 计算数据集中实例的总数。我们也可以在需要时再计算这个值，但是由于代码中多次用到这个值，为了提高代码效率，我们显式地声明一个变量保存实例总数。
    numEntries = len(dataSet)
    # 然后，创建一个数据字典，它的键值是最后一列的数值0。如果当前键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 最后，使用所有类标签的发生频率计算类别出现的概率。我们将用这个概率计算香农熵©，统计所有类标签发生的次数。
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt

# 计算香农熵-练习版
def W_calcShannonEnt(dataSet):
    labels = {}
    for featVec in dataSet:
        labels[featVec] = labels.get(featVec, 0) + 1

    numEntries = len(dataSet)
    shannon = 0.0
    for k in labels:
        prob = float(labels[k]) / numEntries
        shannon -= prob * log(prob, 2)
    return shannon

# 创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels

# 划分数据集 传入参数：待划分的数据集 划分数据集的特征 特征的返回值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

##### 寻找最好的特征划分方式,在分类问题中，我们希望获得的信息增益越大越好，也就是熵值变得最小.熵变大，数据变得无序分散；熵变小，数据变得有序集中
def chooseBestFeatureToSplit(dataSet):
    #第3行代码计算了整个数据集的原始香农熵，我们保存最初的无序度量值,.用于与划分完之后的数据集计算的嫡值进行比较。
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    print('\n\n原始数据集：',dataSet)
    baseEntropy = calcShannonEnt(dataSet)
    print('基准熵值：',baseEntropy)
    bestInfoGain = 0.0; bestFeature = -1
    # 第 1个for 循环遍历数据集中的所有特征。使用列表推导（List Comprehension)来创建新的列表，将数据集中所有第i个特征值或者所有可能存在的值写人这个新list中 。
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        #然后使用python 语言原生的集合set数据类型。从列表中创建集合是python得到列表中唯一元素值的最快方法。
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值，对每个特征划分一次数据集，然后计算数据集的新熵值 ，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            # 信息增益是熵的减少或者是数据无序度的减少，大家肯定对于将熵用于度量数据无序度的减少更容易理解。
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            print('特征值', i, 'value', value,'数据集', subDataSet, 'prob',prob,'新熵值',newEntropy)
        # 注意了，我们这里比较的是那种分类能让熵值减少的最多(信息增益)，也就是数据越有序，越集中。
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        print( '特征值', i,'信息增益：', infoGain)
        #最后，比较所有特征中的信息增益，返回最好特征划分的索引值0。
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

###### 其中当所有的特征都用完时，采用多数表决的方法来决定该叶子节点的分类，即该叶节点中属于某一类最多的样本数，那么我们就说该叶节点属于那一类！。代码如下：
import operator
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建树的代码
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        print('##### ',dataSet,classList,len(classList),classList[0])
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        print('@@@@@',dataSet)
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        print('###@@@####',subLabels,dataSet,bestFeat,value)
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 临时保存决策树结构
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]

    # createPlot(thisTree)

# 应用决策树对输入数据进行判断，并返回分类结果
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


# 使用pickle模块保存决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


# 使用pickle模块读取决策树
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
