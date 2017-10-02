import W_tree

# 导入数据
myDat, labels = W_tree.createDataSet()
print('原始数据集：',myDat,'\n标签：',labels)

# 计算信息熵
print('原始数据集的熵值：',W_tree.calcShannonEnt(myDat))

# 利用特征值进行数据集划分
print('第0位特征值=1时的分支（原始数据有重复）',W_tree.splitDataSet(myDat, 0, 1))  # [[1, 'yes'], [1, 'yes'], [0, 'no']]

# 寻找最佳数据集划分的特征值
print('-----------计算划分数据集的最佳特征值')
print('-----------最佳特征值：',W_tree.chooseBestFeatureToSplit(myDat)) # 0  -这个0表示，第0个特征是最好的用于划分数据集的特征
''' 验证结果的效果
基准熵值： 0.9709505944546686
特征值 0 value 0 数据集 [[1, 'no'], [1, 'no']] prob 0.4 新熵值 0.0
特征值 0 value 1 数据集 [[1, 'yes'], [1, 'yes'], [0, 'no']] prob 0.6 新熵值 0.5509775004326937
特征值 0 信息增益： 0.4199730940219749
特征值 1 value 0 数据集 [[1, 'no']] prob 0.2 新熵值 0.0
特征值 1 value 1 数据集 [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']] prob 0.8 新熵值 0.8
特征值 1 信息增益： 0.17095059445466854
最佳特征值： 0
'''


# 创建决策树
import W_tree
print('-----------计算最佳数据集划分方式及其决策树')
myDat, labels = W_tree.createDataSet()
myTree = W_tree.createTree(myDat,labels)
print('-----------最佳数据集划分方式及其决策树：',myTree)  # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
'''
原始数据集： [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
基准熵值： 0.9709505944546686
特征值 0 value 0 数据集 [[1, 'no'], [1, 'no']] prob 0.4 新熵值 0.0
特征值 0 value 1 数据集 [[1, 'yes'], [1, 'yes'], [0, 'no']] prob 0.6 新熵值 0.5509775004326937
特征值 0 信息增益： 0.4199730940219749
特征值 1 value 0 数据集 [[1, 'no']] prob 0.2 新熵值 0.0
特征值 1 value 1 数据集 [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']] prob 0.8 新熵值 0.8
特征值 1 信息增益： 0.17095059445466854


原始数据集： [[1, 'yes'], [1, 'yes'], [0, 'no']]
基准熵值： 0.9182958340544896
特征值 0 value 0 数据集 [['no']] prob 0.3333333333333333 新熵值 0.0
特征值 0 value 1 数据集 [['yes'], ['yes']] prob 0.6666666666666666 新熵值 0.0
特征值 0 信息增益： 0.9182958340544896
{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
'''

#### 绘制决策树
print('绘制一个简单的决策树模型')
import matplotlib.pyplot as plt
#定义决策节点和叶子节点的风格
decisionNode = dict(boxstyle = "sawtooth",fc="0.8")
#boxstyle = "swatooth"意思是注解框的边缘是波浪线型的，fc控制的注解框内的颜色深度
leafNode = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")#箭头符号
"""
@brief 绘制节点
@param[in] nodeTxt 节点显示文本
@param[in] centerPt 起点位置
@param[in] parentPt 终点位置
@param[in] nodeType 节点风格
"""
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
                            xytext=centerPt,textcoords='axes fraction',\
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False) #绘制子图
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

# 绘制树型的示例程序
createPlot()

# 统计叶子个数和树的层数

import treePlotter

myTree = treePlotter.retrieveTree(0) # 直接从上面的结果导入树结构
print('叶子节点数：', treePlotter.getNumLeafs(myTree)) # 统计叶子节点个数，以计算x轴长度
print('树的层数：', treePlotter.getTreeDepth(myTree)) # 统计树的层数，以计算y轴高度


# 绘制完整的决策树模型
print('绘制完整的决策树模型')
treePlotter.createPlot(myTree)


### 应用数据构造决策树并用于预测
import W_tree
import treePlotter
myDat, labels = W_tree.createDataSet()
myTree = treePlotter.retrieveTree(0)

print('[1,0]的分类结果是', W_tree.classify(myTree, labels, [1, 0]))  # [1,0]的分类结果是 no
print('[1,1]的分类结果是', W_tree.classify(myTree, labels, [1, 1]))  # [1,1]的分类结果是 yes

# 要是每次预测的时候都要重新计算决策树显然不方便，于是我们要将决策树保存起来，随时都可以用
print('保存决策树',myTree)
W_tree.storeTree(myTree,'classifierStorage.txt')
myTree=W_tree.grabTree('classifierStorage.txt')
print('提取决策树',myTree)



# 隐形眼镜的应用
import W_tree
import treePlotter
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
print(lenses)
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=W_tree.createTree(lenses,lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)













