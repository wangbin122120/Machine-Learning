import W_tree
import treePlotter
myDat, labels = W_tree.createDataSet()
print(myDat)
myTree = W_tree.createTree(myDat,labels)
print(myTree)
treePlotter.createPlot(myTree)

print(W_tree.splitDataSet([[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']],0,1))