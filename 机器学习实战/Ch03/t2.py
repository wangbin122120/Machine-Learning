import W_tree
import treePlotter
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
print(lenses)
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=W_tree.createTree(lenses,lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)
