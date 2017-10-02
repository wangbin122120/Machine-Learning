import svmMLiA
from time import time
a=time()
dataArr,labelArr=svmMLiA.loadDataSet('testSet.txt')
print(labelArr)
b,alphas = svmMLiA.smoSimple(dataArr,labelArr,0.6,0.001,40)
#b,alphas = svmMLiA.smoP(dataArr,labelArr,0.6,0.001,40)
print(b)
print(alphas[alphas>0])
import numpy as np
print(np.shape(alphas[alphas>0]))
for i in range(100):
    if alphas[i]>0.0:
        print(dataArr[i],labelArr[i])
print(time()-a)