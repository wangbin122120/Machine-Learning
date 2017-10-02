import W_bayes
listOPosts,listClasses=W_bayes.loadDataSet()
print(listOPosts)
print(listClasses)

myVocabList=W_bayes.createVocabList(listOPosts)
print(myVocabList)

# 检查一个单词是否存在于文章中，将需要检查的单词作为输入，为每个单词构建特征（0-无，1-有）

print(W_bayes.setOfWords2Vec(myVocabList, # 待检查的单词或者向量
                             listOPosts[0]) ) # 目标文章

print(W_bayes.setOfWords2Vec(myVocabList,listOPosts[3]))

# 从词向量计算概率

trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(W_bayes.setOfWords2Vec(myVocabList,postinDoc))

print('trainMat',trainMat)
p0V,p1V,pAb=W_bayes.trainNB0(trainMat, # 文档矩阵
                             listClasses,
                             isLog=True) # 每篇文档类别标签组成的向量 ，# 1 is 不健康的, 0 not

print(pAb)
print(p0V)
print(p1V)


import feedparser