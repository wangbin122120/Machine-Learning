import W_KNN

# 程序1 ： knn示例程序 p22
# 使用Python导入数据
group, labels = W_KNN.createDataSet()
print(group, labels)

# 将数据进行计算
classfiy_result = W_KNN.classify0([0, 0], group, labels, 3)
print(classfiy_result)
