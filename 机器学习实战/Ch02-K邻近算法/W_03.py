# 程序3.手写识别
# 和原文程序使用的输入数据集不同，但是类似的，只是原文的输入方面太繁琐了。
import W_KNN
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)
train_xs, train_ys = mnist.train.next_batch(50000)
test_xs, test_ys = mnist.test.next_batch(1000)

# 分类结果要转换成数字或者字符串，因为在classify0() 中的分类结果字典中key值要能接受。
train_ys = np.argmax(train_ys, axis=1)
test_ys = np.argmax(test_ys, axis=1)

err = 0
for i in range(len(test_xs)):
    classResult = W_KNN.classify0(test_xs[i],
                                  train_xs,
                                  train_ys,
                                  k=3)
    if test_ys[i] != classResult:
        err += 1
print('accuracy:', 1 - err / len(test_xs))
# 训练集 1000 accuracy: 0.885
# 训练集 10000 accuracy: 0.945
# 训练集 20000 accuracy: 0.954
# 训练集 50000 accuracy: 0.966
