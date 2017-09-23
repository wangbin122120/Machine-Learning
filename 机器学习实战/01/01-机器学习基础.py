import numpy as np

# p15 numpy 中矩阵操作

# 通过random直接输出结果是数组
randArray = np.random.rand(4, 4)
# 通过 mat 可以将数组转为矩阵
randMat = np.mat(randArray)

print(randArray)
print(randMat)

print(type(randArray))  # <class 'numpy.ndarray'>
print(type(randMat))  # <class 'numpy.matrixlib.defmatrix.matrix'>

# 矩阵的逆
invRandMat = randMat.I

# 矩阵乘法
print(randMat * invRandMat)
''' 结果应该是单位矩阵：对角线上是1，其他都是0 
[[  1.00000000e+00   3.33066907e-16   0.00000000e+00  -2.22044605e-16]
 [  0.00000000e+00   1.00000000e+00  -1.11022302e-16  -4.44089210e-16]
 [  1.11022302e-16   1.11022302e-16   1.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   1.33226763e-15  -2.22044605e-16   1.00000000e+00]]
 '''

# eye(N, M=None, k=0, dtype=float) 创建单位矩阵
print(np.eye(4))

