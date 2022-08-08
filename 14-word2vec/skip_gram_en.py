#!/usr/bin/endimension python
# -#-coding:utf-8 -*-
# author:by ucas iie 魏兴源
# datetime:2021/10/21 11:24:08
# software:PyCharm


"""
    1.下载数据or找个实验数据,进行数据预处理
    2.将原词汇数据转换为字典映射，求三大参数，即word2index,index2word,word2one-hot
    3.为 skip-gram模型 建立一个扫描器
    4.建立并训练 skip-gram 模型
    5.开始训练模型
    6.结果可视化
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm, trange
import nltk

nltk.download('punkt')

# 这里换成对应的路径
with open("data/enDemoTest2", errors='ignore') as f:
    Data = f.read()
    print(f)
    print(Data[0:10])
    print("Data length=", len(Data))

# 把所有字母转成小写字母
Data = Data.lower()
# 去除字符中的所有数字
Data = re.sub(r"\b\d+\b", '', Data)
# 去除所有的特殊字符，只保留字母
Data = re.sub(r'[^A-Za-z0-9 ]+', '', Data)

"""
    word_tokenize()
    text  =  nltk.word_tokenize( "And now for something completely different" )
    print (text)
    ['And', 'now', 'for', 'something', 'completely', 'different']
"""
words = nltk.word_tokenize(Data)
print("words=", words)

# 生成 word2index和index2word的字典
word2index = {}
index2word = {}
# 计数器
count = 0
# 遍历所有词集合 生成词到索引的字典
for word in words:
    # 这个词索引是不重复的
    if word not in word2index.keys():
        word2index[word] = count
        count = count + 1

# 从前面词的索引表反向生成索引到词的字典
for word in word2index.keys():
    index2word[word2index[word]] = word

print("word2index len", len(word2index))
print("index2word len", len(index2word))
print("word2index=", word2index)
print("index2word=", index2word)


# 返回一个词表
def getWindows(words, window_size):
    Dataset = []
    for i in range(len(words)):
        # 取前后n个单词
        for j in range(i - window_size, i + window_size + 1, 1):
            # print("j=", j)
            if j < 0 or j > (len(words) - 1) or j == i:
                continue
            Dataset.append((words[i], words[j]))
    return Dataset


# 生成词的ont-hot向量
def getOneHotwordvec(word, word2index):
    """
       :param word: 输入的一个词，比如输入一个“政界”
       :param word2index: 字面意思
       :return: 返回输入词所对应的ont-hot向量
       """
    onehotWordvec = np.zeros(shape=(len(word2index), 1))  # 行为所有词的长度，列为1
    onehotWordvec[word2index[word]][0] = 1  # 对应词的索引所对的值赋值为1
    return onehotWordvec


# 测试ont-hot向量生成
word = 'study'


# print(getOneHotwordvec(word, word2index))


# print("get word context dataset = ", getWindows(words, 3)[5])


# 生成训练集
def getTrain(words, window_size, word2index):
    X_train, y_train = [], []
    Dataset = getWindows(words, window_size)
    # print("Dataset=",Dataset)
    batch_size = 100
    for i in trange(1, 100000, batch_size):
        # 每个中心词和上下文词都在当前窗口内
        for centre_word, context_word in tqdm(Dataset[i: i + batch_size - 1]):
            # 拿到x的one-hot向量
            X_train.append(getOneHotwordvec(centre_word, word2index))
            y_train.append(getOneHotwordvec(context_word, word2index))
            # print("X_train", X_train)
            # print("y_train", y_train)
    return X_train, y_train


X_train, y_train = getTrain(words, 3, word2index)
print("X_train len=", len(X_train))
print("y_train len=", len(y_train))

X_train = np.array(X_train)
print("x转换成array完成")
y_train = np.array(y_train)
print("y转换成array完成")

print("X_train array shape=", X_train.shape)
print("y_train array shape=", y_train.shape)

# X_train = X_train.reshape(11398, 99000)
# X_train = X_train.reshape(1063, 99000)
# X_train = X_train.reshape(4498, 61811)
X_train = X_train.reshape(1049, 14012)
print("reshape之后X_train.shape=", X_train.shape)
print("X_train=", X_train)

# y_train = y_train.reshape(11398, 99000)
# y_train = y_train.reshape(1063, 99000)
# y_train = y_train.reshape(4498, 61811)
y_train = y_train.reshape(1049, 14012)
print("reshape之后y_train.shape=", y_train.shape)
print("y_train=", y_train)


# 初始化权重矩阵，有个两个矩阵，分别是VxD 和 DxV，为什么这么定义，参考论文的原理
def weightInit(dimension, pDimension):
    # 第一层矩阵初始化
    W1 = np.random.randn(pDimension, dimension)
    # 第一层偏置初始化
    b1 = np.random.randn(pDimension, 1)
    # 第二层矩阵初始化,维度和是第一个矩阵的转置
    W2 = np.random.randn(dimension, pDimension)
    # 第二层偏置初始化
    b2 = np.random.randn(dimension, 1)
    return W1, b1, W2, b2


# relu激活函数
def relu(z):
    return np.maximum(0, z)


# softmax函数
def softmax(z):
    ex = np.exp(z)
    return ex / np.sum(ex, axis=0)


# 前向传播
def forward(x, W1, b1, W2, b2):
    # 第一层前向传播即 权重与输入的x的乘积加上偏置
    Z1 = np.dot(W1, x) + b1
    # 添加一个rule函数
    Z1 = relu(Z1)
    # 第二层前向传播即 上一层的输出作为当前层的输入与当前权重的乘积,再加上第二层的偏置值
    Z2 = np.dot(W2, Z1) + b2
    # 将第二层的输出值经过softmax函数，得到概率值
    ypred = softmax(Z2)
    return Z1, Z2, ypred


# 误差计算
def errorCalculation(y, ypred, m):
    error = -(np.sum(np.multiply(y, np.log(ypred)))) / m
    return error


# 反向传播
def backProp(W1, b1, W2, b2, Z1, Z2, y, ypred, x):
    dW1 = np.dot(relu(np.dot(W2.T, ypred - y)), x.T)
    db1 = relu(np.dot(W2.T, ypred - y))
    dW2 = np.dot(ypred - y, Z1.T)
    db2 = ypred - y
    return dW1, db1, dW2, db2


# 模型训练
def model(x, y, epoches=10, learning_rate=0.00001):
    # x的行列数
    dimension = x.shape[0]
    m = x.shape[1]
    # 生成词向量维度，300表示生成的词向量用300行数字表示这个词
    pDimension = 300
    W1, b1, W2, b2 = weightInit(dimension, pDimension)
    error = []
    for i in tqdm(range(epoches)):
        Z1, Z2, ypred = forward(x, W1, b1, W2, b2)
        error.append(errorCalculation(y, ypred, m))
        dW1, db1, dW2, db2 = backProp(W1, b1, W2, b2, Z1, Z2, y, ypred, x)
        # 更新权重和偏置
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        print("error=", error)
    return ypred, error, W1, W2


# 传入训练集和epoch次数，以及学习率
ypred, error, W1, W2 = model(X_train, y_train, 10, 0.00001)
# 把中心词向量矩阵和周围词向量矩阵相加求平均即为词向量矩阵
W = np.add(W1, W2.T) / 2

# 生成词嵌入字典，即{单词1:词向量1,单词2:词向量2...}的格式
word2vec = {}
for word in word2index.keys():
    # 词向量矩阵中某个词的索引所对应的那一列即为所该词的词向量
    word2vec[word] = W[:, word2index[word]]

print("word2vec=", word2vec)

"""
    继续降维，比如food，food转为词向量后
    'food': array([-2.17007321e-01, -5.89359459e-01,  1.02294753e+00,  1.90721062e+00,
       -1.81680167e-01,  6.68201929e-01,  9.83180179e-01, -1.81787765e-01,
        7.74756085e-01,  1.98217257e-01, -2.89414012e-01,  5.14236711e-01,
        ......省略70多行
        2.64303636e-01, -4.24010150e-01,  2.40863888e-01, -5.55736922e-01,
        7.84205414e-01, -6.04794130e-01,  2.40553756e-01, -1.19349800e-01,
        5.47898499e-02,  5.63425593e-01,  9.52913677e-02, -1.09280454e+00]),
        维度还是非常高，把其降为2个维度，方便在二维空间画出点。降维的原理查看下面知乎回答
        https://zhuanlan.zhihu.com/p/77151308
"""
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(W.T)

# 降维后在生成一个词嵌入字典，即即{单词1:(维度一，维度二),单词2:(维度一，维度二)...}的格式
word2ReduceDimensionVec = {}
for word in word2index.keys():
    word2ReduceDimensionVec[word] = principalComponents[word2index[word], :]

# 将生成的字典写入到文件中
with open("en_wordvec2.txt", 'w') as f:
    for key in word2index.keys():
        f.write('\n')
        f.writelines('"' + str(key) + '":' + str(word2vec[key]))
    f.write('\n')

# 将词向量可视化
plt.figure(figsize=(20, 20))
# 只画出1000个，太多显示效果很差
count = 0
for word, wordvec in word2ReduceDimensionVec.items():
    if count < 1000:
        plt.scatter(wordvec[0], wordvec[1])
        plt.annotate(word, (wordvec[0], wordvec[1]))
        count += 1
plt.show()
