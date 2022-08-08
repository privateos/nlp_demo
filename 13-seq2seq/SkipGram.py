import torch
from torch import nn
import matplotlib.pyplot as plt

# ----------------------------
# 初始化定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

context_size = 3
# 上下文大小，即当前词与关联词的最大距离

lr = 1e-3
batch_size = 64
li_loss = []

# ----------------------------
# 数据预处理
with open("word.txt", "r", encoding="utf-8") as f:
    fr = f.read()
    # print(type(fr))
    frs = fr.strip()
    lines = frs.split('\n')
    # print(type(lines), len(lines))


set_words = set()
# 存放所有单词集合
for words in lines:
    for word in words.split(' '):
        set_words.add(word)
word_size = len(set_words)
# 单词数
word_to_id = {word: i for i, word in enumerate(set_words)}
# 单词索引id
id_to_word = {word_to_id[word]: word for word in word_to_id}
# id索引单词

train_x = []
train_y = []
for words in lines:
    li_words = words.split()
    # 存放每一行的所有单词
    for i, word in enumerate(li_words):
        for j in range(-context_size, context_size + 1):
            # 对于每个单词，将上下文大小内的词与其进行关联
            if i + j < 0 or i + j > len(li_words) - 1 or li_words[i + j] == word:
                # 对于上下文越界以及当前单词本身不添加关联关系
                continue
            train_x.append(word_to_id[word])
            train_y.append(word_to_id[li_words[i + j]])
            # 训练数据基于Skip-Gram，输入当前词，输出当前词上下文大小内的所有词

# ----------------------------
# 定义模型          
class EmbedWord(nn.Module):
    def __init__(self, word_size, context_size):
        super(EmbedWord, self).__init__()
        self.embedding = nn.Embedding(word_size, 16)
        # 用128维向量表示每个单词
        self.linear = nn.Linear(16, word_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.log_softmax(x)
        return x

model = EmbedWord(word_size, context_size).to(device)

# ----------------------------
# 定义优化器和loss
loss_fun = nn.NLLLoss()
# 这里使用NLL作为损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ----------------------------
# 开始训练
model.train()
for epoch in range(len(li_loss), 150):
    if epoch % 2000 == 0 and epoch > 0:
        optimizer.param_groups[0]['lr'] /= 1.05
        # 每2000轮下降一点学习率
    for batch in range(0, len(train_x) - batch_size, batch_size):
        word = torch.tensor(train_x[batch: batch + batch_size]).long().to(device)
        label = torch.tensor(train_y[batch: batch + batch_size]).to(device)
        out = model(word)
        loss = loss_fun(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    li_loss.append(loss.item())   
    
    if epoch % 50 == 0 and epoch > 0:
        print('epoch: {}, Loss: {}, lr: {}'.format(epoch, loss, optimizer.param_groups[0]['lr']))
        plt.plot(li_loss[-500:])
        plt.show()
        
        # for w in range(5):
        #     # 每50轮测试一下前5个单词的预测结果
        #     pred = model(torch.tensor(w).long().to(device))
        #     print("{} -> ".format(id_to_word[w]), end="\t")
        #     for i, each in enumerate((-pred).argsort()[:10]):
        #         print("{}:{}".format(i, id_to_word[int(each)]), end="   ")
        #     print()
            
# ----------------------------
# 降维可视化
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

result = model(torch.tensor([i for i in range(word_size)]).long().to(device))
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
embed_two = tsne.fit_transform(model.embedding.weight.cpu().detach().numpy())
# 将词向量降到二维查看空间分布
# embed_two = tsne.fit_transform(result.cpu().detach().numpy())
labels = [id_to_word[i] for i in range(200)]
# 这里就查看前200个单词的分布
plt.figure(figsize=(15, 12))
for i, label in enumerate(labels):
    x, y = embed_two[i, :]
    plt.scatter(x, y)
    plt.annotate(label, (x, y), ha='center', va='top')
plt.legend()
plt.show()
# plt.savefig('词向量降维可视化.png')