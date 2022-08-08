from curses import window
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

corpus = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""

# 模型参数
window_size = 2
embeding_dim = 100
hidden_dim = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据预处理
sentences = corpus.split()  # 分词
words = list(set(sentences))
word_dict = {word: i for i, word in enumerate(words)}  # 每个词对应的索引
data = []  # 准备数据
for i in range(window_size, len(sentences)-window_size):
    # content = [sentences[i-1], sentences[i-2],
            #    sentences[i+1], sentences[i+2]]
    content_left = [sentences[i+j] for j in range(-window_size, 0)]
    content_right = [sentences[i+j] for j in range(1, window_size + 1)]
    content = content_left + content_right
    # print('content_left', content_left)
    # print('content_right', content_right)
    # print('content', content)
    # print('target', sentences[i]);exit()

    target = sentences[i]
    data.append((content, target))
# print(data[:5]);exit()

# 处理输入数据
def make_content_vector(content, word_to_ix):
    idx = [word_to_ix[w] for w in content]
    return torch.LongTensor(idx)

# CBOW模型
class CBOW(nn.Module):
    def __init__(self, vocab_size, n_dim, window_size, hidden_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.linear1 = nn.Linear(2*n_dim*window_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, X):
        embeds = self.embedding(X).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# 训练模型
model = CBOW(len(word_dict), embeding_dim, window_size, hidden_dim)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

#to device
model = model.to(device)
criterion = criterion.to(device)
# optimizer = optimizer.to(device)
for epoch in range(5000):
    total_loss = 0
    for content, target in data:
        content_vector = make_content_vector(content, word_dict)
        target = torch.tensor([word_dict[target]], dtype=torch.long)
        # if torch.cuda.is_available():
        #     content_vector = content_vector.cuda()
        #     target = target.cuda()
        content_vector = content_vector.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        log_probs = model(content_vector)
        loss = criterion(log_probs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%03d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
