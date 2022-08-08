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
dict_word = {index:word for word, index in word_dict.items()}
freqs_dict = {}
for sentence in sentences:
    freqs_dict[sentence] = freqs_dict.get(sentence, 0) + 1

freqs = []
for index in range(len(word_dict)):
    word = dict_word[index]
    times = word_dict[word]
    freqs.append(times)
sum_freqs = sum(freqs)
probability = [freq/sum_freqs for freq in freqs]


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
        # log_probs = F.log_softmax(out, dim=1)
        # return log_probs
        return out

# 训练模型
model = CBOW(len(word_dict), embeding_dim, window_size, hidden_dim)
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

#to device
model = model.to(device)
criterion = criterion.to(device)
# optimizer = optimizer.to(device)
sample_size = 3
for epoch in range(5000):
    total_loss = 0
    for content, target in data:
        content_vector = make_content_vector(content, word_dict)
        target = torch.tensor([word_dict[target]], dtype=torch.long)
        content_vector = content_vector.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        #pred.shape = (batch_size, n_vocab)
        pred = model(content_vector)
        batch_size = pred.size(0)

        #weights.shape = (n_vocab,)
        #weights_unsqueeze.shape = (1, n_vocab)
        #weights_tile.shape = (batch_size, n_vocab)
        #negative_sample.shape = (batch_size, sample_size)
        weights = torch.tensor(probability, dtype=torch.float32, device=device)
        weights_unsqueeze = torch.unsqueeze(weights, 0)
        weights_tile = torch.tile(weights_unsqueeze, (batch_size, 1))
        negative_sample = torch.multinomial(weights_tile, sample_size, replacement=False)

        #target.shape = (batch_size, )
        # #positive_sample.shape = (batch_size, 1)
        #sample.shape = (batch_size, sample_size + 1)
        positive_sample = torch.unsqueeze(target, 1)
        sample = torch.cat([positive_sample, negative_sample], dim=1)

        #sampled_pred.shape = (batch_size, sample_size + 1)
        #pred.shape = (batch_size, n_vocab)
        sampled_pred = torch.gather(pred, 1, sample)
        # #mask.shape = (batch_size, n_vocab) ==>True
        # mask = torch.ones_like(pred, dtype=torch.bool, device=device)
        # mask_scatter1 = mask.scatter(1, negative_sample, False)
        # mask_scatter2 = mask_scatter1.scatter(1, positive_sample, False)

        #new_targe.shape = (batch_size,)
        new_target = torch.zeros(batch_size, dtype=torch.long, device=device)

        # loss = criterion(pred, target)
        loss = criterion(sampled_pred, new_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%03d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))



#n_vocab = 6
#【0.1, -0.2, 0.5, 0.7, 1.0, -2.0】
#[0, 0, 0, 1, 0, 0]

#sample_size=2
#[0.1, 1.0]
#[0.7, 0.1, 1.0]
