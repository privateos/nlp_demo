
import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

text = (
    'Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)
# text = (
#     'abc+'
#     '!#$$'
# )
# print(text);exit()


sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n') # filter '.', ',', '?', '!'
# print(sentences);exit()

sentences_join = " ".join(sentences)
word_list = list(set(sentences_join.split()))
# print(word_list);exit()


word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {index:word for word, index in word2idx.items()}
vocab_size = len(word2idx)



token_list = []
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    # print(sentence)
    # print(sentence.split())
    # print(arr)
    # exit()
    token_list.append(arr)

# BERT Parameters
maxlen = 30
batch_size = 6
max_pred = 5 # max tokens of prediction
n_layers = 6
n_heads = 12
d_model = 768#768=64*12
d_ff = 768*4 # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2

# sample IsNext and NotNext to be same in small batch size
def make_data():
    #batch_size = 6
    #max_pred = 5
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        #random.randrange(start, end, step=1):从[start, end)区间中以step为步长随机选取一个数for i in range(start, end, step)
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences)) # sample random index in sentences
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        #n_pred:被mask的单词数量
        n_pred =  min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15 % of tokens in one sentence
        #记录可以被mask的单词的下标
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']
                        ] # candidate masked position
        
        #random.shuffle()
        shuffle(cand_maked_pos)#打乱顺序，方面后续随机选取数据mask
        masked_tokens, masked_pos = [], []
        #选前n_pred个单词进行mask
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)#记录input_ids的下标（被替换的）
            masked_tokens.append(input_ids[pos])#记录被替换之前的单词是哪一个
            if random() < 0.8:  # 80%
                #[0, 0.8)使用[MASK]
                input_ids[pos] = word2idx['[MASK]'] # make mask

                #[0.8, 0.9)不替换
            elif random() > 0.9:  # 10%
                #[0.9, 1.0]随机替换：但是不可以使用'CLS', 'SEP', 'PAD'
                #random.randint(a, b)生成[a, b]的整数
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 4: # can't involve 'CLS', 'SEP', 'PAD'
                  index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace


        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)#PAD
            masked_pos.extend([0] * n_pad)#CLS
        #pred = (batch_size, max_pred, d_model)-->(batch_size, max_pred, vocab_size)-->(batch_size*max_pred, vocab_size)
        #masked_tokens.shape = (batch_size, max_pred)--->(batch_size*max_pred, )
        #nn.CrossEntropyLoss(pred, masked_tokens)


        #input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        #segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        #(batch_size, maxlen, d_model)--->(batch_size, max_pred, d_model):torch.gather
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch
# Proprecessing Finished

#batch=[
# [input_ids, segment_ids, masked_tokens, masked_pos, Bool],
# [...]
# ]
batch = make_data()
# print(batch[0]);exit()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)#解数据操作
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.LongTensor(input_ids),  torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens),\
    torch.LongTensor(masked_pos), torch.LongTensor(isNext)

class MyDataSet(Data.Dataset):
  def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
    self.input_ids = input_ids
    self.segment_ids = segment_ids
    self.masked_tokens = masked_tokens
    self.masked_pos = masked_pos
    self.isNext = isNext
  
  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, idx):
    return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]
dataset = MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext)
loader = Data.DataLoader(dataset, batch_size, True)

def get_attn_pad_mask(seq_q, seq_k):
    #(batch_size, maxlen)
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    #(batch_size, 1, maxlen)
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]

    #pad_attn_mask.shape = (batch_size, 1, maxlen)
    #(batch_size, seq_len, seq_len):seq_len = maxlen
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding:(maxlen, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        #self.norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x, seg):
        #x.shape = (batch_size, maxlen)
        #seg.shape = (batch_size, maxlen)
        seq_len = x.size(1)#seq_len = maxlen
        pos = torch.arange(seq_len, dtype=torch.long)#(0, 1, 2, ..., maxlen-1)
        #pos.shape = (maxlen, )---> (1, maxlen)--->(batch_size, maxlen)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]

        #(batch_size, maxlen, d_model)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)#bias暂时设置为False
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.dot_attention = ScaledDotProductAttention()#
        self.linear = nn.Linear(d_v * n_heads, d_model)
        #(batch_size, input_size)-->(batch_size, output_size)
        #(batch_size, size1, input_size)-->(batch_size, size1, output_size)
        #(*, input_size)--->(*, output_size)

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)#暂时定为False

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        # context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = self.dot_attention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads, d_v]

        #
        output = self.linear(context)
        return self.norm(output + residual) # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        #enc_inputs.shape = [bach_size, seq_len, d_model]
        #enc_self_attn_mask = [batch_size, seq_len, seq_len]
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])


        ######################################
        #(batch_size, seq_len, d_model)
        #(batch_size, d_model)
        self.fc = nn.Sequential(
            #(batch_size, d_model)
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
            #(batch_size, d_model)
        )
        self.classifier = nn.Linear(d_model, 2)
        #(batch_size, 2)
        ######################################


        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight#weight.shape = (vocab_size, d_model)
        self.fc2 = nn.Linear(d_model, vocab_size)#weight.shape = (vocab_size, d_model)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        #word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']
        #[CLS] , sentence1, [SEP], sentence2, [SEP]

        #input_ids.shape = (batch_size, 1 + len1 + 1 + len2 + 1 )-->(batch_size, maxlen)
        #segment_ids.shape = (bat_size, 1 + len1 + 1 + len2 + 1)--->(batch_size, maxlen)
        #masked_pos.shape = (batch_size, n_pred)-->(batch_size, max_pred)

        output = self.embedding(input_ids, segment_ids) # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        #output.shape = (batch_size, seq_len, d_model)
        h_pooled = self.fc(output[:, 0]) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2] predict isNext







        # self.linear = nn.Linear(d_model, d_model)
        # self.activ2 = gelu
        # # fc2 is shared with embedding layer
        # embed_weight = self.embedding.tok_embed.weight#weight.shape = (vocab_size, d_model)
        # self.fc2 = nn.Linear(d_model, vocab_size, bias=False)#weight.shape = (vocab_size, d_model)
        # self.fc2.weight = embed_weight

        #masked_pos.shape = (batch_size, max_pred)-->(batch_size, max_pred, 1)-->(batch_size, max_pred, d_model)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, max_pred, d_model]

        #output.shape = (batch_size, seq_len, d_model)
        h_masked = torch.gather(output, 1, masked_pos)
        #h_masked.shape = (batch_size, max_pred, d_model)
        h_masked = self.activ2(self.linear(h_masked)) 
        #h_masked.shape = [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf

######################################
model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)

for epoch in range(50):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
        #logits_lm.shape = (batch_size, max_pred, vocab_size)
        #logits_clsf.shape = (batch_size, 2)
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        #loss_lm.type = torch.Tensor


        #logits_lm.view(-1, vocab_size)  .shape = (batch_size*max_pred, vocab_size)
        #masked_tokens.shape = (batch_size, max_pred)--->(batch_size*max_pred,)
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
        # loss_lm = (loss_lm.float()).mean()

        #logits_clsf.shape = (batch_size, 2)
        #isNext.shape = (batch_size, )
        loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Predict mask tokens ans isNext
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[1]
print(text)
print('================================')
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
                 torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)


