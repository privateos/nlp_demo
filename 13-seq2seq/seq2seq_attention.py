# code by Tae Hwan Jung(Jeff Jung) @graykode, modify by wmathor
import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# ?: Symbol that will fill in blank sequence if current batch data size is short than n_step

letter = [c for c in 'SE?abcdefghijklmnopqrstuvwxyz']
letter2idx = {n: i for i, n in enumerate(letter)}
idx2letter = {i:n for i, n in enumerate(letter)}

seq_data = [
    ['man', 'women'], 
    ['black', 'white'], 
    ['king', 'queen'], 
    ['girl', 'boy'], 
    ['up', 'down'], 
    ['high', 'low']
]

# Seq2Seq Parameter
n_step = max([max(len(i), len(j)) for i, j in seq_data]) # max_len(=5)
n_hidden = 128
n_class = len(letter2idx) # classfication problem
batch_size = 2

def make_data(seq_data):
    enc_input_all, dec_input_all, dec_output_all = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + '?' * (n_step - len(seq[i])) # 'man??', 'women'

        enc_input = [letter2idx[n] for n in (seq[0] + 'E')] # ['m', 'a', 'n', '?', '?', 'E']
        dec_input = [letter2idx[n] for n in ('S' + seq[1])] # ['S', 'w', 'o', 'm', 'e', 'n']
        dec_output = [letter2idx[n] for n in (seq[1] + 'E')] # ['w', 'o', 'm', 'e', 'n', 'E']

        enc_input_all.append(enc_input)
        dec_input_all.append(dec_input)
        dec_output_all.append(dec_output) # not one-hot

    # make tensor
    return torch.Tensor(enc_input_all).long(), torch.Tensor(dec_input_all).long(), torch.LongTensor(dec_output_all).long()

class TranslateDataSet(Data.Dataset):
    def __init__(self, enc_input_all, dec_input_all, dec_output_all):
        self.enc_input_all = enc_input_all
        self.dec_input_all = dec_input_all
        self.dec_output_all = dec_output_all
    
    def __len__(self): # return dataset size
        return len(self.enc_input_all)
    
    def __getitem__(self, idx):
        return self.enc_input_all[idx], self.dec_input_all[idx], self.dec_output_all[idx]


enc_input_all, dec_input_all, dec_output_all = make_data(seq_data)
loader = Data.DataLoader(TranslateDataSet(enc_input_all, dec_input_all, dec_output_all), batch_size, True)


class Encoder(nn.Module):
    def __init__(self, 
        source_n_vocab, emb_dim, 
        enc_hid_dim, 
        dec_hid_dim, 
        dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(source_n_vocab, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True, batch_first=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src): 
        #src.shape = [batch_size, src_len]

        #embedded.shape = (batch_size, src_len, emb_dim)
        embedded = self.dropout(self.embedding(src))
        
        # enc_output = [batch_size, src_len, enc_hid_dim * num_directions]
        #            = [batch_size, src_len, enc_hid_dim*2]
        # enc_hidden = [batch_size, n_layers * num_directions, hid_dim]
        #            = [batch_size, 1*2, enc_hid_dim]
        enc_output, enc_hidden = self.rnn(embedded)

        #mid.shape = (batch_size, enc_hid_dim*2)
        mid = torch.cat((enc_hidden[:, 0, :], enc_hidden[:, 1, :]), dim=1)
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(mid))
        
        # enc_output = [batch_size, src_len, enc_hid_dim * num_directions]
        #            = [batch_size, src_len, enc_hid_dim*2]
        #s.shape = [batch_size, dec_hid_dim]
        return enc_output, s

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    
    def forward(self, Q, K, V):
        #Q.shape = (batch_size, m, K_size)
        #K.shape = (batch_size, seq_len, K_size)
        #V.shape = (batch_size, seq_len,V_size)
        #K_size = V_size = lstm_hid_size

        K_transpose = torch.transpose(K, 1, 2)
        #K_transpose.shape = (batch_size, K_size, seq_len)
        d = K_transpose.size(1)
        QK = torch.bmm(Q, K_transpose)/math.sqrt(d)
        #QK.shape = QK_softmax=(batch_size, m, seq_len)
        QK_softmax = torch.softmax(QK, dim=2)
        #V.shape = (batch_size, seq_len,V_size)

        #result.shape = (batch_size, m, V_size)
        result = torch.bmm(QK_softmax, V)
        return result
class Decoder(nn.Module):
    def __init__(self, 
        target_n_vocab, 
        emb_dim, 
        enc_hid_dim, 
        dec_hid_dim, 
        dropout, 
    ):
        super(Decoder, self).__init__()
        # self.output_dim = target_n_vocab
        self.tranform_s_for_attention = nn.Linear(dec_hid_dim, enc_hid_dim*2, bias=False)
        self.attention = Attention()
        self.embedding = nn.Embedding(target_n_vocab, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, target_n_vocab)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_input, s, enc_output):
             
        # dec_input.shape = [batch_size]
        # s.shape = [batch_size, dec_hid_dim]
        # enc_output.shape = [batch_size, src_len, enc_hid_dim * 2]

        # dec_input.shape = [batch_size, 1]
        dec_input = dec_input.unsqueeze(1) 
        # embedded.shape = [batch_size, 1, emb_dim]
        embedded = self.dropout(self.embedding(dec_input))
        
        #Q.shape = (batch_size, enc_hid_dim*2)
        Q = self.tranform_s_for_attention(s)  
        #Q.shape = (batch_size, 1, enc_hid_dim*2)
        Q = torch.unsqueeze(Q, 1)
        #K.shape = V.shape = (batch_size, src_len, enc_hid_dim*2)
        K = enc_output
        V = enc_output
        # print(Q.shape, K.shape, V.shape)
        # c.shape = [batch_size, 1, enc_hid_dim * 2]
        c = self.attention(Q, K, V)

        # rnn_input = [batch_size, 1, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, c), dim = 2)
            
        # dec_output = [batch_size, 1, dec_hid_dim]
        # dec_hidden = [1, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))
        
        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(1)
        dec_output = dec_output.squeeze(1)
        c = c.squeeze(1)
        
        # pred = [batch_size, target_n_vocab]
        pred = self.fc_out(torch.cat((dec_output, c, embedded), dim = 1))
        
        return pred, dec_hidden.squeeze(0)



# class Encoder(nn.Module):
#     def __init__(self, 
#         source_n_vocab, emb_dim, 
#         enc_hid_dim, 
#         dec_hid_dim, 
#         dropout):
# class Decoder(nn.Module):
#     def __init__(self, 
#         target_n_vocab, 
#         emb_dim, 
#         enc_hid_dim, 
#         dec_hid_dim, 
#         dropout, 
#     ):
class Seq2Seq(nn.Module):
    def __init__(self, 
        source_n_vocab,
        source_emb_dim,
        enc_hid_dim,

        target_n_vocab,
        target_emb_dim,
        dec_hid_dim,

        dropout=0.1,

        device='cpu'
    ):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(source_n_vocab, source_emb_dim, enc_hid_dim, dec_hid_dim, dropout)
        self.decoder = Decoder(target_n_vocab, target_emb_dim, enc_hid_dim, dec_hid_dim, dropout)
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        # src = [batch_size, src_len]
        # trg = [batch_size, trg_len]
        
        trg_len = trg.shape[1]
        
        # tensor to store decoder outputs
        # outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #enc_output.shape = (batch_size, src_len, enc_hid_dim*2)
        #s.shape = (batch_size, dec_hid_dim)
        enc_output, s = self.encoder(src)
                
        #dec_input.shape = (batch_size)
        dec_input = trg[:, 0]
        
        outputs = []
        for t in range(trg_len):
            if t != 0:
                teacher_force = random.random() < teacher_forcing_ratio
                if teacher_force:
                    dec_input = trg[:, t]
                else:
                    dec_input = dec_output.argmax(1)
            
            #dec_input.shape = (batch_size,)
            #s.shape = (batch_size, dec_hid_dim)
            #enc_output.shape = (batch_size, src_len, enc_hid_dim*2)
            dec_output, s = self.decoder(dec_input, s, enc_output)
            #dec_output.shape = (batch_size, target_n_voca)
            #s.shape = (batch_size, dec_hid_dim)


            # place predictions in a tensor holding predictions for each token
            outputs.append(dec_output)
            

        #outputs = [(batch_size, target_n_vocab)]
        #outputs.shape = (batch_size, trg_len, target_n_vocab)
        outputs = torch.stack(outputs, dim=1)
        return outputs


source_n_vocab = len(letter)
source_emb_dim = 16
enc_hid_dim = 17

target_n_vocab = len(letter)
target_emb_dim = 18
dec_hid_dim = 19


lr = 0.001
epochs = 100
trg = n_step + 1
model = Seq2Seq(source_n_vocab, 
            source_emb_dim, 
            enc_hid_dim, 


            target_n_vocab,
            target_emb_dim,
            dec_hid_dim
        )
optimizer = optim.Adam(model.parameters(), lr)
loss_fn = nn.CrossEntropyLoss()
#train

for epoch in range(epochs):
    model.train()
    teacher_forcing_ratio = 1.0
    train_loss = 0
    train_iters = 0
    for enc_input, dec_input, dec_output in loader:
        # print(enc_input.shape, dec_input.shape, dec_output.shape)
        #enc_input.shape = (batch_size, src_len)
        #dec_input.shape = (batch_size, trg_len)
        #dec_output.shape = (batch_size, trg_len)

        #src, trg, teacher_forcing_ratio
        #pred.shape = (batch_size, trg_len, target_n_vocab)
        # print()
        pred = model(enc_input, dec_input, teacher_forcing_ratio)


        #pred.shape = (batch_size*trg_len, target_n_vocab)
        #label.shape = (batch+size*trg_len, )
        
        pred = torch.reshape(pred, (-1, target_n_vocab))
        label = torch.reshape(dec_output, (-1, ))
        # print(pred.shape, label.shape)

        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iters += 1
    message = 'epoch={0:>4},train_loss={1:>7.4}'
    print(message.format(epoch, train_loss/train_iters))
    # with torch.no_grad():
    #     model.eval()
    #     teacher_forcing_ratio = 0.0
    #     for enc_input, dec_input, dec_output in loader:
    #         # print(enc_input.shape, dec_input.shape, dec_output.shape)
    #         #enc_input.shape = (batch_size, src_len)
    #         #dec_input.shape = (batch_size, trg_len)
    #         #dec_output.shape = (batch_size, trg_len)

    #         #src, trg, teacher_forcing_ratio
    #         #pred.shape = (batch_size, trg_len, target_n_vocab)
    #         # print()
    #         pred = model(enc_input, dec_input, teacher_forcing_ratio)
    #         # print(pred.shape, label.shape)


        
