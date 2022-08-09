from cProfile import label
from pickletools import optimize
from turtle import forward
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels,texts=[],[]
vocab_dic={}
file='datasets/中文数据集/toutiao-text-classfication-dataset-master/toutiao_cat_data.txt'
with open(file, 'r', encoding='UTF-8') as f:
    for line in f:
        _,category,_,text,key_word=line.strip().split('_!_')

        labels.append(int(category)-100)
        text_split=list(text)
        texts.append(text_split)
        # print(int(category)-100, text_split);input()
        for token in text_split:
            vocab_dic[token]=vocab_dic.get(token,0)+1

# print(labels[5],texts[5],vocab_dic);

#print('line28', len(vocab_dic));input()

#print('line30', len(vocab_dic));input()
vocab_list = sorted([_ for _ in vocab_dic.items()], key=lambda x: x[1], reverse=True)
vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
t = len(vocab_dic)
vocab_dic.update({'<UNK>': t, '<PAD>': t + 1})
token2index=vocab_dic
index2token={index:token for token, index in vocab_dic.items()}
#print(len(texts),len(token2index), len(index2token));input()
datas=[]
for line in texts:
    t=[]
    for token in line:
        t.append(token2index[token])
    datas.append(t)

class MyDataset(data.Dataset):
    def __init__(self,datas,target) -> None:
        super(MyDataset).__init__()
        self.datas=datas
        self.target=target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.datas[index],self.target[index]

def collect_fn(x):
    max_len=max(len(i[0]) for i in x)
    xi_list = []
    yi_list = []
    for xi_, yi_ in x:
        xi = xi_ +[token2index['<PAD>']]*(max_len-len(xi_))
        xi_list.append(xi)
        yi_list.append(yi_)
        #data[0]+=[token2index['<PAD>']]*(max_len-len(data[0]))
        
    return torch.tensor(xi_list,dtype=torch.long),\
           torch.tensor(yi_list,dtype=torch.long)
class ConvRelu(nn.Module):
    def __init__(self, num_filters, kernel_size) -> None:
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(1, num_filters, kernel_size)
        self.relu = nn.ReLU()
    
    #(batch_size, 1, seq_len, emb_dim)
    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)

class TextCnn(nn.Module):
    def __init__(self,emb_dim,vocab_size,filter_size,num_filters,n_classs) -> None:
        super(TextCnn,self).__init__()
        self.emb_dim=emb_dim
        self.vocab_size=vocab_size
        self.filter_size=filter_size
        self.num_filters=num_filters
        self.n_classs=n_classs
        self.embedding=nn.Embedding(self.vocab_size,self.emb_dim)
        self.conv_layer= []
        for k in self.filter_size:
            self.conv_layer.append(
                ConvRelu(self.num_filters, (k, emb_dim))
            )
        self.conv_layer = nn.ModuleList(self.conv_layer)
        #经过卷积后变为(batch_size,num_filters,seq_len-k+1,1)
        self.dropout=nn.Dropout(0.0)
        self.fc=nn.Linear(self.num_filters*len(self.filter_size),n_classs)

    def forward(self,x):
        # print(torch.max(x), self.embedding.weight.shape);exit()
        x=self.embedding(x)
        x = self.dropout(x)
        #(batch_size,seq_len,emb_dim)
        x=x.unsqueeze(1)
        #(batch_size,1,seq_len,emb_dim)

        #xs[i].shape = (batch_size, num_filters, seq_len - k + 1)
        xs = [conv(x).squeeze(3) for conv in self.conv_layer]

        pools = []
        for x in xs:
            #pools[i].shape = (batch_size, num_filters)
            pools.append(torch.amax(x, dim=-1))#amax, argmax, max
        
        #cat.shape = (batch_size, num_filters*num_conv_layer)
        cat = torch.cat(pools, dim=1)

        #.shape = (batch_size, n_classes)
        return self.fc(cat)

        x=[nn.ReLU(conv(x).squeeze(3)) for conv in self.conv_layer]
        #经过卷积后变为(batch_size,num_filters,seq_len-k+1)
        F.amax(x, dim=-1)
        x=[nn.MaxPool1d(i,i.size(2))(i).squeeze(2) for i in x]
        #经过池化后变为(batch_size,num_filters)
        x=torch.cat(x,1)
        #拼接后后变为(batch_size,num_filters*len(self.filter_size))
        x=self.fc(x)
        #(batch_size,n_classs)

model=TextCnn(emb_dim=200,vocab_size=len(token2index),filter_size=[3,4,5],num_filters=8,n_classs=17)
dataset=MyDataset(datas,labels)
optimizer=optim.Adam(model.parameters(),lr=1e-3)
loss_fn=nn.CrossEntropyLoss()
dataloader=data.DataLoader(dataset=dataset,batch_size=128,shuffle=True,collate_fn=collect_fn)
#dataloader=data.DataLoader(dataset,128,True,collect_fn)
epochs=10
for epoch in range(epochs):
    model.train()
    print(f'epoch={epoch}')
    train_loss=0
    train_acc=0
    for i,(datas,target) in enumerate(dataloader):
        #datas,target=datas.to(device),target.to(device)
        #print(datas.shape,target.shape);input()
        output=model(datas)
        loss=loss_fn(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pre=torch.argmax(output,1)
        # acc=torch.sum(pre==target)#/datas.size(0)
        acc = torch.sum(pre==target).item()
        #  print(datas.shape,target.shape);input()
        train_loss+=loss.item()
        train_acc+=acc
        # if i%100==0:
        print(f'loss={loss.item()}')
        print(f'acc={float(acc)/datas.size(0)}')
    #print(f'train_loss={train_loss}')
