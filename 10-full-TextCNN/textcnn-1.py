import argparse
import time
import numpy as np
import random
import os
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta

from sklearn import metrics
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.nn as nn

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        #dataset='THUCNews', embedding='random'

        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'# 训练集 
        #'THUCNews/data/train.txt'

        self.dev_path = dataset + '/data/dev.txt'# 验证集 
        #'THUCNews/data/dev.txt'

        self.test_path = dataset + '/data/test.txt'# 测试集 
        #'THUCNews/data/test.txt'

        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]# 类别名单
            #'THUCNews/data/class.txt'
        
        self.vocab_path = dataset + '/data/vocab.pkl'# 词表 
        #'THUCNews/data/vocab.pkl'

        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'# 模型训练结果 
        #'THUCNews/saved_dict/TextCNN.ckpt'

        self.log_path = dataset + '/log/' + self.model_name
        #'THUCNews/log/TextCNN'

        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 设备

        self.dropout = 0.5 # 随机失活
        self.require_improvement = 1000 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list) # 类别数
        self.n_vocab = 0 # 词表大小，在运行时赋值
        self.num_epochs = 20 # epoch数
        self.batch_size = 128 # mini-batch大小
        self.pad_size = 32 # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3 # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300 # 字向量维度
        #[vocab_size, x] vocab_size是词表的大小
        self.filter_sizes = (2, 3, 4) # 卷积核尺寸
        self.num_filters = 256 # 卷积核数量(channels数)
        self.seed = 0

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        convs = []
        for k in config.filter_sizes:#config.filter_sizes = (2,3,4)
            convs.append(nn.Conv2d(1, config.num_filters, (k, config.embed)))
        self.convs = nn.ModuleList(convs)

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        #x.shape = (batch_size, 1, pad_size, embedding_size)

        x = F.relu(conv(x))
        #x.shape = (batch_size, num_filters, pad_size - conv_kernel_size + 1, 1)

        x = x.squeeze(3)
        #x.shape = (batch_size, num_filters, pad_size - conv_kernel_size + 1)

        # import torch.nn.functional as F
        #(batch_size, in_channels, H)
        x = F.max_pool1d(x, x.size(2))#(x, pad_size - conv_kernel_size + 1)
        #x.shape = (batch_size, num_filters, 1)

        x = x.squeeze(2)
        #x.shape = (batch_size, num_filters)
        return x

    def forward(self, x):
        #x[0].shape = (batch_size, pad_size)

        out = self.embedding(x[0])#(x, seq_len)
        #out.shape = (batch_size, pad_size, embedding_size)

        out = out.unsqueeze(1)
        #out.shape = (batch_size, 1, pad_size, embedding_size)


        # out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        outs = []
        for conv in self.convs:
            t = self.conv_and_pool(out, conv)
            #t.shape = (batch_size, num_filters)
            outs.append(t)
        out = torch.cat(outs, 1)
        #out.shape = (batch_size, num_filters*3)

        out = self.dropout(out)
        out = self.fc(out)
        #out.shape = (batch_size, num_classes)
        return out



MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    #file_path = 'THUCNews/data/train.txt'
    #tokenizer = lambda x: [y for y in x]
    #max_size = 10000
    #min_freq = 1

    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):#use_word = False
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        #config.vocab_path='THUCNews/data/vocab.pkl'
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(
            config.train_path, #'THUCNews/data/train.txt'
            tokenizer=tokenizer, #tokenizer = lambda x: [y for y in x]
            max_size=MAX_VOCAB_SIZE, 
            min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))#config.vocab_path='THUCNews/data/vocab.pkl'

    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    #'THUCNews/data/train.txt'

    dev = load_dataset(config.dev_path, config.pad_size)
    #'THUCNews/data/dev.txt'

    test = load_dataset(config.test_path, config.pad_size)
    #'THUCNews/data/test.txt'
    #test = [
    # ([3, 410, 2, 300, ], label, seq_len)
    # ]

    return vocab, train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        #DatasetIterater(train_data, 128, 'cpu')
        self.batch_size = batch_size#128
        self.batches = batches#train_data
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    #iter = DatasetIterater(train_data, 128, 'cpu')
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    #tensorboardX.SummaryWriter
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
        # for (x, seq_len), y in train_iter:
        # for i, ((x, seq_len), y) in enumerate(train_iter):
            outputs = model(trains)#self.embedding(trains[0])
            #(batch_size, num_classes)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:#config.require_improvement = 1000
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:#dev_iter
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    
    config = Config(dataset, embedding)
    setup_seed(config.seed)
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)#args.word = False

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = Model(config).to(config.device)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
