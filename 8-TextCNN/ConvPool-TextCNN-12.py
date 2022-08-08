import torch
import torch.nn as nn
import torch.nn.functional as F
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
        #datas = [
        # ([0, 0, 2, 1, 1, 5, 5], 0, 5),#(x, y, seq_len) '源源爱珊珊<pad><pad>'
        # ([1, 1, 2, 0, 0, 5, 5], 0, 5), #'珊珊爱源源<pad><pad>'
        # ]
        #x = [
        # [0, 0, 2, 1, 1, 5, 5],
        # [1, 1, 2, 0, 0, 5, 5]
        # ]
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)#long int
        #y = [0, 0]
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        #seq_len = [5, 5]
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

#(batch_size, pad_size, embedding_dim)--->(batch_size, out_channels)
class ConvPool(nn.Module):
    #n-gram
    def __init__(self, pad_size, embedding_dim, out_channels, n):
        super(ConvPool, self).__init__()
        
        #(batch_size, 1, pad_size, embeddimg_dim)
        self.conv2d = nn.Conv2d(1, out_channels=out_channels, kernel_size=(n, embedding_dim))
        #(batch_size, out_channels, pad_size - n + 1, 1)


        #(batch_size, out_channels, pad_size - n + 1)
        self.pool = nn.MaxPool1d(kernel_size=pad_size - n + 1)
        #(batch_size, out_channels, 1)

    
    def forward(self, x):
        #x.shape = (batch_size, pad_size, embedding_dim)

        #x.shape = (batch_size, 1, pad_size, embedding_dim)
        x = torch.unsqueeze(x, dim=1)

        #conv2d.shape = (batch_size, out_channels, pad_size - n + 1, 1)
        conv2d = F.relu(self.conv2d(x))

        #x.shape = (batch_size, out_channels, pad_size - n + 1)
        x = torch.squeeze(conv2d, dim=3)

        #(batch_size, out_channels, 1)
        pool = self.pool(x)

        #result.shape = (batch_size, out_channels)
        result = torch.squeeze(pool, dim=2)
        return result

class TextCNN(nn.Module):
    def __init__(self,
        n_vocab,#词表大小
        embedding_dim,
        pad_size,
        num_classes = 10,
        grams=(2, 3, 4),
        num_filters=50,#out_channels
        ):
        super(TextCNN, self).__init__()

        #(batch_size, pad_size)
        self.embedding = nn.Embedding(n_vocab, embedding_dim)
        #(batch_size, pad_size, embedding_dim)

        layers = []
        for n in grams:
            conv_pool = ConvPool(
                pad_size=pad_size,
                embedding_dim=embedding_dim,
                out_channels=num_filters,
                n = n
                )
            layers.append(conv_pool)

        
        #(batch_size, pad_size, embedding_dim)
        self.grams = nn.ModuleList(layers)
        #(batch_size, num_filters)*len(grams)


        self.fc = nn.Linear(num_filters*len(grams), num_classes)

    def forward(self, x):
        #x.shape = (batch_size, pad_size)

        #embedding.shape = (batch_size, pad_size, embedding_dim)
        embedding = self.embedding(x)

        outs = []
        for gram in self.grams:
            #out.shape = (batch_size, num_filters)
            out = gram(embedding)
            outs.append(out)
        
        #grams.shape = (batch_size, num_filters*3)
        grams = torch.cat(outs, dim=1)

        #result.shape = (batch_size, num_classes)
        result = self.fc(grams)
        return result




vocab_dict = {'源': 0, '珊': 1, '爱': 2, '我': 3, '是': 4, '<PAD>':5}
batches = [
    ([0, 0, 2, 1, 1, 5, 5], 0, 5),#(x, y, seq_len) '源源爱珊珊<pad><pad>'
    ([1, 1, 2, 0, 0, 5, 5], 0, 5), #'珊珊爱源源<pad><pad>'
    ([3, 4, 0, 0, 5, 5, 5], 1, 4),  #'我是源源<pad><pad><pad>'
    ([3, 4, 1, 1, 5, 5, 5], 1, 4),#'我是珊珊<pad><pad><pad>'
    ([1, 2, 0, 5, 5, 5, 5], 0, 3),#'珊爱源<pad><pad><pad><pad>'
]
batch_size = 2
device = 'cpu'
it = DatasetIterater(batches, batch_size, device)

embedding_dim = 23
pad_size = 7
text_cnn = TextCNN(
    n_vocab=len(vocab_dict),
    embedding_dim=embedding_dim,
    pad_size=pad_size,
    num_classes=10,
    grams=(2,3,4,5),
    num_filters=13
)

for (x, seq_len), y in it:
    print('x = ',x)
    pred_y = text_cnn(x)
    print('pred_y = ' ,pred_y)
    print('pred_y.shape = ', pred_y.shape)
    input()