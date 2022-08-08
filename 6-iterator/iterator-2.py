import torch

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

# {'源': 0, '珊': 1, '爱': 2, '我': 3, '是': 4}
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
for (x, seq_len), y in it:
    print(x)
    print(seq_len)
    print(y)
    input()