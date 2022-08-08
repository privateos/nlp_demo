#len1
#len2
#1 + len1 + 1 + len2 + 1
#(batch_size, 1 + len1 + 1 + len2 + 1)
#size = 1 + len1 + 1 + len2 + 1
#(batch_size, size, d_model)
#encoders
#(batch_size, 1, d_model)--->(batch_size, d_model)-->(batch_size, 2)

#mask_size = 1
#(batch_size, mask_size, d_model)-->(batch_size, mask_size, vocab_size)
#15%
#[CLS]我的狗很可爱[SEP]企鹅不擅长飞行[SEP]
#      [MASK]

#[CLS]我的[MASK]很可爱[SEP][MASK]不擅长飞行[SEP]
#word_list = [1,2,3,4,6,7, 0, 100]
#seg_list = [0, 0, 0, 0, 1, 1, 1, 1]

#15%
#80%:[MASK]
#10%:猪
#10%:狗

#TransformerEmbedding = Embedding + PositionalEncoding
#BertEmbedding = TokenEmbedding + PositionalEmbedding + SegmentEmbedding

# token_embedding = nn.Embedding(vocab_size, d_model)
# postional_embedding = nn.Embedding(maxlen, d_model)
# segment_embedding = nn.Embedding(2, d_model)
# token_embedding(word_list) + postional_embedding() + segment_embedding(seg_list)
import torch.utils.data as Data
import torch

class Dataset(Data.Dataset):
    def __init__(self) -> None:
        super(Dataset, self).__init__()
        self.count = 0

    def __len__(self):
        return 100

    def __getitem__(self, index):
        print(index, type(index))
        self.count += 1
        if self.count%2 == 1:
            return [[2,3,4],[1,2,5,9]]
        else:
            return torch.randn((2, 4))

def collate_fn(lists):
    print('line 21')
    print(lists)
    for e in lists:
        print(type(e), e)
    print('line 25')
    input()

dataset = Dataset()
dataloader = Data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)#cat  
for _ in dataloader:
    input();print(_.shape);input()