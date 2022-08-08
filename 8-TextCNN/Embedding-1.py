import torch
import torch.nn as nn
import numpy as np
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

vocab_dict = {'源': 0, '珊': 1, '爱': 2, '我': 3, '是': 4, '<PAD>':5}

num_embeddings = len(vocab_dict)#6
embedding_dim = 3
x = np.arange(0, num_embeddings*embedding_dim, dtype=np.float32)
x = np.reshape(x, (num_embeddings, embedding_dim))
print(x);input()
x = torch.FloatTensor(x)
embedding = nn.Embedding.from_pretrained(x, freeze=False)
print(embedding.weight);input()



# ([0, 0, 2, 1, 1, 5, 5], 0, 5),#(x, y, seq_len) '源源爱珊珊<pad><pad>'
# ([1, 1, 2, 0, 0, 5, 5], 0, 5), #'珊珊爱源源<pad><pad>'
x = torch.LongTensor(#(2, 7)
    [
        [0, 0, 2, 1, 1, 5, 5],
        [1, 1, 2, 0, 0, 5, 5]
    ]
)
print('x.shape='+str(x.shape))
ex = embedding(x)#(2, 7, 3)
print(ex.shape)
print(ex)