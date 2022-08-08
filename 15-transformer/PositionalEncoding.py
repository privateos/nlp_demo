import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = []
        for t in range(max_len):
            line = []
            for i in range(d_model):
                if i%2 == 0:
                    wk = 1.0/math.pow(10000, i/d_model)
                    v = math.sin(wk*t)
                    line.append(v)
                else:
                    wk = 1.0/math.pow(10000, (i - 1)/d_model)
                    v = math.cos(wk*t)
                    line.append(v)
            pe.append(line)
        #pe.shape = (max_len, d_model)
        pe = torch.from_numpy(pe).float()

        self.pe = nn.parameter.Parameter(pe, requires_grad=False)
    def forward(self, x):
        #x.shape = (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        #pex.shape = (seq_len, d_model)
        pex = self.pe[0:seq_len, :]
        x = x + pex
        return self.dropout(x)
