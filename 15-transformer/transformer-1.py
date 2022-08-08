import torch
import torch.nn as nn
import math
import numpy as np

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, len_q]
    seq_k: [batch_size, len_k]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    #seq_q = torch.tensor(xx, dtype=torch.long)#Tensor
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    
    
    # eq(zero) is PAD token
    #pad_attn_mask.shape = (batch_size, seq_len)
    pad_attn_mask = seq_k.data.eq(0)#seq_k.data == 0
    
    
    # [batch_size, 1, len_k]
    pad_attn_mask = pad_attn_mask.unsqueeze(1)
    
    
    
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    #attn_shape = (batch_size, tgt_len, tgt_len)
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()#astype(np.uint8)
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #pe.shape = (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        #position.shape = (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        #d_model = 768
        #t1 = [0, 2, 4, 6, .., 766]
        #t1.shape = (d_model/2, )
        #div_term.shape (d_model/2, )
        t1 = torch.arange(0, d_model, 2).float()
        t0 = -math.log(10000.0) / d_model
        div_term = torch.exp(t1*t0)
        
        
        #position.shape = (max_len, 1)---->(max_len, d_model/2)
        #div_term.shape = (d_model/2, )--->(max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        #self.pe.shape = (max_len, d_model)
        self.pe = nn.parameter.Parameter(pe, required_grad=False)

    def forward(self, x):
        #x.shape = (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        #pex.shape = (seq_len, d_model)
        pex = self.pe[0:seq_len, :]
        x = x + pex
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn


#(batch_size, input_size)--->(batch_size, hidden_size)
#(batch_size, seq_len, input_size)--->(batch_size*seq_len, input_size)-->(batch_size*seq_len, hidden_size)-->(batch_size, seq_len, hidden_size)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, n_heads*d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads*d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads*d_v, bias=False)
        
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.sqrt_dk = math.sqrt(d_k)



    #res.shape = (batch_size, len_q, d_model) = (batch_size, tgt_len, d_model)
    #QK_softmax.shape = (batch_size, n_heads, len_q, len_k) = (batch_size, n_heads, tgt_len, src_len)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model] = (batch_size, tgt_len, d_model)
        input_K: [batch_size, len_k, d_model] = (batch_size, src_len, d_model)
        input_V: [batch_size, len_v(=len_k), d_model] = (batch_size, src_len, d_model)
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        batch_size, len_q, _ = input_Q.size()
        _, len_k, _ = input_K.size()
        _, len_v, _ = input_V.size()
        d_k = self.d_k
        d_v = self.d_v
        n_heads = self.n_heads

        #Q,K.shape = (batch_size, seq_len, n_heads*d_k)
        #V.shape = (batch_size, seq_len, n_heads*d_v)
        Q = self.W_Q(input_Q)
        K = self.W_K(input_K)
        V = self.W_V(input_V)
        

        #Q.shape = (batch_size, len_q, n_heads, d_k)
        #K.shape = (batch_size, len_k, n_heads, d_k)
        #V.shape = (batch_size, len_v, n_heads, d_v)
        Q = torch.reshape(Q, (batch_size, len_q, n_heads, d_k))
        K = torch.reshape(K, (batch_size, len_k, n_heads, d_k))
        V = torch.reshape(V, (batch_size, len_v, n_heads, d_v))
        
       
        #Q.shape = (batch_size, n_heads, len_q, d_k)
        #K.shape = (batch_size, n_heads, len_k, d_k)
        #V.shape = (batch_size, n_heads, len_v=len_k, d_v) 
        Q = torch.transpose(Q, 1, 2)
        K = torch.transpose(K, 1, 2)
        V = torch.transpose(V, 1, 1)
        
        #(batch_size, n_heads, len_q, d_k) @ (batch_size, n_heads, d_k, len_k)
        #QK.shape = (batch_size, n_heads, len_q, len_k) -->(len_q, len_k)
        QK = torch.matmul(Q, torch.transpose(K, 2, 3))/self.sqrt_dk
        
        
        #attn_mask.shape = [batch_size, seq_len, seq_len]
        #mask.shape = (batch_size, 1, seq_len, seq_len)
        mask = torch.unsqueeze(attn_mask, 1)
        
        #mask.shape = (batch_size, n_heads, len_q, len_k)
        mask = mask.expand(batch_size, n_heads, len_q, len_k)
        QK = QK.masked_fill_(mask, -1e9)
        
        QK_softmax = torch.softmax(QK, dim=3)
        
        
        #(batch_size, n_heads, len_q, len_k) @ (batch_size, n_heads, len_v=len_k, d_v)
        #QKV.shape = (batch_size, n_heads, len_q, d_v)
        QKV = torch.matmul(QK_softmax, V)
        
        #QKV_transpose.shape = (batch_size, len_q, n_heads, d_v)
        #QKV_reshape.shape = (batch_size, len_q, n_heads*d_v)
        QKV_transpose = torch.transpose(QKV, 1, 2)
        QKV_reshape = torch.reshape(QKV_transpose, (batch_size, len_q, n_heads*d_v))
        
        
        #result.shape = (batch_size, len_q, d_model)
        result = self.fc(QKV_reshape)
        
        res = self.layer_norm(result + input_Q)
        
        #res.shape = (batch_size, len_q, d_model)
        #QK_softmax.shape = (batch_size, n_heads, len_q, len_k)
        return res, QK_softmax
        

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            #(batch_size, src_len, d_model)
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            #(batch_size, src_len, d_ff)
            nn.Linear(d_ff, d_model, bias=False)
            #(batch_size, src_len, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs.shape = [batch_size, src_len, d_model], 
        # attn.shape = [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        
        #enc_outputs.shape = [batch_size, src_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs) 
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, n_layers, d_model, d_k, d_v, d_ff, n_heads):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        #enc_inputs.shape = (batch_size, src_len)
        
        # [batch_size, src_len, d_model]
        # [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs) 
        enc_outputs = self.pos_emb(enc_outputs) 
        
        
        # [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) 
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)


        #dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):

        #dec_inputs.shape = (batch_size, tgt_len, d_model)
        #enc_outputs.shape = (batch_size, src_len, d_model)
        #dec_self_attn_mask.shape = (batch_size, tgt_len, tgt_len)
        #dec_enc_attn_mask.shape = (batch_size, tgt_len, src_len)
        
   
   
        #dec_outputs.shape = (batch_size, tgt_len, d_model)
        #dec_self_attn.shape = (batch_size, n_heads, tgt_len, tgt_len)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        
        
        
        #param:
        #dec_outputs.shape = (batch_size, tgt_len, d_model) len_q = tgt_len
        #enc_outputs.shape = (batch_size, src_len, d_model) len_k = len_v = src_len
        
        #return:
        #dec_outputs.shape = (batch_size, tgt_len, d_model)
        #dec_enc_attn.shape = (batch_size, n_heads, tgt_len, src_len)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        
        
        #dec_outputs.shape = (batch_size, tgt_len, d_model)
        dec_outputs = self.pos_ffn(dec_outputs)
        
        
        #dec_outputs.shape = (batch_size, tgt_len, d_model)
        #dec_self_attn.shape = (batch_size, n_heads, tgt_len, tgt_len)
        #dec_enc_attn.shape = (batch_size, n_heads, tgt_len, src_len)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, n_layers, d_model, d_k, d_v, d_ff, n_heads):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        #dec_inputs.shape = (batch_size, tgt_len)
        #enc_inputs.shape = (batch_size, src_len)
        #enc_outputs.shape = (batch_size, src_len, d_model)
        
        #dec_outputs.shape = (batch_size, tgt_len, d_model)
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)
        
        
        #dec_self_attn_pad_mask.shape = [batch_size, tgt_len, tgt_len]
        #dec_self_attn_subsequence_mask.shape = (batch_size, tgt_len, tgt_len)
        #dec_self_attn_mask.shape = [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) 
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, 
                 d_model, d_k, d_v, d_ff, n_heads,
                 
                 src_vocab_size,
                 enc_n_layers,
                 
                 tgt_vocab_size, 
                 dec_n_layers,
        ):
        
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, enc_n_layers, d_model, d_k, d_v, d_ff, n_heads)
        self.decoder = Decoder(tgt_vocab_size, dec_n_layers, d_model, d_k, d_v, d_ff, n_heads)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        #enc_outputs.shape = (batch_size, src_len, d_model)
        #enc_self_attns = [xxx]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        
        
        #dec_inputs.shape = (batch_size, tgt_len)
        #enc_inputs.shape = (batch_size, src_len)
        #enc_outputs.shape = (batch_size, src_len, d_model)
        #dec_outputs.shape = (batch_size, tgt_len, d_model)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        
        
        #dec_logits.shape = (batch_size, tgt_len, tgt_vocab_size)
        dec_logits = self.projection(dec_outputs)
        
        tgt_vocab_size = dec_logits.size(2)
        
        #result.shape = (batch_size*tgt_len, tgt_vocab_size)
        result = torch.reshape(dec_logits, (-1, tgt_vocab_size))
        return result, enc_self_attns, dec_self_attns, dec_enc_attns
