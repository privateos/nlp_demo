import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embedding, d_projection, sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()
        self.embedding = nn.Embedding(n_token, d_embedding, sparse=sample_softmax)
        self.projection = None
        if d_embedding != d_projection:
            self.projection = nn.Linear(d_embedding, d_projection, bias=False)
    
    def forward(self, x):
        #x.shape = (batch_size, seq_len)
        #embedding.shape = (batch_size, seq_len, d_embedding)
        embedding = self.embedding(x)
        
        if self.projection is None:return embedding
        
        #return value.shape = (batch_size, seq_len, d_projection)
        return self.projection(embedding)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_embedding):
        super(PositionalEmbedding, self).__init__()
        inv_freq = 1.0/(
            10000**(  torch.arange(0.0, d_embedding, 2.0)/d_embedding    )
            )
        #self.param.shape = (1, d_embedding//2)
        self.param = nn.parameter.Parameter(torch.unsqueeze(inv_freq, 0), requires_grad=False)
    
    def forward(self, position_seq):
        #position_seq = [seq_len - 1, seq_len-2, ..., 0]
        #position_seq.shape = (seq_len, )
        #position_seq_unsqueeze.shape = (seq_len, 1)
        position_seq_unsqueeze = torch.unsqueeze(position_seq, 1)

        #pp.shape = (seq_len, d_embedding//2)
        pp = torch.matmul(position_seq_unsqueeze, self.param)
        pp_sin = torch.sin(pp)
        pp_cos = torch.cos(pp)

        #position_embedding.shape = (seq_len, d_embedding)
        position_embedding = torch.cat([pp_sin, pp_cos], dim=1)

        #position_embedding_unsqueeze.shape = (1, seq_len, d_embedding)
        position_embedding_unsqueeze = torch.unsqueeze(position_embedding, 0)
        return position_embedding_unsqueeze

class RelativePartialLearnableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head, pre_norm=False, drop_attention=0.0, dropout=0.0):
        super(RelativePartialLearnableMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.pre_norm = pre_norm
        self.scale = 1.0/(d_head**0.5)
        self.attention_drop = nn.Dropout(drop_attention)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv_net = nn.Linear(d_model, 3*n_head*d_head, bias=False)
        self.pe_net = nn.Linear(d_model, n_head*d_head, bias=False)
        self.drop = nn.Dropout(dropout)

        self.output_net = nn.Linear(n_head*d_head, d_model, bias=False)

    
    def relative_shift(self, BD):
        #BD.shape = (batch_size, n_head, q_len, k_len)
        batch_size, n_head, q_len, k_len = BD.size()
        device = BD.device
        dtype = BD.dtype

        #zero_pad.shape = (batch_size, n_head, q_len, 1)
        zero_pad = torch.zeros((batch_size, n_head, q_len, 1), dtype=dtype, device=device)

        #BD_padded.shape = (batch_size, n_head, q_len, 1 + k_len)
        BD_padded = torch.cat([zero_pad, BD], dim=3)

        #BD_reshaped.shape = (batch_size, n_head, 1 + k_len, q_len)
        BD_reshaped = torch.reshape(BD_padded, (batch_size, n_head, 1 + k_len, q_len))

        #BD_sliced.shape = (batch_size, n_head, k_len, q_len)
        BD_sliced = BD_reshaped[:, :, 1:, :]

        #BD_sliced_reshape.shape = (batch_size, n_head, q_len, k_len)
        BD_sliced_reshape = torch.reshape(BD_sliced, (batch_size, n_head, q_len, k_len))

        return BD_sliced_reshape
    #hidden.shape = (batch_size, x_len, d_model)
    #position_embedding.shape = (1, k_len, d_model)
    #u,v.shape = (n_head, d_head)
    #mask.shape = (q_len, k_len)
    #memory.shape = (batch_size, memory_length=m_len, d_model)
    def forward(self, hidden, position_embedding, u, v, mask=None, memory=None):
        batch_size = hidden.size(0)
        k_len = position_embedding.size(0)
        q_len = hidden.size(1)
        d_head = self.d_head
        n_head = self.n_head

        if memory is not None:
            #mh.shape = (batch_size, x_len + m_len=k_len, d_model)
            mh = torch.cat([memory, hidden], dim=1)

            #qkv_heads.shape = (batch_size, k_len, 3*n_head*d_head)
            if self.pre_norm:
                qkv_heads = self.qkv_net(self.layer_norm(mh))
            else:
                qkv_heads = self.qkv_net(mh)
            #Q,K,V.shape = (batch_size, k_len, n_head*d_head)
            Q, K, V = torch.chunk(qkv_heads, 3, dim=-1)

            #Q.shape = (batch_size, q_len, n_head*d_head)
            Q = Q[:, -q_len:, :]
        else:
            #qkv_heads.shape = (batch_size, q_len, 3*n_head*d_head)
            if self.pre_norm:
                qkv_heads = self.qkv_net(self.layer_norm(hidden))
            else:
                qkv_heads = self.qkv_net(hidden)
            
            #Q, K, V.shape = (batch_size, q_len, n_head*d_head)
            Q, K, V = torch.chunk(qkv_heads, 3, dim=-1)
        #pe.shape = (batch_size, k_len, n_head*d_head)
        pe = self.pe_net(position_embedding)

        

        Q = torch.reshape(Q, (batch_size, q_len, n_head, d_head))
        K = torch.reshape(K, (batch_size, k_len, n_head, d_head))
        V = torch.reshape(V, (batch_size, k_len, n_head, d_head))
        pe = torch.reshape(pe, (batch_size, k_len, n_head, d_head))


        #Qu.shape = (batch_size, q_len, n_head, d_head)
        #AC.shape = (batch_size, n_head, q_len, d_head)
        #           @(batch_size, n_head, d_head, k_len)
        #         = (batch_size, n_head, q_len, k_len)
        Qu = Q + u
        AC = torch.matmul(
                torch.transpose(Qu, 1, 2),#(batch_size, n_head, q_len, d_head)
                torch.permute(K, (0, 2, 3, 1))#(batch_size, n_head, d_head, k_len)
            )


        #BD.shape = (batch_size, n_head, q_len, k_len)
        Qv = Q + v
        BD =  torch.matmul(
                torch.transpose(Qv, 1, 2),#(batch_size, n_head, q_len, d_head)
                torch.permute(pe, (0, 2, 3, 1))#(batch_size, n_head, d_head, k_len)
            )
        self.relative_shift(BD)
        #BD = self._rel_shift(BD)################################################
        attention_score = (AC + BD)*self.scale

        #mask.shape = (q_len, k_len)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask, -math.inf)
        #attention_probability.shape = (batch_size, n_head, q_len, k_len)
        attention_probability = F.softmax(attention_score, dim=3)
        attention_probability = self.attention_drop(attention_probability)


        #V.shape = (batch_size, k_len, n_head, d_head)
        #attention_vector.shape = (batch_size, n_head, q_len, d_head)
        attention_vector = torch.matmul(
            attention_probability,#(batch_size, n_head, q_len, k_len)
            torch.transpose(V, 1, 2)#(batch_size, n_head, k_len, d_head)
        )

        
        #attention_vector.shape = (batch_size, q_len, n_head, d_head)
        attention_vector = torch.transpose(attention_vector, 1, 2)


        #attention_vector.shape = (batch_size, q_len, n_head*d_head)
        attention_vector = torch.reshape(attention_vector, (batch_size, q_len, n_head*d_head))


        #attention_out.shape = (batch_size, q_len, d_model)
        attention_out = self.output_net(attention_vector)
        attention_out = self.drop(attention_out)


        if self.pre_norm:
            #hidden.shape = (batch_size, x_len=q_len, d_model)
            #atttention_out.shape = (batch_size, x_len=q_len, d_model)
            #output.shape = (batch_size, x_len=q_len, d_model)
            output = hidden + attention_out
        else:
            output = self.layer_norm(hidden + attention_out)

        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_norm=False):
        super(PositionwiseFeedForward, self).__init__()

        self.core_net = nn.Sequential(
            #(batch_size, x_len=q_len, d_model)
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            #(batch_size, x_len=q_len, d_inner),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
            #(batch_size, x_len=q_len, d_model)
        )
        self.pre_norm = pre_norm
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
    
    def forward(self, x):
        #x.shape = (batch_size, x_len=q_len, d_model)
        if self.pre_norm:
            core_out = self.core_net(self.layer_norm(x))

            output = core_out + x
        else:
            core_out = self.core_net(x)
            output = self.layer_norm(x + core_out)

        return output

class RelativePartialLearnableDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_inner, 
        pre_norm=False, drop_attention=0.0, dropout=0.0
    ):
        super(RelativePartialLearnableDecoderLayer, self).__init__()
        self.attention = RelativePartialLearnableMultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            d_head=d_head,
            pre_norm=pre_norm,
            drop_attention=drop_attention,
            dropout=dropout
        )

        self.pff = PositionwiseFeedForward(
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            pre_norm=pre_norm
        )
    
    #hidden.shape = (batch_size, x_len=q_len, d_model)
    #position_embedding.shape = (1, k_len, d_model)
    #u,v.shape = (n_head, d_head)
    #mask.shape = (q_len, k_len)
    #memory.shape = (batch_size, memory_length, d_model)
    def forward(self, hidden, position_embedding, u, v, mask=None, memory=None):
        #output.shape = (batch_size, x_len, d_model)
        output = self.attention(
            hidden, 
            position_embedding,
            u,
            v,
            mask,
            memory
        )

        #output.shape = (batch_size, x_len=q_len, d_model)
        output = self.pff(output)
        return output

class TransformerXL(nn.Module):
    def __init__(self,
        n_token, n_layer, 
        n_head, d_head, d_model, d_inner,
        memory_length,
        # target_length,
        pre_norm=False,
        d_embedding=None,
        dropout=0.0,
        drop_attention=0.0,
        ):
        super(TransformerXL, self).__init__()
        if d_embedding is None:
            d_embedding = d_model
        self.n_layer = n_layer
        self.memory_length = memory_length
        self.u = nn.Parameter(torch.Tensor(n_head, d_head), requires_grad=True)
        self.v = nn.Parameter(torch.Tensor(n_head, d_head), requires_grad=True)

        self.word_embedding = AdaptiveEmbedding(n_token=n_token, d_embedding=d_embedding, d_projection=d_model)
        self.postional_embedding = PositionalEmbedding(d_embedding=d_model)
        self.drop = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                RelativePartialLearnableDecoderLayer(
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    d_inner=d_inner,
                    pre_norm=pre_norm,
                    drop_attention=drop_attention,
                    dropout=dropout
                )
            )
        self.linear_for_classification = nn.Linear(d_model, n_token)
    
    def update_memory(self, hiddens, mems, q_len, m_len):
        with torch.no_grad():
            new_mems = []
            end_index = m_len + max(0, q_len)
            begin_index = max(0, end_index - self.memory_length)
            for i, hidden in enumerate(hiddens):
                #hidden.shape = (batch_size, q_len, d_model)
                if mems is None:
                    cat = hidden
                else:
                    #cat.shape = (batch_size, mem_len + q_len, d_model)
                    cat = torch.cat([mems[i], hidden], dim=1)
                new_mems.append(cat[:, begin_index:end_index, :, :].detach())
        return new_mems

    def _forward(self, X, mems):
        device = X.device
        dtype = X.dtype

        #X.shape = (batch_size, x_len)
        batch_size, x_len = X.size()
        q_len = x_len


        #word_embedding.shape = (batch_size, x_len, d_model)
        word_embedding = self.word_embedding(X)

        #m_len = memory_length
        m_len = 0 if mems is None else mems[0].size(1)
        k_len = q_len + m_len

        all_ones = torch.ones((q_len, k_len), dtype=torch.bool, device=device)
        attention_mask = torch.triu(
            all_ones, diagonal=1+m_len
        )

        


        #position_seq = [k_len - 1, k_len - 2, ..., 0]
        position_seq = torch.arange(
            k_len - 1, -1, -1.0, 
            device=device, dtype=dtype
        )
        #position_embedding.shappe = (1, k_len, d_model)
        position_embedding = self.postional_embedding(position_seq)


        #position_embedding.shappe = (1, k_len, d_model)
        #word_embedding.shape = (batch_size, x_len, d_model)
        position_embedding = self.drop(position_embedding)
        word_embedding = self.drop(word_embedding)

        hiddens = []
        hiddens.append(word_embedding)
        hidden_i = word_embedding
        u = self.u
        v = self.v
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]

            hidden_i = layer(
                #hidden_i.shape = (batch_size, x_len, d_model)
                #position_embedding.shape = (1, k_len, d_model)
                #u,v.shape = (n_head, d_head)
                #mask.shape = (q_len, k_len)
                #memory.shape = (batch_size, memory_length, d_model)
                hidden_i, 
                position_embedding,
                u,
                v,
                mask=attention_mask,
                memory=mems_i
            )
            hiddens.append(hidden_i)
        
        output = self.drop(hidden_i)
        new_mems = self.update_memory(hiddens, mems, m_len, q_len)

        #output.shape = (batch_size, q_len=x_len, d_model)
        return output, new_mems
    
    def forward(self, X, Y, mems=None):
        #0 < y_len <= x_len
        #X.shape = (batch_size, x_len)
        #Y.shape = (batch_size, y_len)
        
        # if mems is None:
        #     mems = []
        #     param = next(self.parameters())
        #     dtype = param.dtype
        #     device = param.device
        #     for i in range(self.n_layer + 1):
        #         empty = torch.empty(0, dtype=dtype, device=device)
        #         mems.append(empty)
        
        ############################################################
        #hidden.shape = (batch_size, x_len, d_model)
        hidden, new_mems = self._forward(X, mems)
        ############################################################

        #predict_hidden.shape = (batch_size, y_len, d_model)
        batch_size = Y.size(0)
        y_len = Y.size(1)
        predict_hidden = hidden[:, -y_len:, :]

        #predict_output.shape = (batch_size, y_len=target_len, n_token)
        predict_output = self.linear_for_classification(predict_hidden)
        
        #predict_output.shape = (batch_size*y_len, n_token)
        predict_output = torch.reshape(predict_output, (batch_size*y_len, -1))

        return predict_output, new_mems