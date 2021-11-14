import torch
import torch.nn as nn
import math


# Positional Encoding Module
class PositionalEncoding(nn.Module):
    """
    Return input matrix with added positional encoding
    Replicated from "Attention Is All You Need"

    Input Shape:
        (len, batch_size, dim)
    Args:
        dim (int): dimensionality of embedding vector for each token
        max_len (int): max length of source sequence
    Output Shape:
        (len, batch_size, dim)
    """

    def __init__(self, dim, max_len):
        super(PositionalEncoding, self).__init__()
        self.pos_encode = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        _2i = torch.arange(0, dim, step=2).float()
        self.pos_encode[:, 0::2] = torch.sin(position / (10000 ** (_2i / dim)))  
        self.pos_encode[:, 1::2] = torch.cos(position / (10000 ** (_2i / dim)))
        # (max_len, 1) / (1, dim) --> (max_len, dim)
        self.pos_encode = self.pos_encode.unsqueeze(0).transpose(0, 1)  # (max_len, 1, dim)
        
    def forward(self, x):
        seq_len = x.size(0)
        
        # position matrix of (length, 1, dim) added for each batch
        output = x + self.pos_encode[:seq_len, :]  
        # (len, batch_size, dim) + (len, 1, dim) --> (len, batch_size, dim)
        return output


# Padding Mask
def padding_mask(seq_q, seq_k, pad_index):
    """
    create a boolen tensor as the mask where the positions of padding have the value of True

    Input Shape:
        seq_q: (batch_size, len_q)
        seq_k: (batch_size, len_k)
    Args:
        pad_index (int): index of padding in the vocabulary
    Output Shape:
        (batch_size, len_q, len_k)
    """

    len_q = seq_q.size()[1]
    len_k = seq_k.size()[1]

    pad_mask = seq_k.eq(pad_index)  # (batch_size, len_k)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
    # (batch_size, len_k) --> (batch_size, 1, len_k) --> (batch_size, len_q, len_k)
    return pad_mask


# Self-Attention Module
class ScaleDotProductAttention(nn.Module):
    """
    Perform attention calculation on matrices of Q, K, V transformed from token embedding
    Attention(Q,K,V) = softmax(QK^T/sqrt(dim_k))V

    Fill the position of padding with a very large negative number to avoid paying attention to the padding
    Since the scores will be close to 0 after the softmax layer

    Input Shape: 
        q: (batch_size, len_q, dim_q)
        k: (batch_size, len_k, dim_k)
        v: (batch_size, len_v, dim_v)
        attn_mask: (batch_size, len_q, len_k)
    Output Shape:
        v_weighted_sum: (batch_size, len_q, dim_v)
        attention_weights: (batch_size, len_q, len_k)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        batch_size, len_k, dim_k = k.size()

        attention_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(dim_k)  
        # (batch_size, len_q, dim_q) * (batch_size, dim_k, len_k) --> (batch_size, len_q, len_k)

        # apply padding mask
        if attn_mask is not None:
            attention_scores = attention_scores.masked_fill_(attn_mask, -1e9)

        attention_weights = self.softmax(attention_scores)
        v_weighted_sum = torch.bmm(attention_weights, v)  
        # (batch_size, len_q, len_k) * (batch_size, len_v, dim_v) --> (batch_size, len_q, dim_v) 
        # for len_k = len_v
        return v_weighted_sum, attention_weights


# Multi-Head Attention Module
class MultiHeadAttention(nn.Module):
    """
    Input Shape: 
        x: (batch_size, len, dim)
        attn_mask: (batch_size, len_q, len_k)
    Args:
        dim (int): dimensionality of embedding vector for each token
        n_head (int): number of heads in the multi-head attention module
        dropout_rate (float): dropout applied to the output before "Add & Norm", default 0.1
    Output Shape:
        output: (batch_size, len, dim)
        attention_weights_: (batch_size, n_head, len, len)
    """

    def __init__(self, dim, n_head, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.n_head = n_head
        self.dim_head = self.dim // self.n_head

        self.linear_q = nn.Linear(self.dim, self.dim_head * self.n_head)
        self.linear_k = nn.Linear(self.dim, self.dim_head * self.n_head)
        self.linear_v = nn.Linear(self.dim, self.dim_head * self.n_head)

        self.linear_out = nn.Linear(self.dim_head * self.n_head, self.dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, attn_mask=None):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        # (batch_size, len, dim) --> (batch_size, len, dim_head * n_head)

        batch_size = x.size()[0]
        len_ = x.size()[1]
        q_ = q.view(batch_size * self.n_head, len_, self.dim_head)
        k_ = k.view(batch_size * self.n_head, len_, self.dim_head)
        v_ = v.view(batch_size * self.n_head, len_, self.dim_head)
        # (batch_size, len, dim_head * n_head) --> (batch_size * n_head, len, dim_head)
        
        v_weighted_sum, attention_weights = ScaleDotProductAttention()(q_, k_, v_, attn_mask)
        # v_weighted_sum: (batch_size * n_head, len, dim_head)
        # attention_weights: (batch_size * n_head, len, len)

        v_weighted_sum_ = v_weighted_sum.view(batch_size, len_, self.dim_head * self.n_head)
        attention_weights_ = attention_weights.view(batch_size, self.n_head, len_, len_)

        output = self.linear_out(v_weighted_sum_)
        # (batch_size, len, dim_head * n_head) --> (batch_size, len, dim)
        output = self.dropout(output)
        return output, attention_weights_


# Layer Normalization
class LayerNormalization(nn.Module):
    """
    output = (gamma * (tensor - mean) / (std + eps)) + beta

    Input Shape: 
        (batch_size, len, dim)
    Args:
        dim (int): dimensionality of the layer output to normalize
        eps (float): epsilon to avoid zero in denominator, default 1e-6
    Output Shape: 
        (batch_size, len, dim)
    """

    def __init__(self, dim, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))  # register as parameter (requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (self.gamma * (x - mean) / (std + self.eps)) + self.beta
        return output


# Position-Wise Feed-Forward Module
class PositionWiseFeedForward(nn.Module):
    """
    Input Shape: 
        (batch_size, len, dim)
    Args:
        dim (int): dimensionality of embedding vector for each token
        hidden_dim (int): dimensionality of the inner layer of feed-forward network
        dropout_rate (float): dropout applied to the output before "Add & Norm", default 0.1
    Output Shape: 
        (batch_size, len, dim)
    """

    def __init__(self, dim, hidden_dim, dropout_rate=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.linear_1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear_2(output)
        return output
        
        
# Encoder Layer Module
class EncoderLayer(nn.Module):
    """
    Integrate all modules in the transformer encoder block

    Input Shape: 
        x: (batch_size, len, dim)
        attn_mask: (batch_size, len_q, len_k)
    Args:
        dim (int): dimensionality of embedding vector for each token
        ffn_hidden (int): dimensionality of the inner layer of feed-forward network
        n_head (int): number of heads in the multi-head attention module
        dropout_prob (float): dropout applied to the output before "Add & Norm", default 0.1
        eps (float): epsilon to avoid zero in denominator, default 1e-6
    Output Shape: 
        output: (batch_size, len, dim)
        attention_weights_: (batch_size, n_head, len, len)
    """

    def __init__(self, 
                 dim, 
                 ffn_hidden, 
                 n_head, 
                 drop_prob, 
                 eps):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(dim=dim, n_head=n_head, dropout_rate=drop_prob)
        self.norm_1 = LayerNormalization(dim=dim, eps=eps)
        self.dropout_1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(dim=dim, hidden_dim=ffn_hidden, dropout_rate=drop_prob)
        self.norm_2 = LayerNormalization(dim=dim, eps=eps)
        self.dropout_2 = nn.Dropout(p=drop_prob)

    def forward(self, x, attn_mask=None):
        x_res = x
        output, attention_weights_ = self.attention(x, attn_mask)
        output = self.norm_1(output + x_res)
        output = self.dropout_1(output)

        x_res = output
        output = self.ffn(output)
        output = self.norm_2(output + x_res)
        output = self.dropout_2(output)
        return output, attention_weights_


# Encoder Module
class Encoder(nn.Module):
    """
    Integrate input embedding, positional encoding, and encoder layers

    Input Shape:
        (batch_size, len)
    Args:
        dim (int): dimensionality of embedding vector for each token
        max_len (int): max length of source sequence
        vocab_size (int): size of the vocabulary to retrieve stored word embeddings
        pad_index (int): index of padding in the vocabulary
        num_layers (int): number of encoder layers
        ffn_hidden (int): dimensionality of the inner layer of feed-forward network
        n_head (int): number of heads in the multi-head attention module
        dropout_prob (float): dropout applied to the output before "Add & Norm", default 0.1
        eps (float): epsilon to avoid zero in denominator, default 1e-6
    Output Shape:
        output: (batch_size, len, dim)
        attentions: (n_layers, batch_size, n_head, len, len)
    """

    def __init__(self, 
                 dim, 
                 max_len, 
                 vocab_size,
                 pad_index, 
                 num_layers,
                 ffn_hidden, 
                 n_head, 
                 drop_prob, 
                 eps):
        super(Encoder, self).__init__()
        self.pad_index = pad_index
        self.n_head = n_head
        
        self.src_embedding = nn.Embedding(vocab_size, dim, self.pad_index)
        self.pos_embedding = PositionalEncoding(dim, max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(dim, ffn_hidden, n_head, drop_prob, eps) 
                                            for _ in range(num_layers)])

    def forward(self, inputs):
        output = self.src_embedding(inputs)  # (batch_size, len, dim)
        output = self.pos_embedding(output.transpose(0, 1)).transpose(0, 1)
        # (batch_size, len, dim) --> (len, batch_size, dim) --> (batch_size, len, dim)

        attn_mask = padding_mask(inputs, inputs, self.pad_index)  # (batch_size, len, len)
        attn_mask = attn_mask.repeat(self.n_head, 1, 1) # (batch_size * n_head, len, len)
        
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, attn_mask)
            attentions.append(attention)

        return output, attentions
