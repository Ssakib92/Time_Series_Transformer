# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 00:12:46 2022

@author: Sadman Sakib
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_seq_len=200):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()

        two_i = torch.arange(0, d_model, step=2).float()
        div_term = torch.pow(10000, (two_i/torch.Tensor([d_model]))).float()
        pe[:, 0::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # shape(x) : (B, seq_len, D)

        one_batch_pe: torch.Tensor = self.pe[:, :x.shape[1]].detach()
        repeated_pe = one_batch_pe.repeat([x.shape[0], 1, 1]).detach()
        x = x.add(repeated_pe)
        # shape(x) : (B, seq_len, D)

        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads=8, dropout=0.2):
        super().__init__()
        assert d_model % heads == 0

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.Q_linear = nn.Linear(d_model, d_model)
        self.V_linear = nn.Linear(d_model, d_model)
        self.K_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None, dropout=None):
        """
        # shape(Q) = (B, head, len(Q), D/head)
        # shape(K, V) = (B, head, len(KV), D/head)
        """

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        #(B, head, len(Q), D/head) dot (B, head, D/head, len(KV)) -> (B, head, len(Q), len(KV))

        if mask is not None:
            #mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)

        if dropout is not None:
            weights = dropout(weights)

        output = torch.matmul(weights, V)
        #(B, head, len(Q), len(KV) dot (B, head, len(KV), D/head)) -> (B, head, len(Q), D/head))
        # shape(output) : (B, head, len(Q), D/head)

        return output

    def forward(self, pre_Q, pre_K, pre_V, mask=None):
        """
        # shape(Q) = (B, seq_len, D)
        (if in encoder, seq_len = src_seq_len; if in decoder, seq_len = trg_seq_len)
        # shape(K, V) = (B, seq_len, D) (always SRC_seq_len unless in masked-multihead-attention)
        mask: (B, 1, 1, seq_len)
        """

        batch_size = pre_Q.size(0)

        Q = self.Q_linear(pre_Q)
        K = self.K_linear(pre_K)
        V = self.V_linear(pre_V)
        # shape(Q,K,V) : (B, seq_len, D)

        # spliting into number of heads
        Q = Q.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        # shape(Q,K,V) : (B, head, seq_len, d_k)
        # d_k = D / head

        # calculate attention
        scores = self.scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        # shape(scores) : (B, head, len(Q), D/head)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        output = self.out(concat)
        # shape(output) : (B, seq_len, D)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=128, dropout=0.2):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # shape(x) : (B, seq_len, D)

        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        # shape(x) : (B, seq_len, D)

        return x


class ResidualNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6, dropout=0.2):
        super().__init__()

        self.size = d_model

        # alpha,bias are two learnable parameters
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):

        x = x + residual
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

        return self.dropout(norm)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff=128, dropout=0.2):
        super().__init__()

        self.norm_1 = ResidualNorm(d_model, dropout)
        self.norm_2 = ResidualNorm(d_model, dropout)
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        # shape(x) : (B, seq_len, D)

        score = self.attn(x, x, x, mask)
        norm1 = self.norm_1(score, x)
        ff = self.ff(norm1)
        norm2 = self.norm_2(ff, norm1)
        # shape(norm2) : (B, seq_len, D)

        return norm2


class Encoder(nn.Module):
    def __init__(self, src_size, d_model, heads, num_layers, d_ff=128, dropout=0.2):
        super().__init__()
        # src_size : input features of the source time series

        self.num_layers = num_layers

        self.en_input = nn.Linear(src_size, d_model)

        self.pe = PositionalEncoder(d_model)
        self.encoders = nn.ModuleList([EncoderLayer(
            d_model, heads, d_ff, dropout
        ) for layer in range(num_layers)])

    def forward(self, x, mask=None):
        # shape(x) : (B, src_seq_len, D)

        en_inputs = self.en_input(x)
        encoding = self.pe(en_inputs)

        for encoder in self.encoders:
            encoding = encoder(encoding, mask)
        # shape(encoding) : (B, src_seq_len, D)

        return encoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff=128, dropout=0.2):
        super().__init__()

        self.norm_1 = ResidualNorm(d_model, dropout)
        self.norm_2 = ResidualNorm(d_model, dropout)
        self.norm_3 = ResidualNorm(d_model, dropout)

        self.masked_attn = MultiHeadAttention(d_model, heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, heads, dropout)

        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, encoder_outputs, trg_mask, src_mask):
        """
        shape(x) = (B, trg_seq_len, D)
        shape(encoder_outputs) = (B, trg_seq_len, D)
        """

        masked_score = self.masked_attn(x, x, x, mask=trg_mask)
        # shape(masked_score) = (B, trg_seq_len, D)

        norm1 = self.norm_1(masked_score, x)
        # shape(norm1) = (B, trg_seq_len, D)

        enc_dec_score = self.enc_dec_attn(norm1, encoder_outputs,
                                          encoder_outputs, mask=src_mask)
        # shape(enc_dec_mha) = (B, trg_seq_len, D)

        norm2 = self.norm_2(enc_dec_score, norm1)
        # shape(norm2) = (B, trg_seq_len, D)

        ff = self.ff(norm2)
        norm3 = self.norm_3(ff, norm2)
        # shape(norm3) = (B, trg_seq_len, D)

        return norm3


class Decoder(nn.Module):
    def __init__(self,
                 trg_size,
                 d_model,
                 heads,
                 num_layers,
                 d_ff=16,
                 dropout=0.2):
        super().__init__()
        # trg_size : features of the target output

        self.de_input = nn.Linear(trg_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.dropout = nn.Dropout(dropout)

        self.decoders = nn.ModuleList([DecoderLayer(
            d_model,
            heads,
            d_ff,
            dropout,
        ) for layer in range(num_layers)])

    def forward(self, x, encoder_output, trg_mask, src_mask=None):
        # shape(x) = (B, trg_seq_len, D)

        de_inputs = self.de_input(x.unsqueeze(-1))
        decoding = self.pe(de_inputs)
        # shape(decoding) = (B, trg_seq_len, D)

        for decoder in self.decoders:
            decoding = decoder(decoding, encoder_output, trg_mask, src_mask)
            # shape(decoding) = (B, trg_seq_len, D)

        return decoding


class TS_Transformer(nn.Module):
    def __init__(self, src_size, trg_size, d_model,
                 heads, num_layers, trg_feature, d_ff=16, dropout=0.2):
        super().__init__()

        self.heads = heads

        self.encoder = Encoder(src_size, d_model,
                               heads, num_layers, d_ff, dropout)
        self.decoder = Decoder(trg_size, d_model,
                               heads, num_layers, d_ff, dropout)

        self.linear_layer = nn.Linear(d_model, trg_feature)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_masks(self, size):

        mask = torch.triu(torch.ones(size, size)).transpose(
            0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, src, trg):
        # shape(src) = (B, src_seq_len, D)
        # shape(trg) = (B, trg_seq_len, D)

        trg_mask = self.create_masks(trg.shape[1])
        # shape(trg_mask) = (B, 1, trg_seq_len, trg_seq_len)

        encoder_outputs = self.encoder(src, mask=None)
        # shape(encoder_outputs) = (B, src_seq_len, D)

        decoder_outputs = self.decoder(
            trg, encoder_outputs, trg_mask, src_mask=None)
        # shape(decoder_outputs) = (B, trg_seq_len, D)

        out = self.linear_layer(decoder_outputs)
        # shape(out) : (B, trg_seq_len, trg_feature)

        return out.squeeze(-1)
    
    def encode(self, src, mask = None):
        encoder_outputs = self.encoder(src, mask = None)
        return encoder_outputs
    
    def decode(self, trg, encoder_outputs, trg_mask, src_mask = None):
        
        decoder_outputs = self.decoder(trg, encoder_outputs, trg_mask, src_mask = None)
        out = self.linear_layer(decoder_outputs)
        # shape(out) : (B, trg_seq_len, trg_feature)
        
        return out.squeeze(-1) 
