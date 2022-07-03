# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):
    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,activation)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_encoder_layers,None)
        self.embeddings = DecoderEmbeddings(config)
        decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,dropout,activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer,num_decoder_layers,decoder_norm)
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, src, tgt, tgt_mask): 
        src = src.permute(1,0,2) # [S,N,E]

        # 获取tgt [T,N,E]
        tgt = self.embeddings(tgt).permute(1, 0, 2)  # 对caption进行普通embedding, 并且加上位置编码

        # 获取memory
        memory = self.encoder.forward(src,src_key_padding_mask=None)
        hs = self.decoder.forward(tgt,
                            memory,
                            memory_key_padding_mask=None,
                            tgt_key_padding_mask=tgt_mask,
                            tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device) 
            )
        return hs


class DecoderEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x): # x: [32,128]  text
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)  
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings




def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask
"""
mask = [
	[0.,-inf,-inf],
	[0.,  0.,-inf],
	[0.,  0.,   0]
]

"""

def build_transformer(config):
    return Transformer(
        config,
        d_model=config.hidden_dim,
        dropout=config.dropout,
        nhead=config.nheads,
        dim_feedforward=config.dim_feedforward,
        num_encoder_layers=config.enc_layers,
        num_decoder_layers=config.dec_layers
    )
