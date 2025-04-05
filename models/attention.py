# -*- coding: utf-8 -*-            
# @Author : Hao Wei
# @Time : 2025/4/5 16:15
import math
import torch
import torch.nn as nn
import numpy as np
import itertools


class Attention2D(nn.Module):
    def __init__(self,
                 seq_len=256,
                 d_model=512,
                 num_heads=8,
                 scale_factor=None,
                 dropout=0.0,
                 device=torch.device('cuda'),
                 use_relative_position=False):  
        super(Attention2D, self).__init__()

        self.seq_len = seq_len
        self.width = int(math.sqrt(seq_len))
        d = d_model // num_heads
        self.scale_factor = scale_factor or nn.Parameter(torch.tensor(1.0 / math.sqrt(d)))
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.device = device
        self.use_relative_position = use_relative_position  

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.sigma_projection = nn.Linear(d_model, 2 * num_heads)
        self.out_projection = nn.Linear(d_model, d_model)

        distances_x = np.load('distances/distances_x_{}.npy'.format(seq_len))
        distances_y = np.load('distances/distances_y_{}.npy'.format(seq_len))
        distances_x = torch.from_numpy(distances_x)
        distances_y = torch.from_numpy(distances_y)
        self.distances_x = distances_x.to(device)
        self.distances_y = distances_y.to(device)

        if use_relative_position:  
            
            points = list(itertools.product(range(self.width), range(self.width)))
            N = len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])

            
            self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
            self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    def forward(self, query, key, value, return_attention=True):
        B, L, _ = query.shape
        _, S, _ = key.shape

        if return_attention:
            sigma = self.sigma_projection(query).view(B, L, self.num_heads, -1)
        query = self.query_projection(query).view(B, L, self.num_heads, -1)
        key = self.key_projection(key).view(B, S, self.num_heads, -1)
        value = self.value_projection(value).view(B, S, self.num_heads, -1)

        scores = torch.einsum("blhe,bshe->bhls", query, key)

        
        if self.use_relative_position:
            scores = self.scale_factor * scores
            scores += self.attention_biases[:, self.attention_bias_idxs]
        else:
            scores = self.scale_factor * scores

        softmax_scores = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", softmax_scores, value)

        out = out.contiguous().view(B, L, -1)
        self.out_projection(out)

        if return_attention:
            sigma = sigma.transpose(1, 2)
            sigma = torch.sigmoid(sigma * 5) + 1e-5
            sigma = torch.pow(3, sigma) - 1

            sigma1 = sigma[:, :, :, 0]
            sigma2 = sigma[:, :, :, 1]
            sigma1 = sigma1.unsqueeze(-1).repeat(1, 1, 1, self.seq_len)
            sigma2 = sigma2.unsqueeze(-1).repeat(1, 1, 1, self.seq_len)

            distances_x = self.distances_x.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(
                self.device)
            distances_y = self.distances_y.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(
                self.device)

            target = 1.0 / (2 * math.pi * sigma1 * sigma2) * torch.exp(
                -distances_y / (2 * sigma1 ** 2) - distances_x / (2 * sigma2 ** 2))
            return out, softmax_scores, target
        else:
            return out, None, None
