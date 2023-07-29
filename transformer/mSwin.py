import os
import math
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.layers import PixelWiseDotProduct, miniSwinEncoder, PatchTransformerEncoder
# from layers import PixelWiseDotProduct, miniSwinEncoder, miniSwinEncoder2

class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps

class mSwin(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=8, layers=(2, 2), n_query_channels=8,
            window_size=7, heads=(4, 4), head_dim=8, downscaling_factors=(4, 2),
            dim_out=16, relative_pos_embedding=True, norm='linear'):
        super().__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.swintransformer = miniSwinEncoder(channels=in_channels,
                                                hidden_dim=hidden_dim,
                                                layers=layers,
                                                window_size=window_size,
                                                heads=heads,
                                                head_dim=head_dim,
                                                downscaling_factors=downscaling_factors, 
                                                relative_pos_embedding=relative_pos_embedding)
        self.dot_product_layer = PixelWiseDotProduct()
        self.conv = nn.Sequential(
                    nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8, bias=False),
                    nn.Conv2d(8, n_query_channels, 1, 1, 0, bias=False),
                    )
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))
    def forward(self, x1, x2):
        tgt = self.swintransformer(x2.clone())  # .shape = S, N, E
        n, c, h, w = tgt.size()
        # print(tgt.size())
        tgt = tgt.view(n, c, h*w).permute(2, 0, 1)
        # print(tgt.size())
    
        # x1 = self.conv3x3(x1)
        x1 = self.conv(x1)

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]
        # print(regression_head.size())
        # print(regression_head.size())

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 2, 0)
        range_attention_maps = self.dot_product_layer(x1, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps

