import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from models.position_embedding import *


class I3DTokenizer(nn.Module):
    def __init__(self, i3d_dim=1024, embed_dim=128, n_timesteps=8):
        super().__init__()
        self.i3d_dim = i3d_dim
        self.embed_dim = embed_dim
        self.n_timesteps = n_timesteps

        self.proj = nn.Linear(self.i3d_dim, self.embed_dim)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_timesteps + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, x, batch_first=False):
        """
        params:
            - x: [batch_size, n_timesteps, i3d_dim] tensor of features extracted from video using i3d
            - batch_first: if True, output will be of form [batch_size, n_patches, embed_dim], else
              [n_patches, batch_size, embed_dim]
        """
        batch_size = x.shape[0]
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        p = self.pos_embedding.repeat(batch_size, 1, 1)

        z = torch.cat((cls_token, self.proj(x)), dim=1) + p

        if not batch_first:
            z = z.transpose(1, 0)

        return z


class ASTTokenizer(nn.Module):
    def __init__(self, patch_size=16, embed_dim=128, n_timesteps=1000, freq_dim=128):
        super().__init__()
        self.p = patch_size
        self.embed_dim = embed_dim
        self.n_timesteps = n_timesteps
        self.freq_dim = freq_dim
        assert self.freq_dim % self.p == 0
        self.n_patches = (n_timesteps // patch_size) * (freq_dim // patch_size)

        self.proj = nn.Conv2d(
            1, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, x, batch_first=False):
        batch_size = x.shape[0]
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        p = self.pos_embedding.repeat(batch_size, 1, 1)

        x = x.unsqueeze(1)
        z = self.proj(x).flatten(2).transpose(2, 1)
        z = torch.cat((cls_token, z), dim=1) + p

        if not batch_first:
            z = z.transpose(1, 0)
        return z


class TextTokenizer(nn.Module):
    """
    For use with mosei with pre-extracted text features
    NOTE: ASSUMES BATCH FIRST INPUT FOR MOSEI
    """

    def __init__(self, seq_len=50, input_dim=300, embed_dim=128, dropout=0.0):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.dropout = dropout


        self.proj = nn.Linear(self.input_dim, self.embed_dim)

        # self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.pos_embedding = nn.Parameter(torch.randn(
            1, seq_len + 1, self.embed_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, x, batch_first=False):
        """ Batch x seq len x feature """

        # logging.debug(['text input dims', x.shape])

        batch_size = x.shape[0]
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        p = self.pos_embedding.repeat(batch_size, 1, 1)

        z = self.proj(x)
        z = F.dropout(torch.cat((cls_token, z), dim=1) + p, p=self.dropout, training=self.training)

        # logging.debug([' z text input dims', z.shape])
        if not batch_first:
            z = z.transpose(1, 0)
        return z


class FeatureTokenizer(nn.Module):
    """
    For use with mosei with pre-extracted audio/visual features
    NOTE: ASSUMES BATCH FIRST INPUT FOR MOSEI
    """

    def __init__(self, seq_len=500, input_dim=300, embed_dim=128, dropout=0.0):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.dropout = dropout

        self.proj = nn.Linear(self.input_dim, self.embed_dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(
            1, seq_len + 1, self.embed_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, x, batch_first=False):
        """ Batch x seq len x feature """

        batch_size = x.shape[0]
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        p = self.pos_embedding.repeat(batch_size, 1, 1)

        z = self.proj(x)
        z = F.dropout(torch.cat((cls_token, z), dim=1) + p, p=self.dropout, training=self.training)

        if not batch_first:
            z = z.transpose(1, 0)
        return z


class ZeroesTokenizer(nn.Module):
    """
    Returns vector of zeroes
    """
    def forward(self, x, batch_first=False):
        """ Batch x seq len x feature """

        batch_size = x.shape[0]
        z = torch.zeros(batch_size, 1, self.embed_dim)
        
        return z