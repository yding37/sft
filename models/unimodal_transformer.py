from numpy.lib.function_base import interp
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from models.bottleneck_transformer import *
from models.wrapper import *
import logging
import numpy as np

from utils.common import get_tb
from models.tokenizers import *


class UnimodalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed_dim = config.embed_dim
        self.dropout = config.dropout
        self.n_layers_pre_fusion = config.n_layers_pre_fusion
        self.n_layers_post_fusion = config.n_layers_post_fusion
        self.total_layers = config.n_layers_pre_fusion + config.n_layers_post_fusion

        self.modality = config.modality

        assert self.modality in ['rgb', 'spec', 'text']

        self.attns = nn.ModuleList()

        self.mlps = nn.ModuleList()

        for i in range(self.n_layers_pre_fusion):
            self.attns.append(nn.MultiheadAttention(
                config.embed_dim, num_heads=config.n_heads))

            self.mlps.append(nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(self.dropout),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            ))

        self.attns_fused = nn.ModuleList()
        self.mlps_fused = nn.ModuleList()

        self.fusion_pool = nn.MaxPool1d(config.kernel_sz[0], stride=config.stride_sz[0])

        logging.info(["Reducing sequence during furion by factors of ", config.stride_sz, ' with kernel ', config.kernel_sz])

        for i in range(self.n_layers_post_fusion):
            self.attns_fused.append(nn.MultiheadAttention(
                config.embed_dim, num_heads=config.n_heads))

            self.mlps_fused.append(nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(self.dropout),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            ))

        
        self.dataset = config.dataset
    
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.n_classes)
        )

        self.tb = get_tb()

    def apply_transformer_block(self, z, mha, mlp):
        z_res = F.layer_norm(z, (self.embed_dim, ))
        z_res, _ = mha(z_res, z_res, z_res)

        z = z + F.dropout(z_res, p=self.dropout, training=self.training)
        z_res = mlp(z)

        z = z + F.dropout(z_res, p=self.dropout, training=self.training)

        return z


    def forward(self, z_rgb=None, z_spec=None, z_text=None):#, mixup_lam=None, mixup_index=None, latent_stats_tracker=None):
        """
        Assumes z_rgb and z_spec have already been pre-processed with positional encoding and a classifier token
        (rgb and spec cls tokens assumed to be z_rgb[0] and z_spec[0] respectively)
        params:
            - z_rgb: [n_rgb_tokens + 1, batch_size, feat_dim] ***note sequence first ordering***
            - z_spec: [n_audio_tokens + 1, batch_size, feat_dim]
        returns:
            - logits_rgb: [batch_size, n_classes]
            - logits_spec: [batch_size, n_classes]
        """
        z = z_rgb

        if self.modality == 'spec':
            logging.debug("spec!")
            z = z_spec
        elif self.modality == 'text':
            logging.debug("text!")
            z = z_text

        batch_sz = z.shape[1]

        logging.debug(['input dims', z.shape])

        for i in range(self.n_layers_pre_fusion):
            # ----- rgb -----
            z = self.apply_transformer_block(z, self.attns[i], self.mlps[i])
            
        
        z_cls = z[0, :, :]
        z = self.fusion_pool(torch.permute(z[1:,:,:], (2, 1, 0)))
        z = torch.permute(z, (2, 1, 0))
        z = torch.cat([z_cls.view(1, batch_sz, -1), z], 0)


        logging.debug(['catted', z.shape])
        for i in range(self.n_layers_post_fusion):

            z = self.apply_transformer_block(z, self.attns_fused[i], self.mlps_fused[i])

        logits = self.classifier(z[0])

        return logits
