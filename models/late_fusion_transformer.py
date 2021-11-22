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


class LateFusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed_dim = config.embed_dim
        self.dropout = config.dropout
        self.n_layers_pre_fusion = config.n_layers_pre_fusion
        self.n_layers_post_fusion = config.n_layers_post_fusion
        self.total_layers = config.n_layers_pre_fusion + config.n_layers_post_fusion

        self.attns_rgb = nn.ModuleList()
        self.attns_spec = nn.ModuleList()
        self.attns_text = nn.ModuleList()

        self.mlps_rgb = nn.ModuleList()
        self.mlps_spec = nn.ModuleList()
        self.mlps_text = nn.ModuleList()

        for i in range(self.total_layers):
            self.attns_rgb.append(nn.MultiheadAttention(
                config.embed_dim, num_heads=config.n_heads))

            self.mlps_rgb.append(nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(self.dropout),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            ))

            self.attns_spec.append(nn.MultiheadAttention(
                config.embed_dim, num_heads=config.n_heads))

            self.mlps_spec.append(nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(self.dropout),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            ))

            self.attns_text.append(nn.MultiheadAttention(
                config.embed_dim, num_heads=config.n_heads))

            self.mlps_text.append(nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(self.dropout),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            ))

        self.fusion_pool = nn.ModuleList([
            nn.MaxPool1d(config.kernel_sz[0], stride=config.stride_sz[0]),
            nn.MaxPool1d(config.kernel_sz[1], stride=config.stride_sz[1]),
            nn.MaxPool1d(config.kernel_sz[2], stride=config.stride_sz[2]),
        ])

        logging.info(["Reducing sequence during furion by factors of ",
                      config.stride_sz, ' with kernel ', config.kernel_sz])

        self.dataset = config.dataset
        if config.dataset == 'ek':
            self.verb_classifier = nn.Sequential(
                nn.LayerNorm(config.embed_dim),
                nn.Linear(config.embed_dim, config.n_verb_classes)
            )
            self.noun_classifier = nn.Sequential(
                nn.LayerNorm(config.embed_dim),
                nn.Linear(config.embed_dim, config.n_noun_classes)
            )
        else:
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

    # , mixup_lam=None, mixup_index=None, latent_stats_tracker=None):
    def forward(self, z_rgb=None, z_spec=None, z_text=None):
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

        logits = []

        for ix, z in enumerate([z_rgb, z_spec, z_text]):
            attns = self.attns_rgb
            mlps = self.mlps_rgb

            if ix == 1:
                attns = self.attns_spec
                mlps = self.mlps_spec
            elif ix == 2:
                attns = self.attns_text
                mlps = self.mlps_text

            batch_sz = z.shape[1]

            logging.debug(['input dims', z.shape])

            z = self.apply_transformer_block(z, attns[0], mlps[0])
            # pool
            z_cls = z[0, :, :]
            # z = self.fusion_pool[ix](torch.permute(z[1:, :, :], (2, 1, 0)))
            # z = torch.permute(z, (2, 1, 0))
            z = torch.cat([z_cls.view(1, batch_sz, -1), z], 0)
            logging.debug(['catted', z.shape])

            for i in range(1, self.total_layers):
                # ----- rgb -----
                z = self.apply_transformer_block(z, attns[i], mlps[i])

            logits.append(self.classifier(z[0]))

        return sum(logits)
