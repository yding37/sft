import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from models.position_embedding import SinusoidalPositionalEmbedding
from models.wrapper import *

from utils.mixup import *
import logging
import numpy as np
import math

from models.tokenizers import *

class MultimodalBottleneckTransformer(nn.Module):
    def __init__(self, config):
        
        super().__init__()

        self.config = config
        self.bottleneck_tokens = config.n_bn_tokens
        self.embed_dim = config.embed_dim

        self.n_layers_pre_fusion = config.n_layers_pre_fusion
        self.n_layers_post_fusion = config.n_layers_post_fusion
        self.n_layers = config.n_layers_pre_fusion + config.n_layers_post_fusion

        # self.l_F = fusion_layer
        self.dropout = config.dropout
        self.n_heads = config.n_heads
        self.n_classes = config.n_classes

        self.mixup_type = config.mixup
        self.alpha_type = config.alpha_type
        self.randomize_layer = True

        self.attns_rgb = nn.ModuleList()
        self.attns_spec = nn.ModuleList()
        self.attns_text = nn.ModuleList()

        self.mlps_rgb = nn.ModuleList()
        self.mlps_spec = nn.ModuleList()
        self.mlps_text = nn.ModuleList()

        for i in range(self.n_layers):
            self.attns_rgb.append(nn.MultiheadAttention(
                self.embed_dim, num_heads=self.n_heads))
            self.attns_spec.append(nn.MultiheadAttention(
                self.embed_dim, num_heads=self.n_heads))
            self.attns_text.append(nn.MultiheadAttention(
                self.embed_dim, num_heads=self.n_heads))

            self.mlps_rgb.append(nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(self.dropout),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            ))
            self.mlps_spec.append(nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(self.dropout),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            ))
            self.mlps_text.append(nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(self.dropout),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            ))

        self.dataset = config.dataset
        if config.dataset ==  'ek':
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

    def forward(self, z_rgb, z_spec, z_text=None, mixup_lam=None, mixup_index=None, latent_stats_tracker=None):
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
        batch_size = z_rgb.shape[1]
        z_fsn = torch.zeros((self.bottleneck_tokens, batch_size, self.embed_dim)).type_as(z_rgb)

        interp_layer = 0
        if self.randomize_layer:
            if self.mixup_type == MIXUP_WITHIN_MODALITY:
                interp_layer = np.random.randint(
                    0, self.l_F) if self.l_F > 0 else 0
            elif self.mixup_type == MIXUP_FUSED:
                interp_layer = np.random.randint(
                    self.l_F, self.L)
            elif self.mixup_type == MIXUP_ANY_LAYER:
                interp_layer = np.random.randint(0, self.n_layers)
            else:
                raise NotImplementedError

        for i in range(self.n_layers):
            # --- RGB transformer ---
            # attention part
            z = z_rgb
            if i > self.n_layers_pre_fusion:      # concat bottleneck tokens if mid-fusion layer
                z = torch.cat((z, z_fsn), dim=0)  # N_rgb + b, batch, d

            z_res = F.layer_norm(z, (self.embed_dim, ))
            z_res, attn_weights = self.attns_rgb[i](z_res, z_res, z_res)
            z = z + F.dropout(z_res, p=self.dropout, training=self.training)

            # mlp part
            z_res = self.mlps_rgb[i](z)
            z = z + F.dropout(z_res, p=self.dropout, training=self.training)

            if i > self.n_layers_pre_fusion:   # separate rgb tokens from bottleneck tokens after transformer layer
                z_rgb = z[:-self.bottleneck_tokens]
                z_fsn = z[-self.bottleneck_tokens:]
            else:
                z_rgb = z

            # --- Audio transformer ---
            # attention part
            z = z_spec
            if i > self.n_layers_pre_fusion:      # concat bottleneck tokens if mid-fusion layer
                z = torch.cat((z, z_fsn), dim=0)  # N_rgb + b, batch, d
            z_res = F.layer_norm(z, (self.embed_dim, ))
            z_res, attn_weights = self.attns_spec[i](z_res, z_res, z_res)
            z = z + F.dropout(z_res, p=self.dropout, training=self.training)

            # mlp part
            z_res = self.mlps_spec[i](z)
            z = z + F.dropout(z_res, p=self.dropout, training=self.training)

            if i > self.n_layers_pre_fusion:   # separate spec tokens from bottleneck tokens after transformer layer
                z_spec = z[:-self.bottleneck_tokens]
                z_fsn = z[-self.bottleneck_tokens:]
            else:
                z_spec = z

            # --- Text transformer ---
            if z_text != None:
                # attention part
                z = z_text
                if i > self.n_layers_pre_fusion:      # concat bottleneck tokens if mid-fusion layer
                    z = torch.cat((z, z_fsn), dim=0)  # N_rgb + b, batch, d
                z_res = F.layer_norm(z, (self.embed_dim, ))
                z_res, attn_weights = self.attns_text[i](z_res, z_res, z_res)
                z = z + F.dropout(z_res, p=self.dropout,
                                  training=self.training)

                # mlp part
                z_res = self.mlps_text[i](z)
                z = z + F.dropout(z_res, p=self.dropout,
                                  training=self.training)

                if i > self.n_layers_pre_fusion:   # separate spec tokens from bottleneck tokens after transformer layer
                    z_text = z[:-self.bottleneck_tokens]
                    z_fsn = z[-self.bottleneck_tokens:]
                else:
                    z_text = z

            # --- mixup ---
            # mixup first layer after fusion for fused mode
            # Currently: Mixup all Z excluding bottleneck repr, do we want to only use bottleneck tokens? (or fused tokens)
            if i == interp_layer and mixup_lam is not None:
                if self.mixup_type in [MIXUP_WITHIN_MODALITY, MIXUP_FUSED, MIXUP_ANY_LAYER]:
                    logging.debug(['mixup', self.mixup_type, i])
                    if self.alpha_type == ALPHA_PER_MODALITY:
                        z_rgb = mixup(z_rgb, mixup_lam[0], mixup_index, self.training, self.config.permute_seq)
                        z_spec = mixup(z_spec, mixup_lam[1], mixup_index, self.training, self.config.permute_seq)
                        z_text = mixup(z_text, mixup_lam[2], mixup_index, self.training, self.config.permute_seq)
                    else:
                        raise NotImplementedError
        

        z_rgb_cls = z_rgb[0]
        z_spec_cls = z_spec[0]
        z_text_cls = z_text[0]
        
        logits_rgb = self.classifier(z_rgb_cls)
        logits_spec = self.classifier(z_spec_cls)
        logits_text = self.classifier(z_text_cls)

        logits = logits_rgb + logits_spec + logits_text
        return logits / 3.
