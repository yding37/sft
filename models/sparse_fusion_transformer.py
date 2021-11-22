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
from fairseq.modules.sparse_multihead_attention import SparseMultiheadAttention


class TopKPool(nn.Module):
    def __init__(self, k=7):
        super(TopKPool, self).__init__()
        self.k = k

    def forward(self, x):
        """
        Args:
            - x: [batch_size, feature_dim, seq_len]
        Returns: [batch_size, feature_dim, k]
        """
        z, idx = torch.topk(x, k=self.k, dim=2, largest=True, sorted=True)
        return z


class AttnWeightedAvgPool(nn.Module):
    def __init__(self, kernel_size=7, stride=5, padding=0):
        super(AttnWeightedAvgPool, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size, stride, padding)
        self.kernel_size = kernel_size

    def forward(self, x, w):
        """
        Args:
            - x: [batch_size, feature_dim, seq_len] tensor of features
            - w: [batch_size, seq_len, seq_len] tensor of attention weights
        Returns: [batch_size, k, feature_dim]
        """
        w = torch.mean(
            w, dim=1, keepdim=True)  # get saliency weight for each feature
        x = x * w.expand(*x.shape)
        x_sum = self.avg_pool(x)
        w_sum = self.avg_pool(w)
        # NOTE: vvv dividing by two average pools results in 1/n term cancelling, equivalent to weighted average by w
        return x_sum / (w_sum.expand(*x_sum.shape) + 1e-5)


class AttnTopKPool(nn.Module):
    def __init__(self, k=5):
        super(AttnTopKPool, self).__init__()
        self.k = k

    def forward(self, x, w):
        """
        Args:
            - x: [batch_size, feature_dim, seq_len] tensor of features
            - w: [batch_size, seq_len, seq_len] tensor of attention weights
        Returns: [batch_size, k, feature_dim]
        """
        w = torch.mean(w, dim=1)  # get saliency weight for each feature
        # get top weights and indices of those weights
        w_topk, idx = torch.topk(w, k=self.k, dim=1)
        # gather features using topk indices
        x = torch.stack([x[b, :, idx[b]] for b in range(x.shape[0])], dim=0)
        return x


class StridedRandomPool(nn.Module):
    def __init__(self, kernel_size=5, padding=0):
        super(StridedRandomPool, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x):
        """
        Args:
            - x: [batch_size, feature_dim, seq_len] tensor of features
        Returns: [batch_size, feature_dim, k]
        """
        # first, pad the sequence
        batch_size, feature_dim, seq_len = x.shape
        x_pad = torch.zeros((batch_size, feature_dim, self.padding)).type_as(x)
        x = torch.cat((x_pad, x, x_pad), dim=2)
        # batch, feature_dim, kernel_size, n_windows
        x = x.view(batch_size, feature_dim, self.kernel_size, -1)
        n_windows = x.shape[-1]

        # choose an index for every window
        idx = torch.randint(0, self.kernel_size, size=(
            batch_size, n_windows), dtype=torch.long, device=x.device)
        idx = idx.unsqueeze(1).unsqueeze(2).expand(
            batch_size, feature_dim, 1, n_windows)

        # gather features according to that idx
        x = torch.gather(x, dim=2, index=idx).squeeze(2)

        return x


class RandomKPool(nn.Module):
    def __init__(self, k=5):
        super(RandomKPool, self).__init__()
        self.k = k

    def forward(self, x):
        """
        Args:
            - x: [batch_size, feature_dim, seq_len] tensor of features
        Returns: [batch_size, k, feature_dim]
        """
        idx = [torch.randperm(x.shape[2])[:self.k]
               for _ in range(x.shape[0])]  # generate random indices
        # select features using random indices
        x = torch.stack([x[b, :, idx[b]] for b in range(x.shape[0])], dim=0)
        return x


class AttnWeightedRandomKPool(nn.Module):
    def __init__(self, k=5):
        super(AttnWeightedRandomKPool, self).__init__()
        self.k = k

    def forward(self, x, w):
        """
        Args:
            - x: [batch_size, feature_dim, seq_len] tensor of features
            - w: [batch_size, seq_len, seq_len] tensor of attention weights
        Returns: [batch_size, feature_dim, k]
        """
        w = torch.mean(w, dim=1)  # get saliency weight for each feature
        idx = torch.multinomial(w, self.k, replacement=False)
        # select features using random indices
        x = torch.stack([x[b, :, idx[b]] for b in range(x.shape[0])], dim=0)
        return x


class AttnWeightedKMedoidsPool(nn.Module):
    def __init__(self, k=5, iters=5):
        super(AttnWeightedKMedoidsPool, self).__init__()
        self.k = k
        self.iters = iters

    def forward(self, x, w):
        """
        Args:
            - x: [batch_size, feature_dim, seq_len] tensor of features
            - w: [batch_size, seq_len, seq_len] tensor of attention weights
        Returns: [batch_size, feature_dim, k]
        """
        if self.k >= x.shape[2]:
            return x
        batch_size, feature_dim, seq_len = x.shape
        w = torch.mean(w, dim=1)  # get saliency weight for each feature
        d = self.get_token_dist(x, p=1)     # get L1 distance matrix

        # ctr_idx corresponds to the k indices per batch of the features representing the cluster centers
        # initialize ctr_idx to correspond to features with k highest saliency weights
        # get indices of topk saliency weights
        _, ctr_idx = torch.topk(w, k=self.k, dim=1)

        for _ in range(self.iters):
            # first, cluster features in x according to distance to cluster centers
            i_to_ctr_dists = torch.stack(
                [d[b, :, ctr_idx[b]] for b in range(batch_size)], dim=0)
            cluster_assignment = torch.argmin(i_to_ctr_dists, dim=2)

            # then, calculate new cluster indices according to cluster centers, weighted by saliency score
            seq_range = torch.arange(seq_len).type_as(ctr_idx)
            for k in range(self.k):
                for b in range(batch_size):
                    c_assignment = cluster_assignment[b] == k
                    x_k = x[b, :, c_assignment]
                    w_k = w[b, c_assignment]

                    # get weighted distance to all other tokens
                    d_k = d[b, c_assignment][:, c_assignment]
                    d_k_weighted = d_k * w_k.unsqueeze(0).expand(*d_k.shape)
                    new_ctr = torch.argmin(torch.sum(d_k_weighted, dim=1))
                    new_ctr = seq_range[c_assignment][new_ctr]
                    ctr_idx[b, k] = new_ctr

        # gather features using cluster ctr
        x = torch.gather(x, dim=2, index=ctr_idx.unsqueeze(
            1).expand(batch_size, feature_dim, self.k))
        return x

    @staticmethod
    def get_token_dist(x, p=1):
        """
        Args:
            - x: [batch_size, feature_dim, seq_len] tensor of features
            - p: int indicating which distancemetric to use. Default is L1, as this is better than L2 for high dim data
        Returns: [batch_size, seq_len, seq_len] matrix, i,j,kth entry is distance from x[i,:,j] to x[i,:,k], i.e.
                 || x[i,:,j] - x[i,:,k] ||_p
        """
        batch_size, feature_dim, seq_len = x.shape
        x_rpt_1 = torch.repeat_interleave(x, seq_len, dim=2)
        x_rpt_2 = x.repeat(1, 1, seq_len)
        d = torch.norm(x_rpt_1 - x_rpt_2, p=p,
                       dim=1).view(batch_size, seq_len, seq_len)
        return d


def get_fusion_pool(pool_type, kernel_sz, stride_sz, padding, pool_k):
    if pool_type == 'max':
        return nn.ModuleList([
            nn.MaxPool1d(
                kernel_sz[0], stride=stride_sz[0], padding=padding[0]),
            nn.MaxPool1d(
                kernel_sz[1], stride=stride_sz[1], padding=padding[1]),
            nn.MaxPool1d(
                kernel_sz[2], stride=stride_sz[2], padding=padding[2]),
        ])
    elif pool_type == 'avg':
        return nn.ModuleList([
            nn.AvgPool1d(
                kernel_sz[0], stride=stride_sz[0], padding=padding[0]),
            nn.AvgPool1d(
                kernel_sz[1], stride=stride_sz[1], padding=padding[1]),
            nn.AvgPool1d(
                kernel_sz[2], stride=stride_sz[2], padding=padding[2]),
        ])
    elif pool_type == 'topk':
        return nn.ModuleList([
            TopKPool(k=pool_k),
            TopKPool(k=pool_k),
            TopKPool(k=pool_k),
        ])
    elif pool_type == 'attn':
        return nn.ModuleList([
            AttnWeightedAvgPool(
                kernel_size=kernel_sz[0], stride=stride_sz[0], padding=padding[0]),
            AttnWeightedAvgPool(
                kernel_size=kernel_sz[1], stride=stride_sz[1], padding=padding[1]),
            AttnWeightedAvgPool(
                kernel_size=kernel_sz[2], stride=stride_sz[2], padding=padding[2]),
        ])
    elif pool_type == 'attn_topk':
        return nn.ModuleList([
            AttnTopKPool(k=pool_k),
            AttnTopKPool(k=pool_k),
            AttnTopKPool(k=pool_k),
        ])
    elif pool_type == 'random':
        return nn.ModuleList([
            RandomKPool(k=pool_k),
            RandomKPool(k=pool_k),
            RandomKPool(k=pool_k),
        ])
    elif pool_type == 'attn_wkmedoids':
        return nn.ModuleList([
            AttnWeightedKMedoidsPool(k=pool_k, iters=5),
            AttnWeightedKMedoidsPool(k=pool_k, iters=5),
            AttnWeightedKMedoidsPool(k=pool_k, iters=5),
        ])
    elif pool_type == 'attn_random':
        return nn.ModuleList([
            AttnWeightedRandomKPool(k=pool_k),
            AttnWeightedRandomKPool(k=pool_k),
            AttnWeightedRandomKPool(k=pool_k),
        ])
    elif pool_type == 'strided_random':
        return nn.ModuleList([
            StridedRandomPool(kernel_size=kernel_sz[0], padding=padding[0]),
            StridedRandomPool(kernel_size=kernel_sz[1], padding=padding[1]),
            StridedRandomPool(kernel_size=kernel_sz[2], padding=padding[2]),
        ])
    elif pool_type == 'none':
        return None
    else:
        raise NotImplementedError


class SparseFusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.mixup_type = config.mixup
        self.alpha_type = config.alpha_type

        self.embed_dim = config.embed_dim
        self.dropout = config.dropout
        self.n_layers_pre_fusion = config.n_layers_pre_fusion
        self.n_layers_post_fusion = config.n_layers_post_fusion
        self.total_layers = config.n_layers_pre_fusion + config.n_layers_post_fusion
        self.pool_type = config.pool_type

        self.attns_rgb = nn.ModuleList()
        self.attns_spec = nn.ModuleList()
        self.attns_text = nn.ModuleList()

        self.mlps_rgb = nn.ModuleList()
        self.mlps_spec = nn.ModuleList()
        self.mlps_text = nn.ModuleList()

        for i in range(self.n_layers_pre_fusion):
            self.attns_rgb.append(nn.MultiheadAttention(
                config.embed_dim, num_heads=config.n_heads))

            self.attns_spec.append(nn.MultiheadAttention(
                config.embed_dim, num_heads=config.n_heads))

            self.attns_text.append(nn.MultiheadAttention(
                config.embed_dim, num_heads=config.n_heads))

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

        self.attns_fused = nn.ModuleList()
        self.mlps_fused = nn.ModuleList()

        self.sparse_attn = nn.ModuleList([
            SparseMultiheadAttention(
                config.embed_dim, config.n_heads
            ),
            SparseMultiheadAttention(
                config.embed_dim, config.n_heads
            ), SparseMultiheadAttention(
                config.embed_dim, config.n_heads
            )
        ])

        self.modality_pools = get_fusion_pool(
            config.pool_type, config.kernel_sz, config.stride_sz, config.padding, config.pool_k)
        logging.info(["reduction factors, pool", config.pool_k])

        if config.fusion_layer_enabled:
            self.fusion_attn = nn.ModuleList([])
            self.fusion_mlps = nn.ModuleList([])
            self.fusion_pool = get_fusion_pool(
                config.pool_type, config.ks_fused, config.ks_fused, [0, 0, 0], config.ks_pool)

            logging.info(
                ["Reducing fused repr with kernel-stride", config.ks_fused])

            for i in range(3):
                # 3 modalities
                self.fusion_attn.append(nn.MultiheadAttention(
                    config.embed_dim, num_heads=config.n_heads))

                self.fusion_mlps.append(nn.Sequential(
                    nn.LayerNorm(self.embed_dim),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Dropout(self.dropout),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dim, self.embed_dim)
                ))

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

    def apply_transformer_block(self, z, mha, mlp, return_weights=False):

        z, w = self.apply_mha(z, mha)
        z_res = mlp(z)

        z = z + F.dropout(z_res, p=self.dropout, training=self.training)

        if return_weights:
            return z, w

        return z

    def apply_mha(self, z, mha):
        z_res = F.layer_norm(z, (self.embed_dim, ))
        z_res, w = mha(z_res, z_res, z_res)

        z = z + F.dropout(z_res, p=self.dropout, training=self.training)

        return z, w

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
        z = 0.0

        interp_layer = 0
        batch_sz = z_rgb.shape[1]

        if self.mixup_type == MIXUP_FUSED:
            interp_layer = np.random.randint(
                self.n_layers_pre_fusion, self.total_layers)
        elif self.mixup_type == MIXUP_WITHIN_MODALITY or self.mixup_type == MIXUP_CROSS_MODALITY:
            interp_layer = np.random.randint(0, self.n_layers_pre_fusion)
        elif self.mixup_type == MIXUP_ANY_LAYER:
            interp_layer = np.random.randint(0, self.total_layers)
        elif self.mixup_type != MIXUP_NONE:
            raise NotImplementedError

        logging.debug(['input dims', z_rgb.shape, z_spec.shape, z_text.shape])

        for i in range(self.n_layers_pre_fusion):

            z_rgb, w_rgb = self.apply_transformer_block(z_rgb, self.attns_rgb[i], self.mlps_rgb[i],
                                                        return_weights=True)

            z_spec, w_spec = self.apply_transformer_block(z_spec, self.attns_spec[i], self.mlps_spec[i],
                                                          return_weights=True)

            if z_text != None:
                z_text, w_text = self.apply_transformer_block(z_text, self.attns_text[i], self.mlps_text[i],
                                                              return_weights=True)

            if mixup_lam is not None:

                if self.mixup_type in [MIXUP_WITHIN_MODALITY, MIXUP_ANY_LAYER] and i == interp_layer:
                    logging.debug(['mixup ', self.mixup_type, i])
                    if self.alpha_type == ALPHA_PER_MODALITY:
                        z_rgb = mixup(
                            z_rgb, mixup_lam[0], mixup_index, self.training, self.config.permute_seq)
                        z_spec = mixup(
                            z_spec, mixup_lam[1], mixup_index, self.training, self.config.permute_seq)
                        z_text = mixup(
                            z_text, mixup_lam[2], mixup_index, self.training, self.config.permute_seq)
                    else:
                        raise NotImplementedError

        # max pool before
        # swap

        if z_text != None:

            z_rgb, w_rgb = self.apply_mha(z_rgb, self.sparse_attn[0])
            z_spec, w_spec = self.apply_mha(z_spec, self.sparse_attn[1])
            z_text, w_text = self.apply_mha(z_text, self.sparse_attn[2])

            z_cls = z_rgb[0, :, :] + z_spec[0, :, :] + z_text[0, :, :]

            if self.pool_type != 'none':
                # permute to [batch_size, feature_dim, sequence_length]
                z_rgb = torch.permute(z_rgb[1:, :, :], (1, 2, 0))
                z_spec = torch.permute(z_spec[1:, :, :], (1, 2, 0))
                z_text = torch.permute(z_text[1:, :, :], (1, 2, 0))

                if 'attn' in self.pool_type:
                    z_rgb = self.modality_pools[0](z_rgb, w_rgb[:, 1:, 1:])
                    z_spec = self.modality_pools[1](z_spec, w_spec[:, 1:, 1:])
                    z_text = self.modality_pools[2](z_text, w_text[:, 1:, 1:])

                else:
                    z_rgb = self.modality_pools[0](z_rgb)
                    z_spec = self.modality_pools[1](z_spec)
                    z_text = self.modality_pools[2](z_text)

                z_rgb = torch.permute(z_rgb, (2, 0, 1))
                z_spec = torch.permute(z_spec, (2, 0, 1))
                z_text = torch.permute(z_text, (2, 0, 1))

                if self.config.fusion_layer_enabled:
                    z_rgb = torch.cat([z_cls.view(1, batch_sz, -1), z_rgb], 0)
                    z_spec = torch.cat(
                        [z_cls.view(1, batch_sz, -1), z_spec], 0)
                    z_text = torch.cat(
                        [z_cls.view(1, batch_sz, -1), z_text], 0)

                    logging.debug(
                        ['modality pool dims', z_rgb.shape, z_spec.shape, z_text.shape])

                    # weak-fusion layer
                    z_cls = z_rgb[0, :, :] + z_spec[0, :, :] + z_text[0, :, :]

                    z_rgb = torch.permute(z_rgb[1:, :, :], (1, 2, 0))
                    z_spec = torch.permute(z_spec[1:, :, :], (1, 2, 0))
                    z_text = torch.permute(z_text[1:, :, :], (1, 2, 0))

                    if 'attn' in self.pool_type:
                        z_rgb = self.fusion_pool[0](z_rgb, w_rgb[:, 1:, 1:])
                        z_spec = self.fusion_pool[1](z_spec, w_spec[:, 1:, 1:])
                        z_text = self.fusion_pool[2](z_text, w_text[:, 1:, 1:])

                    else:
                        z_rgb = self.fusion_pool[0](z_rgb)
                        z_spec = self.fusion_pool[1](z_spec)
                        z_text = self.fusion_pool[2](z_text)

                    z_rgb = torch.permute(z_rgb, (2, 0, 1))
                    z_spec = torch.permute(z_spec, (2, 0, 1))
                    z_text = torch.permute(z_text, (2, 0, 1))

                    logging.debug(
                        ['modality fused dims', z_rgb.shape, z_spec.shape, z_text.shape])

                z = torch.cat([z_cls.view(1, batch_sz, -1),
                               z_rgb, z_spec, z_text], 0)

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        logging.debug(['fused dim', z.shape])
        for i in range(self.n_layers_post_fusion):

            z = self.apply_transformer_block(
                z, self.attns_fused[i], self.mlps_fused[i])

            if mixup_lam is not None:

                if self.mixup_type in [MIXUP_ANY_LAYER, MIXUP_FUSED] and i == interp_layer - self.n_layers_pre_fusion:
                    if self.alpha_type == ALPHA_PER_MODALITY:
                        # still perform fusion with any layer, just take first...
                        z = mixup(z, mixup_lam[0], mixup_index,
                                  self.training, self.config.permute_seq)
                    else:
                        raise NotImplementedError
        

        logits = self.classifier(z[0])

        return logits
