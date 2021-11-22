"""
Modified from Multimodal Transformer:
https://github.com/yaohungt/Multimodal-Transformer
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.position_embedding import SinusoidalPositionalEmbedding
from models.wrapper import *
from torch.nn import Parameter
import torch.nn.functional as F

import logging
from utils.mixup import *
from utils.common import *

FUSION_TECHNIQUES = ['early', 'mid', 'late']

SUPPORTED_MIXUP_TYPES = {
    MIXUP_ANY_LAYER,
    MIXUP_WITHIN_MODALITY,
    MIXUP_FUSED,
    MIXUP_NONE
}


class MultimodalAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * \
            num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat(
                [k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat(
                [v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [
            bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(),
                                 dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(
            attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [
            bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class MultimodalTransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultimodalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        # The "Add & Norm" part in the paper
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList(
            [LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(
        torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


class MulT(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MulT, self).__init__()

        self.num_heads = hyp_params.n_heads
        self.fused_layers = hyp_params.n_layers_post_fusion
        self.cross_attn_layers = hyp_params.n_layers_pre_fusion

        self.fusion_type = hyp_params.fusion_type
        assert self.fusion_type in FUSION_TECHNIQUES

        self.attn_dropout = hyp_params.dropout
        self.attn_dropout_a = hyp_params.dropout
        self.attn_dropout_v = hyp_params.dropout
        self.relu_dropout = hyp_params.dropout
        self.res_dropout = hyp_params.dropout
        self.out_dropout = hyp_params.dropout
        self.embed_dropout = hyp_params.dropout

        self.attn_mask = hyp_params.dropout
        # self.mixup_type = hyp_params.mixup
        # assert self.mixup_type in SUPPORTED_MIXUP_TYPES

        self.n_modalities = hyp_params.n_modalities

        self.embed_dim = hyp_params.embed_dim

        self.combined_dim = 2 * self.embed_dim * self.n_modalities

        # This is actually not a hyperparameter :-)
        output_dim = hyp_params.n_classes

        self.embed_positions = SinusoidalPositionalEmbedding(self.embed_dim)

        # 1. Temporal convolutional layers

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(
            self.cross_attn_layers, self_type='la')
        self.trans_l_with_v = self.get_network(
            self.cross_attn_layers, self_type='lv')
        self.trans_a_with_l = self.get_network(
            self.cross_attn_layers, self_type='al')
        self.trans_a_with_v = self.get_network(
            self.cross_attn_layers, self_type='av')
        self.trans_v_with_l = self.get_network(
            self.cross_attn_layers, self_type='vl')
        self.trans_v_with_a = self.get_network(
            self.cross_attn_layers, self_type='va')

        self.cross_modal_norm = LayerNorm(self.embed_dim)

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(
            self.fused_layers, self_type='l_mem')
        self.trans_a_mem = self.get_network(
            self.fused_layers, self_type='a_mem')
        self.trans_v_mem = self.get_network(
            self.fused_layers, self_type='v_mem')

        self.atten_norm = LayerNorm(2*self.embed_dim)

        # Projection layers
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)

        # self.out_layer = nn.Linear(self.combined_dim, output_dim)

        self.dataset = hyp_params.dataset
        if hyp_params.dataset == 'ek':
            self.verb_classifier = nn.Sequential(
                nn.LayerNorm(self.combined_dim),
                nn.Linear(self.combined_dim, hyp_params.n_verb_classes)
            )
            self.noun_classifier = nn.Sequential(
                nn.LayerNorm(self.combined_dim),
                nn.Linear(self.combined_dim, hyp_params.n_noun_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.combined_dim),
                nn.Linear(self.combined_dim, hyp_params.n_classes)
            )

        # logging.info("MulT Initialized using %s mixup" % self.mixup_type)

    def get_network(self, n_layers, self_type='l'):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.embed_dim, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.embed_dim, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.embed_dim, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.embed_dim, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.embed_dim, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.embed_dim, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        # logging.debug(['embed dim', self.embed_dim, self.num_heads])

        enc_layers = nn.ModuleList()

        for l in range(n_layers):
            enc_layers.append(MultimodalTransformerEncoderLayer(
                embed_dim,
                num_heads=self.num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=self.relu_dropout,
                res_dropout=self.res_dropout
            ))

        return enc_layers

    # , mixup_lam=None, mixup_index=None):
    def forward(self, proj_x_v, proj_x_a, proj_x_l):
        """
        Input: [seq, batch, nfeatures]


        # ORIGINAL, refactored
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        input: [batch_size, seq_len, n_features]
        transpose: [batch, features, seq_len]
        proj-permute: [seq_len, batch, features]
        """

        h_l_with_as = self.trans_l_with_a[0](proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v[0](proj_x_l, proj_x_v, proj_x_v)

        # ---- attend onto A ----
        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l[0](proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v[0](proj_x_a, proj_x_v, proj_x_v)

        # ---- attend onto V ----
        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l[0](proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a[0](proj_x_v, proj_x_a, proj_x_a)

        # after proj_permute: seq_len x batch x n_features
        for i in range(1, self.cross_attn_layers):

            # EARLY FUSION
            # ---- attend onto L----
            # (V,A) --> L # Dimension (L, N, d_l)
            h_l_with_as = self.trans_l_with_a[i](
                h_l_with_as, proj_x_a, proj_x_a)
            h_l_with_vs = self.trans_l_with_v[i](
                h_l_with_vs, proj_x_v, proj_x_v)

            # ---- attend onto A ----
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l[i](
                h_a_with_ls, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v[i](
                h_a_with_vs, proj_x_v, proj_x_v)

            # ---- attend onto V ----
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l[i](
                h_v_with_ls, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a[i](
                h_v_with_as, proj_x_a, proj_x_a)

        h_l_with_as, h_l_with_vs = self.cross_modal_norm(
            h_l_with_as), self.cross_modal_norm(h_l_with_vs)
        h_a_with_ls, h_a_with_vs = self.cross_modal_norm(
            h_a_with_ls), self.cross_modal_norm(h_a_with_vs)
        h_v_with_ls, h_v_with_as = self.cross_modal_norm(
            h_v_with_ls), self.cross_modal_norm(h_v_with_as)

        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)

        last_h_l = 0
        last_h_a = 0
        last_h_v = 0

        for i in range(self.fused_layers):

            h_ls = self.trans_l_mem[i](h_ls)
            h_as = self.trans_a_mem[i](h_as)
            h_vs = self.trans_v_mem[i](h_vs)

        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        # Take the last output for prediction
        last_h_l = last_hs = h_ls[0]

        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[0]

        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[0]

        # logging.debug(['last_h stuff', last_h_l.shape, last_h_a.shape, last_h_v.shape])

        last_h_l = self.atten_norm(last_h_l)
        last_h_a = self.atten_norm(last_h_a)
        last_h_v = self.atten_norm(last_h_v)

        # if self.partial_mode == 3:
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(
            F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        # output = self.out_layer(last_hs_proj)

        logits = self.classifier(last_hs_proj)

        # logging.debug(['logit', logits.shape])
        return logits
