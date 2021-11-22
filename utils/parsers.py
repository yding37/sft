import logging

def add_lft_parser_args(p, dataset):
    if dataset == 'mosei_senti':
        p.add_argument('--n_rgb_timesteps', type=int, default=500)
        p.add_argument('--n_spec_timesteps', type=int, default=1000)
        p.add_argument('--spec_dim', type=int, default=128)
        p.add_argument('--rgb_feat_type', type=str, default='I3D')
                       

def add_bt_parser_args(p, dataset):

    p.add_argument('--n_bn_tokens', type=int, default=4,
                    help='default 4')

    if dataset == 'mosei_senti':
        p.add_argument('--n_rgb_timesteps', type=int, default=500)
        p.add_argument('--n_spec_timesteps', type=int, default=1000)
        p.add_argument('--spec_dim', type=int, default=128)
        p.add_argument('--rgb_feat_type', type=str, default='I3D')

    # if dataset == 'vgg':
    #     # p.add_argument('--rgb_feat_type', type=str, default='I3D')
    #     p.add_argument('--n_bn_tokens', type=int, default=4,
    #                    help='default 4')

                       

def add_mult_parser_args(p, dataset):
    # Dropouts
    p.add_argument('--attn_dropout', type=float, default=0.2,
                    help='attention dropout')
    p.add_argument('--attn_dropout_a', type=float, default=0.2,
                    help='attention dropout (for audio)')
    p.add_argument('--attn_dropout_v', type=float, default=0.2,
                    help='attention dropout (for visual)')
    p.add_argument('--relu_dropout', type=float, default=0.2,
                    help='relu dropout')
    p.add_argument('--embed_dropout', type=float, default=0.2,
                    help='embedding dropout')
    p.add_argument('--res_dropout', type=float, default=0.2,
                    help='residual block dropout')
    p.add_argument('--out_dropout', type=float, default=0.2,
                    help='output layer dropout')

    # Architecture
    # p.add_argument('--transformer_encoder_layers', type=int, default=5,
    #                 help='number of layers in the network (default: 5)')
    p.add_argument('--n_fused_blocks', type=int, default=4,
                    help='number of layers in the network (default: 4)')
    p.add_argument('--n_cross_attention_blocks', type=int, default=4,
                    help='number of layers in the network (default: 5)')
    p.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

    # if dataset == 'mosei_senti':

