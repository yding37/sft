import torch
import argparse
import logging

from scripts import train_eval_vgg

import scripts.train_eval_mosei as train_eval_mosei
import scripts.train_eval_vgg as train_eval_vgg

from utils.mixup import MIXUP_TYPES
from utils.parsers import *

from utils.common import AVAILABLE_MODELS, init_tensorboard


parser = argparse.ArgumentParser(
    description='Interpolating Multimodal Representations')
parser.add_argument('-f', default='', type=str)

# Commands to run different mixup and dataset experiments
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (options: MulT, mbt, lft) (default: MulT)')

parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti) [mosei_senti, vgg]')

parser.add_argument('--debug', action='store_true', default=False,
                    help='run in debug mode (default: off)')

parser.add_argument('--tb_dir', type=str, default='tb_logs',
                    help='Tensorboard output directory (default: "tb_logs")')

parser.add_argument('--mixup', type=str, default='within',
                    help='mixup method (within, cross, any, temporal, fused, perturbed, bottleneck) (default: within)')

parser.add_argument('--eval_only', action='store_true', default=False,
                    help='run in evaluation mode only (default: off)')

parser.add_argument('--best_epoch', type=int, default=0,
                    help='epoch to evaluate (default: 0)')

parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')

parser.add_argument('--delay_stop', type=int, default=0,
                    help='minimum # of epochs to train for (default: 0)')

parser.add_argument('--n_bn_tokens', type=int, default=4,
                    help='number of bottleneck tokens for mbt')

parser.add_argument('--alpha_type', type=str, default='static',
                    help='method for choosing alpha [static, permodality] (default: static)')


# Global Hyper parameters
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--when', type=int, default=10,
                    help='when to decay learning rate (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')

parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay, default 0.0')

parser.add_argument('--n_classes', type=int, default=7)

parser.add_argument('--n_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--embed_dim', type=int, default=30)
parser.add_argument('--fusion_type', type=str, default='early',
                    help='supported = [early, mid, late]')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='default 0.0')


parser.add_argument('--class_set', type=int, default=0,
                    help='which class set to use for vgg')


parser.add_argument('--n_layers_pre_fusion', type=int, default=6,
                    help='default 6')
parser.add_argument('--n_layers_post_fusion', type=int, default=6,
                    help='default 6')
parser.add_argument('--pool_type', type=str, default='max', help='pool type used to reduce sequence dimension in SFT'
                                                                 '(default: max) [max, avg, topk, attn, attn_topk, '
                                                                 'random, none, attn_wkmedoids, attn_random]')
parser.add_argument('--pool_k', type=int, default=5,
                    help='desired # of tokens for each modality after pooling')


# parser.add_argument('--strides', type=str, default='5,5,5',
#                     help='csv of stride for max pool, default: 5,5,5')
# parser.add_argument('--kernels', type=str, default='7,7,7',
#                     help='kernel sizes for max pool, default: 7,7,7')

# parser.add_argument('--alpha_scale', type=float, default=2.0,
#                     help='scale alpha up or down during dynamic sampling')


# parser.add_argument('--permute_seq', action='store_true', default=False,
#                     help='permute along sequence dimension of mixup')

parser.add_argument("--custom_ks", action='store_true', default=False,
                    help='enable custom kernel stride padding')

parser.add_argument("--ks_modality", type=str, default=None,
                    help='csv of kernel and stride for max pool per modality, default: None')

parser.add_argument("--pad_modality", type=str, default='0,0,0',
                    help='custom modality padding')


parser.add_argument("--fusion_layer_enabled", action='store_true', default=False,
                    help='enable custom kernel stride padding')

# parser.add_argument("--ks_fused", type=str, default=None,
#                     help='csv of kernel and stride for max pool per fused modality, default: None')

parser.add_argument("--ks_pool", type=int, default=5,
                    help='target number of tokens after fusion')
                    
# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='do not use cuda')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='cuda device (default: 0)')
parser.add_argument('--exp_name', type=str, default='default',
                    help='name of the trial (default: "default")')

# Label Noise Correction
parser.add_argument('--noise_level', type=float, default=0.0,
                    help='percentage of noise added to the data (values from 0. to 100.) (default: 0.)')
parser.add_argument('--loss_modeling', action='store_true', default=False,
                    help='use beta mixture model for label loss modeling and correction (default: off)')
parser.add_argument('--alpha', type=float, default=.3,
                    help='alpha parameter for the mixup distribution (default: .3)')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help="Number of epochs before performing mixup  (default: 10)")
parser.add_argument('--reg_term', type=float, default=0.,
                    help="Parameter of the regularization term (default: 0).")

parser.add_argument('--modality', type=str, default='all',
                    help='which modality to evaluate, default all, (rgb, spec, text')


if __name__ == '__main__':

    args, _ = parser.parse_known_args()

    log_path = "./logs/" + args.exp_name + "." + args.model + "." + args.dataset + \
        '.log'

    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('==========================================')
    logging.info('Sparse Fusion for Multimodal Transformers')
    logging.info('==========================================')

    if args.debug:
        logging.info("Running in DEBUG mode.")
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Seed: %d ", args.seed)
    torch.manual_seed(args.seed)
    dataset = str.lower(args.dataset.strip())

    #################################################################
    # add pre-defined settings for different models and datsets here
    #################################################################

    assert args.model in AVAILABLE_MODELS

    config = parser.parse_args()

    logging.info('Experiment config')
    logging.info(config)

    use_cuda = False

    if torch.cuda.is_available():
        if config.no_cuda:
            logging.warning(
                "You have a CUDA device, so you should probably not run with --no_cuda")
        else:
            torch.cuda.manual_seed(args.seed)
            use_cuda = True

    config.use_cuda = use_cuda

    init_tensorboard(config)
    if dataset == 'mosei_senti':
        train_eval_mosei.run(config)
    elif dataset == 'vgg':
        train_eval_vgg.run(config)

    else:
        logging.error("Invalid dataset. Choose one of [mosei_senti, vgg]")
