import torch
import os

from torch.utils.tensorboard.writer import SummaryWriter

from models.mult import *
import torch.optim as optim
import json
import logging

WRITER = None
FUSION_TECHNIQUES = ['early', 'mid', 'late']
AVAILABLE_MODELS = ['mbt', 'sft', 'MulT', 'unimodal', 'concat', 'latefuse']
POOL_TYPES = ['max', 'avg', 'topk', 'attn', 'attn_topk', 'random', 'none', 'attn_wkmedoids', 'attn_random',
              'strided_random']
distance = torch.nn.CosineSimilarity(dim=-1)


def init_tensorboard(config):
    global WRITER

    path = os.path.join(
        "./", config.tb_dir, config.exp_name + "." + config.model + "." + config.dataset)

    WRITER = SummaryWriter(path)


def get_tb():
    return WRITER

def save_state(state, config, dir='./checkpoints'):

    model = state['model']
    optimizer = state['optimizer']

    model_name = config.model  # type(model).__name__
    optimizer_name = type(optimizer).__name__

    path = dir + "/" + config.exp_name + "." + model_name + "." + config.dataset + \
        '.ep_' + str(state['epoch']) + '.pth'

    logging.info("   Saving model to %s", path)
    torch.save({
        'epoch': state['epoch'],
        'model_name': model_name,
        'model': model.state_dict(),
        'optimizer_name': optimizer_name,
        'optimizer': optimizer.state_dict(),
        'config': config
    }, path)

    return path


def load_state(model, dataset, epoch, exp_name, dir='./checkpoints'):

    path = dir + "/" + exp_name + "." + model + "." + \
        dataset + '.ep_' + str(epoch) + '.pth'

    state = torch.load(path)

    logging.info("loaded state at %s", path)

    return state, state['config']
