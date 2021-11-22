import torch
from torch import nn
import sys
from datasets.vggsound_dataset import VggH5pyDataset

from models.bottleneck_transformer import *
from models.sparse_fusion_transformer import *
from models.unimodal_transformer import *
from models.late_fusion_transformer import *
from models.concat_fusion_transformer import *
from models.tokenizers import *
from models.mult import *
from models.wrapper import *

from utils.common import *
from utils.loss_modeling_utils import *
# from utils.mixup import *
from torch.utils.data import DataLoader
from utils.latent_statistics import PerClassRunningAverage, PerClassPerModalityRunningAverage

import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from utils.vggsound_eval import *
import os
import pickle

# use to select different class sets
predefined_classes = [
    [],
    ['baby_babbling', 'baby_crying'],
    ['baby_crying', 'airplane'],
]


def init_dataloaders(config):
    logging.info("loading data....")

    classes = predefined_classes[config.class_set]

    logging.info("Performing classification on ")
    logging.info(classes)

    # TODO include training, test and validation
    dataset_name = 'vgg100' if config.n_classes == 100 else 'vgg'

    train_data = VggH5pyDataset(os.path.join(
        config.data_path, dataset_name + '_train.h5py'), ds_type='train', predefined_classes=classes)
    valid_data = VggH5pyDataset(os.path.join(
        config.data_path, dataset_name + '_train.h5py'), ds_type='valid', predefined_classes=classes)

    # use validation until ready...
    test_data = VggH5pyDataset(os.path.join(
        config.data_path, dataset_name + '_test.h5py'), ds_type='test', predefined_classes=classes)

    logging.info("Datsets loaded:")
    logging.info(["train", len(train_data)])
    logging.info(["valid", len(valid_data)])
    logging.info(["test", len(test_data)])

    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(
        valid_data, batch_size=config.batch_size, num_workers=4)
    test_loader = DataLoader(
        test_data, batch_size=config.batch_size, num_workers=4)

    loaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    # TODO refactor out
    logging.info('Finish loading the data....')

    # change input output dimensions based on dataset size of MulT
    config.rgb_dim, config.spect_dim, config.flow_dim = train_data.get_dims()
    config.rgb_seq_len, config.spect_seq_len, config.flow_seq_len = train_data.get_seq_lens()
    config.n_train, config.n_valid, config.n_test = len(
        train_data), len(valid_data), len(test_data)

    return loaders


def init_model(model_name, config):
    rgb_tokenizer = FeatureTokenizer(
        config.rgb_seq_len, config.rgb_dim, config.embed_dim)
    flow_tokenizer = FeatureTokenizer(
        config.flow_seq_len, config.flow_dim, config.embed_dim)
    spect_tokenizer = FeatureTokenizer(
        config.spect_seq_len, config.spect_dim, config.embed_dim)

    model = eval(model_name)(config)

    return FullModelShell(rgb_tokenizer, spect_tokenizer, flow_tokenizer, model)


def init_models(config):

    logging.info("Initializing models...")

    model = None

    assert config.model in AVAILABLE_MODELS

    if config.model == 'mbt':
        model = init_model('MultimodalBottleneckTransformer', config)
    elif config.model == 'MulT':
        model = init_model('MulT', config)
    elif config.model == 'sft':
        model = init_model('SparseFusionTransformer', config)
    elif config.model == 'unimodal':
        model = init_model('UnimodalTransformer', config)
    elif config.model == 'latefuse':
        model = init_model('LateFusionTransformer', config)
    elif config.model == 'concat':
        model = init_model('ConcatFusionTransformer', config)
    else:
        raise NotImplementedError

    assert model != None

    if config.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, config.optim)(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = getattr(nn.functional, config.criterion)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=config.when, factor=0.1, verbose=True)

    train_state = {'model': model,
                   'optimizer': optimizer,
                   'criterion': criterion,
                   'scheduler': scheduler
                   }  # ,
    #    'latent_stats_tracker': latent_stats_tracker}

    # prints model, optimizer, criterion, scheduler
    logging.debug(str(train_state))
    logging.info('...done')
    return train_state


def init_config(config):
    """ any additional custom settings for this model """
    if config.debug:
        logging.debug("Running in DEBUG mode")
        logging.debug("Reducing epochs to 5")

        config.warmup_epochs = 2
        config.num_epochs = 5

    dataset = str.lower(config.dataset.strip())
    config.permute_seq = False

    config.dataset = dataset
    config.model = config.model.strip()
    config.criterion = 'cross_entropy'
    config.n_modalities = 3

    config.embed_dim = 40
    config.n_heads = 5

    # determine stride and kernel size
    if config.custom_ks:
        # custom stride and kernel

        config.stride_sz = config.ks_modality.split(",")
        config.stride_sz = [int(x) for x in config.stride_sz]

        config.kernel_sz = config.stride_sz
        config.padding = [int(x) for x in config.pad_modality.split(',')]

    elif 'topk' not in config.pool_type:
        rgb_len = 38
        spec_len = 1200
        flow_len = 38
        config.stride_sz = [
            ((rgb_len - 1) // config.pool_k) + 1,
            ((spec_len - 1) // config.pool_k) + 1,
            ((flow_len - 1) // config.pool_k) + 1,
        ]
        config.kernel_sz = config.stride_sz

        config.padding = [
            min((config.stride_sz[0]*config.pool_k -
                 rgb_len) // 2, config.kernel_sz[0] // 2),
            min((config.stride_sz[1]*config.pool_k -
                 spec_len) // 2, config.kernel_sz[1] // 2),
            min((config.stride_sz[2]*config.pool_k -
                 flow_len) // 2, config.kernel_sz[2] // 2),
        ]

        logging.info('Kernel sizes: {}'.format(config.kernel_sz))
        logging.info('Padding: {}'.format(config.padding))
        logging.info('RGB sequence length will be reduced to {}'.format(
            (rgb_len + 2*config.padding[0]) // config.kernel_sz[0]))
        logging.info('Audio sequence length will be reduced to {}'.format(
            (spec_len + 2*config.padding[1]) // config.kernel_sz[1]))
        logging.info('Flow sequence length will be reduced to {}'.format(
            (flow_len + 2*config.padding[2]) // config.kernel_sz[2]))

    assert config.pool_type in POOL_TYPES

    logging.info("Running with config %s", str(config))

    return config


def train_mixup(state, loader, config):

    tb = get_tb()

    epoch = state['epoch']
    model = state['model']
    optimizer = state['optimizer']
    criterion = state['criterion']
    scheduler = state['scheduler']
    latent_stats_tracker = None

    model.train()
    if latent_stats_tracker is not None:
        prev_epoch_avg = latent_stats_tracker.current_average.clone()
        # save per epoch class centroids
        fname = './data/centroids.ep.%d.%s.npy' % (epoch, config.exp_name)
        with open(fname, 'wb') as f:
            pickle.dump(prev_epoch_avg, f)

        latent_stats_tracker.reset()

    epoch_loss = 0
    proc_loss, proc_size = 0, 0
    losses = []
    num_batches = config.n_train // config.batch_size

    start_time = time.time()
    for i_batch, batch in enumerate(loader):

        if config.debug:
            if i_batch > 10:
                break

        rgb = batch['rgb'].float()
        spect = batch['spect'].float()
        flow = batch['flow'].float()

        targets = batch['label'].squeeze(-1)   # if num of labels is 1

        logging.debug(["Batches", batch['rgb'].shape,
                       batch['spect'].shape, batch['flow'].shape, targets.shape])

        batch_size = rgb.size(0)
        if batch_size < config.batch_size:  # use full batches only, simplifes logic..
            continue

        model.zero_grad()

        if config.use_cuda:
            with torch.cuda.device(config.gpu_id):
                rgb, spect, flow = rgb.cuda(), spect.cuda(), flow.cuda()
                targets = targets.cuda()

        raw_loss = 0
        net = nn.DataParallel(model) if batch_size > 10 else model

        if config.mixup != MIXUP_NONE:
            index = torch.randperm(batch_size)

            if config.use_cuda:
                with torch.cuda.device(config.gpu_id):
                    index = index.cuda()

            targets_a, targets_b = targets, targets[index]
            lam = None

            if config.alpha_type in [ALPHA_STATIC]:
                lam = sample_interpolation_param(config.alpha, batch_size)
            elif config.alpha_type in [ALPHA_PER_MODALITY]:
                lam = [sample_interpolation_param(
                    config.alpha, batch_size) for m in range(config.n_modalities)]
                lam = torch.stack(lam, dim=0)

            if config.use_cuda:
                with torch.cuda.device(config.gpu_id):
                    lam = lam.cuda()

            output = net(rgb, spect, flow,
                         mixup_lam=lam, mixup_index=index, latent_stats_tracker=latent_stats_tracker)

            if config.alpha_type in [ALPHA_PER_MODALITY]:
                lam = torch.mean(lam, dim=0)

            raw_loss = mixup_criterion(
                output, targets_a, targets_b, lam, criterion, model.training, lam_vec=True)

        else:
            preds = net(rgb, spect, flow)

            raw_loss = criterion(preds, targets)

        raw_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)

        optimizer.step()
        losses.append(raw_loss.clone().detach())

        if config.loss_modeling:
            # tracking training loss
            state['bmm_model'], state['bmm_model_maxLoss'], state['bmm_model_minLoss'] = \
                track_training_loss(
                    model, config.use_cuda, loader)

        proc_loss += raw_loss.detach().item() * batch_size
        proc_size += batch_size
        epoch_loss += raw_loss.detach().item() * batch_size

        if latent_stats_tracker is not None:
            logging.debug(['updaing latent mean...'])
            latent_stats_tracker.calc_min_max()

            tb.add_scalar('train/distance_norm', torch.norm(
                latent_stats_tracker.pairwise_distances).clone().detach().cpu(), epoch*num_batches + i_batch + 1)

        avg_loss = torch.tensor(losses).mean()
        if i_batch % config.log_interval == 0:  # and i_batch > 0:
            avg_loss = proc_loss / proc_size
            elapsed_time = time.time() - start_time
            logging.info('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                         format(epoch, i_batch, num_batches, elapsed_time * 1000 / config.log_interval, avg_loss))

            tb.add_scalar(
                "loss/train_iter", torch.tensor(losses).mean().cpu(), epoch*num_batches + i_batch + 1)
            proc_loss, proc_size = 0, 0
            start_time = time.time()

    return epoch_loss / config.n_train


def train_warmup(state, loader, config):
    tb = get_tb()

    epoch = state['epoch']
    model = state['model']
    optimizer = state['optimizer']
    criterion = state['criterion']
    scheduler = state['scheduler']

    model.train()

    epoch_loss = 0
    proc_loss, proc_size = 0, 0
    losses = []
    num_batches = config.n_train // config.batch_size

    start_time = time.time()
    for i_batch, batch in enumerate(loader):

        if config.debug:
            if i_batch > 10:
                break

        rgb = batch['rgb'].float()
        spect = batch['spect'].float()
        flow = batch['flow'].float()

        targets = batch['label'].squeeze(-1)   # if num of labels is 1

        logging.debug(["Batches", batch['rgb'].shape,
                       batch['spect'].shape, batch['flow'].shape, targets.shape])

        batch_size = rgb.size(0)
        if batch_size < config.batch_size:  # use full batches only, simplifes logic..
            continue

        model.zero_grad()

        if config.use_cuda:
            with torch.cuda.device(config.gpu_id):
                rgb, spect, flow = rgb.cuda(), spect.cuda(), flow.cuda()
                targets = targets.cuda()

        net = nn.DataParallel(model) if batch_size > 10 else model

        preds = net(rgb, spect, flow)

        raw_loss = criterion(preds, targets)

        raw_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)

        optimizer.step()
        losses.append(raw_loss.clone().detach())

        proc_loss += raw_loss.clone().detach().item() * batch_size
        proc_size += batch_size
        epoch_loss += raw_loss.clone().detach().item() * batch_size

        avg_loss = torch.tensor(losses).mean()
        if i_batch % config.log_interval == 0:  # and i_batch > 0:
            avg_loss = proc_loss / proc_size
            elapsed_time = time.time() - start_time
            logging.info('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                         format(epoch, i_batch, num_batches, elapsed_time * 1000 / config.log_interval, avg_loss))

            tb.add_scalar(
                "loss/train_iter", torch.tensor(losses).mean().detach().cpu().clone(), epoch*num_batches + i_batch + 1)
            proc_loss, proc_size = 0, 0
            start_time = time.time()

    return epoch_loss / config.n_train


def evaluate(state, loader, config):

    model = state['model']
    model.eval()

    criterion = state['criterion']

    total_loss = 0.0

    results = []
    truths = []

    with torch.no_grad():
        for i_batch, batch in enumerate(loader):

            if config.debug:
                if i_batch > 10:
                    break

            rgb = batch['rgb'].float()
            spect = batch['spect'].float()
            flow = batch['flow'].float()

            logging.debug(["Batches", batch['rgb'].shape,
                           batch['spect'].shape, batch['flow'].shape])

            targets = batch['label'].squeeze(dim=-1)  # if num of labels is 1

            if config.use_cuda:
                with torch.cuda.device(config.gpu_id):
                    rgb, spect, flow = rgb.cuda(), spect.cuda(), flow.cuda()
                    targets = targets.cuda()

            batch_size = rgb.size(0)

            net = nn.DataParallel(model) if batch_size > 10 else model
            preds = net(rgb, spect, flow)

            logging.debug(preds.shape)

            total_loss += criterion(preds, targets).item() * batch_size

            # Collect the results into dictionary
            results.append(preds.detach())
            truths.append(targets.detach())

    truths = torch.cat(truths)
    results = torch.cat(results)

    avg_loss = total_loss / len(results)

    return avg_loss, results, truths


def run(config):

    config = init_config(config)

    loader = init_dataloaders(config)

    state = init_models(config)

    model = state['model']
    optimizer = state['optimizer']
    criterion = state['criterion']
    scheduler = state['scheduler']
    tb = get_tb()

    best_valid = 1e8
    best_epoch = 0

    if not config.eval_only:
        for epoch in range(config.num_epochs):
            start = time.time()

            logging.info('==========================================')

            state['epoch'] = epoch

            train_loss = 0
            if epoch < config.warmup_epochs:
                train_loss = train_warmup(state, loader['train'], config)
            else:
                train_loss = train_mixup(state, loader['train'], config)

            logging.info('Running validation...')
            val_loss, results, truths = evaluate(
                state, loader['valid'], config)

            logging.info('Validation results')
            logging.info('-------------------')

            results = get_vgg_stats(results, truths, config.n_classes)

            logging.info(['results', results])

            # test_loss, _, _ = evaluate(state, loader['test'], config)

            end = time.time()
            duration = end-start
            # Decay learning rate by validation loss
            scheduler.step(val_loss)

            if tb:
                # currently only recording losses, write all under loss/
                tb.add_scalar("loss/train", train_loss, epoch)
                tb.add_scalar("loss/val", val_loss, epoch)
                # tb.add_scalar("loss/test", test_loss, epoch)
                tb.add_scalar('validation/top1', results['top1'], epoch)
                tb.add_scalar('validation/top5', results['top5'], epoch)
                tb.add_scalar('validation/mAP', results['mAP'], epoch)
                tb.add_scalar('validation/mAUC', results['mAUC'], epoch)
                tb.add_scalar('validation/dprime',
                              results['dprime'], epoch)
                tb.flush()

            logging.info("-" * 50)
            logging.info(str(config.n_layers_pre_fusion) + "pre." +
                         str(config.n_layers_post_fusion) + "post")
            logging.info('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(
                epoch, duration, val_loss, val_loss))
            logging.info("-"*50)

            if val_loss < best_valid:
                path = save_state(state, config)
                if epoch > config.delay_stop:
                    best_valid = val_loss
                    best_epoch = epoch

                logging.info("Model saved to %s" % path)

        # save last epoch
        path = save_state(state, config)
        logging.info("Model saved to %s" % path)
        logging.info("Evaluating last epoch...")

        # eval last epoch
        _, results, truths = evaluate(state, loader['test'], config)

        results = get_vgg_stats(results, truths, config.n_classes)
        logging.info(['last epoch results', results])

        if tb:
            tb.add_scalar('test/last_top1', results['top1'], config.num_epochs)
            tb.add_scalar('test/last_top5', results['top5'], config.num_epochs)
            tb.add_scalar('test/last_mAP', results['mAP'], config.num_epochs)
            tb.add_scalar('test/last_mAUC', results['mAUC'], config.num_epochs)
            tb.add_scalar('test/last_dprime',
                          results['dprime'], config.num_epochs)

            tb.flush()
    else:
        best_epoch = config.best_epoch

    # load best epoch by validation accuracy for evaluation
    logging.info("Best epoch: %d" % best_epoch)

    saved_state, config = load_state(
        config.model, config.dataset, best_epoch, config.exp_name)

    state = init_models(config)

    state['model'].load_state_dict(saved_state['model'])
    state['optimizer'].load_state_dict(saved_state['optimizer'])

    _, results, truths = evaluate(state, loader['test'], config)

    results = get_vgg_stats(results, truths, config.n_classes)
    logging.info(['results', results])

    if tb:
        tb.add_scalar('test/top1', results['top1'], best_epoch)
        tb.add_scalar('test/top5', results['top5'], best_epoch)
        tb.add_scalar('test/mAP', results['mAP'], best_epoch)
        tb.add_scalar('test/mAUC', results['mAUC'], best_epoch)
        tb.add_scalar('test/dprime', results['dprime'], best_epoch)

        tb.flush()

    sys.stdout.flush()
