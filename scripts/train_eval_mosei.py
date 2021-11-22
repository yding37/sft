import torch
from torch import nn
import sys

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
from utils.mosei_eval import *
from utils.mixup import *

from torch.utils.data import DataLoader

import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

MOSEI_SENTI_OUTPUT_DIM = 1


def get_data(config, dataset, split='train'):
    alignment = 'na'
    data_path = os.path.join(config.data_path, dataset) + \
        f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        logging.info(f"  - Creating new {split} data")
        data = MultimodalDatasets(config, split)
        torch.save(data, data_path)
    else:
        logging.info(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def init_dataloaders(config):
    logging.info("loading data....")

    dataset = 'mosei_senti'

    train_data = get_data(config, dataset, 'train')
    valid_data = get_data(config, dataset, 'valid')
    test_data = get_data(config, dataset, 'test')

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

    # change input output dimensions based on dataset size of mosei
    config.n_train, config.n_valid, config.n_test = len(
        train_data), len(valid_data), len(test_data)

    return loaders


def init_model(model_name, config):

    video_tokenizer = FeatureTokenizer(
        500, config.input_video_dim, config.embed_dim)
    audio_tokenizer = FeatureTokenizer(
        500, config.input_audio_dim, config.embed_dim)
    text_tokenizer = TextTokenizer(50, config.input_text_dim, config.embed_dim)

    model = eval(model_name)(config)

    return FullModelShell(video_tokenizer, audio_tokenizer, text_tokenizer, model)


def init_models(config):

    logging.info("Initializing models...")

    model = None

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
        with torch.cuda.device(config.gpu_id):
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
                   }

    # prints model, optimizer, criterion, scheduler
    logging.debug(str(train_state))
    logging.info('...done')
    return train_state


def init_config(config):
    """ any additional custom settings for this dataset """
    if config.debug:
        logging.debug("Running in DEBUG mode")
        logging.debug("Reducing epochs to 5")

        config.num_epochs = 5
        config.warmup_epochs = 2

    config.dataset = str.lower(config.dataset.strip())
    config.model = config.model.strip()

    config.n_classes = 7
    config.criterion = 'cross_entropy'
    config.n_modalities = 3

    config.input_audio_dim = 74
    config.input_video_dim = 35
    config.input_text_dim = 300

    config.input_text_seq_len = 50
    config.input_audio_seq_len = 500
    config.input_video_seq_len = 500

    config.embed_dim = 40
    config.n_heads = 5

    config.permute_seq = False


    # determine stride and kernel size
    if config.custom_ks:
        # custom stride and kernel

        # if config.ks_modality != None:
        config.stride_sz = config.ks_modality.split(",")
        config.stride_sz = [int(x) for x in config.stride_sz]

        config.kernel_sz = config.stride_sz

        config.padding = [int(x) for x in config.pad_modality.split(',')]

    elif 'topk' not in config.pool_type:
        config.ks_fused = [1, 1, 1]
        rgb_len = 500
        spec_len = 500
        txt_len = 50
        config.stride_sz = [
            ((rgb_len - 1) // config.pool_k) + 1,
            ((spec_len - 1) // config.pool_k) + 1,
            ((txt_len - 1) // config.pool_k) + 1,
        ]
        config.kernel_sz = config.stride_sz

        config.padding = [
            min((config.stride_sz[0]*config.pool_k -
                 rgb_len) // 2, config.kernel_sz[0] // 2),
            min((config.stride_sz[1]*config.pool_k -
                 spec_len) // 2, config.kernel_sz[1] // 2),
            min((config.stride_sz[2]*config.pool_k -
                 txt_len) // 2, config.kernel_sz[2] // 2),
        ]

        logging.info('Kernel sizes: {}'.format(config.kernel_sz))
        logging.info('Padding: {}'.format(config.padding))
        logging.info('RGB sequence length will be reduced to {}'.format(
            (rgb_len + 2*config.padding[0]) // config.kernel_sz[0]))
        logging.info('Audio sequence length will be reduced to {}'.format(
            (spec_len + 2*config.padding[1]) // config.kernel_sz[1]))
        logging.info('Text sequence length will be reduced to {}'.format(
            (txt_len + 2*config.padding[2]) // config.kernel_sz[2]))

    logging.info("Running with config %s", str(config))

    logging.info(["MAX pool stride and filter size:",
                  config.stride_sz, config.kernel_sz])

    return config


def train_mixup(state, loader, config):

    tb = get_tb()

    epoch = state['epoch']
    model = state['model']
    optimizer = state['optimizer']
    criterion = state['criterion']
    scheduler = state['scheduler']
    # latent_stats_tracker = state['latent_stats_tracker']
    latent_stats_tracker = None

    model.train()
    if latent_stats_tracker is not None:
        prev_epoch_avg = latent_stats_tracker.current_average.clone()
        latent_stats_tracker.reset()

    epoch_loss = 0
    proc_loss, proc_size = 0, 0
    losses = []
    num_batches = config.n_train // config.batch_size

    start_time = time.time()
    for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):

        if config.debug:
            if i_batch > 10:
                break

        sample_ind, text, audio, vision = batch_X

        logging.debug(["Batches", text.shape, audio.shape, vision.shape])

        eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1

        model.zero_grad()

        if config.use_cuda:
            with torch.cuda.device(config.gpu_id):
                text, audio, vision, eval_attr = text.cuda(
                ), audio.cuda(), vision.cuda(), eval_attr.cuda()

        batch_size = text.size(0)
        if batch_size < config.batch_size:  # use full batches only, simplifes logic..
            continue

        raw_loss = 0
        net = nn.DataParallel(model) if batch_size > 10 else model

        if config.mixup != MIXUP_NONE:
            index = torch.randperm(batch_size).to(
                'cuda' if config.use_cuda else 'cpu')

            targets_a, targets_b = eval_attr, eval_attr[index]
            lam = None

            if config.alpha_type in [ALPHA_STATIC]:
                lam = sample_interpolation_param(config.alpha, batch_size)
            elif config.alpha_type in [ALPHA_PER_MODALITY, ALPHA_PER_CLASS_PER_MODALITY]:
                lam = [sample_interpolation_param(
                    config.alpha, batch_size) for m in range(config.n_modalities)]
                lam = torch.stack(lam, dim=0)

            if config.use_cuda:
                with torch.cuda.device(config.gpu_id):
                    lam = lam.cuda()

            index = torch.randperm(batch_size).to(
                'cuda' if config.use_cuda else 'cpu')

            targets_a, targets_b = eval_attr, eval_attr[index]

            logging.debug(['lambda clipped', lam])

            output = net(vision, audio, text,
                         mixup_lam=lam, mixup_index=index, latent_stats_tracker=latent_stats_tracker)


            if config.alpha_type == ALPHA_PER_MODALITY:
                lam = torch.mean(lam, dim=0)

            raw_loss = mixup_criterion(
                output, targets_a, targets_b, lam, criterion, model.training, lam_vec=True)
        else:
            # no mixup
            preds = net(vision, audio, text)
            raw_loss = criterion(preds, eval_attr)

        raw_loss.backward()
        losses.append(raw_loss.detach())

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

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
                "loss/train_iter", torch.tensor(losses).mean(), epoch*num_batches + i_batch + 1)
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
    for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):

        if config.debug:
            if i_batch > 10:
                break

        sample_ind, text, audio, vision = batch_X

        logging.debug(["Batches", vision.shape, audio.shape, text.shape])

        eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1

        model.zero_grad()

        if config.use_cuda:
            with torch.cuda.device(config.gpu_id):
                text, audio, vision, eval_attr = text.cuda(
                ), audio.cuda(), vision.cuda(), eval_attr.cuda()

        batch_size = text.size(0)
        if batch_size < config.batch_size:  # use full batches only, simplifes logic..
            continue

        raw_loss = 0
        net = nn.DataParallel(model) if batch_size > 10 else model

        preds = net(vision, audio, text)

        raw_loss = criterion(preds, eval_attr)

        raw_loss.backward()
        losses.append(raw_loss.clone().detach())

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

        proc_loss += raw_loss.detach().item() * batch_size
        proc_size += batch_size
        epoch_loss += raw_loss.detach().item() * batch_size

        avg_loss = torch.tensor(losses).mean()
        if i_batch % config.log_interval == 0:  # and i_batch > 0:
            avg_loss = proc_loss / proc_size
            elapsed_time = time.time() - start_time
            logging.info('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                         format(epoch, i_batch, num_batches, elapsed_time * 1000 / config.log_interval, avg_loss))

            tb.add_scalar(
                "loss/train_iter", torch.tensor(losses).mean(), epoch*num_batches + i_batch + 1)
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
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):

            if config.debug:
                if i_batch > 9:
                    break

            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

            if config.use_cuda:
                with torch.cuda.device(config.gpu_id):
                    text, audio, vision, eval_attr = text.cuda(
                    ), audio.cuda(), vision.cuda(), eval_attr.cuda()

            batch_size = text.size(0)

            net = nn.DataParallel(model) if batch_size > 10 else model
            preds = net(vision, audio, text)

            total_loss += criterion(preds, eval_attr).item() * batch_size

            # Collect the results into dictionary
            results.append(preds.detach())
            truths.append(eval_attr.detach())

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
            logging.debug(
                ['results, truth shape', results.shape, truths.shape])
            results = eval_mosei_senti(results, truths, True)

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
                # MOSEI original

                # MOSEI Classification
                tb.add_scalar('validation/top1', results['top1'], epoch)
                tb.add_scalar('validation/top3', results['top3'], epoch)
                tb.add_scalar('validation/mAP', results['mAP'], epoch)
                tb.add_scalar('validation/mAUC', results['mAUC'], epoch)
                tb.add_scalar('validation/prime', results['dprime'], epoch)

                tb.add_scalar('validation/acc2', results['acc2'], epoch)
                tb.add_scalar('validation/f1', results['f1'], epoch)

                tb.flush()

            logging.info("-"*50)
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

        # eval last epoch
        _, results, truths = evaluate(state, loader['test'], config)

        results = eval_mosei_senti(results, truths, config.n_classes)
        logging.info(['last epoch results', results])

        if tb:
            tb.add_scalar('test/last_top1', results['top1'], config.num_epochs)
            tb.add_scalar('test/last_top3', results['top3'], config.num_epochs)
            tb.add_scalar('test/last_mAP', results['mAP'], config.num_epochs)
            tb.add_scalar('test/last_mAUC', results['mAUC'], config.num_epochs)
            tb.add_scalar('test/last_dprime',
                          results['dprime'], config.num_epochs)
            tb.add_scalar('test/acc2', results['acc2'], config.num_epochs)
            tb.add_scalar('test/f1', results['f1'], config.num_epochs)

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

    results = eval_mosei_senti(results, truths, True)

    logging.info(['results', results])

    if tb:
        tb.add_scalar('test/top1', results['top1'], best_epoch)
        tb.add_scalar('test/top3', results['top3'], best_epoch)
        tb.add_scalar('test/mAP', results['mAP'], best_epoch)
        tb.add_scalar('test/mAUC', results['mAUC'], best_epoch)
        tb.add_scalar('test/dprime', results['dprime'], best_epoch)
        tb.add_scalar('test/acc2', results['acc2'], best_epoch)
        tb.add_scalar('test/f1', results['f1'], best_epoch)

        tb.flush()

    sys.stdout.flush()

