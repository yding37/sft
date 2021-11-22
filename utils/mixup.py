import numpy as np
import torch
import logging
import torch.nn.functional as F


MIXUP_WITHIN_MODALITY = 'within'
MIXUP_CROSS_MODALITY = 'cross'
MIXUP_FUSED = 'fused'
MIXUP_ANY_LAYER = 'any'
MIXUP_TEMPORAL = 'temporal'
MIXUP_BOTTLENECK = 'bottleneck'
MIXUP_DYNAMIC = 'dynamic'
MIXUP_NONE = 'none'

MIXUP_TYPES = {
    MIXUP_WITHIN_MODALITY,
    MIXUP_CROSS_MODALITY,
    MIXUP_ANY_LAYER,
    MIXUP_FUSED,
    MIXUP_TEMPORAL,
    MIXUP_BOTTLENECK,
    MIXUP_DYNAMIC,
    MIXUP_NONE
}

ALPHA_STATIC = 'static'
ALPHA_PER_CLASS = 'perclass'
ALPHA_PER_MODALITY = 'permodality'
ALPHA_PER_CLASS_PER_MODALITY = 'perclass_permodality'
ALPHA_PRESET_PER_CLASS = 'preset_perclass'
ALPHA_PRESET_PER_CLASS_PER_MODALITY = 'preset_perclass_permodality'

ALPHA_TYPES = {
    ALPHA_STATIC,
    ALPHA_PER_CLASS,
    ALPHA_PER_MODALITY,
    ALPHA_PER_CLASS_PER_MODALITY,
    ALPHA_PRESET_PER_CLASS,
    ALPHA_PRESET_PER_CLASS_PER_MODALITY,
}


def sample_interpolation_param(alpha, batch_sz):

    lam = torch.tensor([np.random.beta(alpha, alpha) for _ in range(batch_sz)])
    return lam


def sample_dynamic_mixup_mosei(labels, shuffled_labels, alpha_scale=2.0):
    mixup_alpha = (
        1 - ((torch.abs(labels - shuffled_labels)) / 6) + .2) / alpha_scale

    logging.debug(['alphas', mixup_alpha])
    lams = []
    for i in range(mixup_alpha.shape[0]):
        lams.append(np.random.beta(mixup_alpha[i], mixup_alpha[i]))

    return lams


def sample_interpolation_param_with_sample_difference(labels, shuffled_labels, class_centers, min_dist, max_dist, alpha_scale=2.0):

    dist = torch.pairwise_distance(
        class_centers[labels], class_centers[shuffled_labels])
    lam = torch.zeros_like(dist)
    means = torch.zeros_like(dist)
    for i in range(labels.shape[0]):
        alpha = (dist[i] - min_dist) / torch.clip(max_dist - min_dist, .01)
        alpha = (1 - torch.clip(alpha, 0, .99)) / alpha_scale
        means[i] = alpha

        logging.debug(['slpha', alpha])

        beta = torch.distributions.Beta(alpha, alpha)
        lam[i] = beta.sample()

    logging.debug(['alpha sampling params', alpha.detach()])

    return lam, alpha

def sample_interpolation_param_with_class_centers(labels, shuffled_labels, class_centers, min_dist, max_dist, alpha_scale=2.0):

    dist = torch.pairwise_distance(
        class_centers[labels], class_centers[shuffled_labels])
    lam = torch.zeros_like(dist)
    means = torch.zeros_like(dist)
    for i in range(labels.shape[0]):
        alpha = (dist[i] - min_dist) / torch.clip(max_dist - min_dist, .01)
        alpha = (1 - torch.clip(alpha, 0, .99)) / alpha_scale
        means[i] = alpha

        logging.debug(['slpha', alpha])

        beta = torch.distributions.Beta(alpha, alpha)
        lam[i] = beta.sample()

    logging.debug(['alpha sampling params', alpha.detach()])

    # # similarity is always 0, do this by scaling distances?
    # average_errors = torch.abs((class_centers[labels] - class_centers[shuffled_labels]).sum(-1))
    # logging.debug(['err', average_errors])

    # pairwise_dist = torch.pairwise_distance(class_centers[labels], class_centers[shuffled_labels])
    # cosine_sim = torch.cosine_similarity(class_centers[labels], class_centers[shuffled_labels], dim=1)

    # logging.debug(['cosin sim, min, max', cosine_sim])
    # #min-max scaling
    # cosine_sim = torch.clip(cosine_sim, 1e-3)
    # lam = torch.zeros_like(cosine_sim)
    # for i in range(labels.shape[0]):
    #     beta = torch.distributions.Beta(cosine_sim[i], cosine_sim[i])
    #     lam[i] = beta.sample()

    return lam, alpha


def sample_interpolation_param_with_alpha_presets(labels, shuffled_labels, alpha_presets):
    lam = torch.zeros(labels.shape[0], dtype=torch.float32, device=labels.device)
    for i in range(labels.shape[0]):
        alpha = alpha_presets[labels[i], shuffled_labels[i]]

        logging.debug(['slpha', alpha])

        beta = torch.distributions.Beta(alpha, alpha)
        lam[i] = beta.sample()

    return lam


def mixup_criterion(pred, y_a, y_b, lam, criterion, training, lam_vec=False):
    if training and not lam_vec:
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    elif training and lam_vec:
        val = lam * criterion(pred, y_a, reduction='none') + \
            (1 - lam) * criterion(pred, y_b, reduction='none')
        return val.mean()
    else:
        return criterion(pred, y_a)


def mixup(x, lam, index, training, permute_seq=False, batch_first=False):
    # print(lam.shape, x.shape, index.shape)
    if training:
        lam = torch.tensor(lam, dtype=torch.float).view(-1, 1).cuda()
        logging.debug(['mixlamb', lam.shape])

        if batch_first:
            # print('mixup shape: ', x.shape)
            x = x.permute(1,0,2)
            x = lam * x + (1 - lam) * x[:, index, :]
            # print(x.shape)
            x = x.permute(1,0,2)

            return x

        if permute_seq:
            logging.debug(['permuting...'])
            seqs = torch.cat([torch.Tensor([0]), torch.randperm(x.shape[0] - 1) + 1]).long()
            # logging.debug(['seqs len', seqs.shape, x.shape, x[seqs, :, :].shape])
            y = x[seqs,:,:]
            return lam * x + (1 - lam) * y[:, index, :]
        else:
            # logging.debug(['nah...'])
            return lam * x + (1 - lam) * x[:, index, :]

    else:
        return x


# def mixup_cross_modal_criterion(pred, targets, lam, indexes, criterion, training):
#     if training:
#         n_modalities = len(targets)

#         ret = lam * criterion(pred, targets[0])

#         lam_norm = (1.0 - lam) / n_modalities
#         for i in range(1, n_modalities):
#             ret += lam_norm * criterion(pred, targets[i])

#         return ret
#     else:
#         raise Exception("don't use cross modal criterion for eval")


def mixup_cross_modal(inputs, lam, indexes, training):
    if training:

        n_inputs = len(inputs)

        output = inputs[0] * lam

        lam_norm = (1.0 - lam) / n_inputs
        for i in range(1, n_inputs):
            logging.debug(['cross', i])
            output += lam_norm * inputs[i][:, indexes, :]

        return output

    else:
        return inputs[0]
