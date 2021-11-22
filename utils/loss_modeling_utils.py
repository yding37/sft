import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import torch
import torch.nn.functional as F
import torch.nn as nn

###################### LOSS MODELING AND MIXUP ######################


def reg_loss_class(mean_tab, num_classes=10):
    loss = 0
    for items in mean_tab:
        loss += (1./num_classes)*torch.log((1./num_classes)/items)
    return loss


def compute_probabilities_batch(text, audio, vision, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    model.eval()
    output, hiddens = model(text, audio, vision)
    # outputs = F.log_softmax(outputs, dim=1)
    batch_losses = F.l1_loss(output, target.long(), reduction='none')
    batch_losses.detach_()
    output.detach_()
    model.train()
    batch_losses = (batch_losses - bmm_model_minLoss) / \
        (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
    batch_losses[batch_losses >= 1] = 1-10e-4
    batch_losses[batch_losses <= 0] = 10e-4

    #B = bmm_model.posterior(batch_losses,1)
    B = bmm_model.look_lookup(
        batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

    return torch.FloatTensor(B)


# def mixup_criterion(pred, y_a, y_b, lam):
#     y_a, y_b = y_a.long(), y_b.long()
#     return lam * F.l1_loss(pred, y_a) + (1 - lam) * F.l1_loss(pred, y_b)

################### CODE FOR THE BETA MODEL  ########################


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan,
                          self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l  # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def track_training_loss(model, use_cuda, loader):
    model.eval()

    all_losses = torch.Tensor()
    all_predictions = torch.Tensor()
    all_probs = torch.Tensor()
    all_argmaxXentropy = torch.Tensor()

    for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
        sample_ind, text, audio, vision = batch_X
        target = batch_Y.squeeze(-1)   # if num of labels is 1

        model.zero_grad()

        if use_cuda:
            with torch.cuda.device(0):
                text, audio, vision, target = text.cuda(
                ), audio.cuda(), vision.cuda(), target.cuda()

        batch_size = text.size(0)

        raw_loss = 0
        net = nn.DataParallel(model) if batch_size > 10 else model

        preds, hiddens = net(text, audio, vision)

        idx_loss = F.l1_loss(preds, target.long(), reduction='none')
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
        probs = preds.clone()
        probs.detach_()
        all_probs = torch.cat((all_probs, probs.cpu()))
        arg_entr = torch.max(preds, dim=1)[1]
        arg_entr = arg_entr.reshape(-1, 1)
        arg_entr = F.l1_loss(preds,
                             arg_entr.to('cuda' if use_cuda else 'cpu'), reduction='none')
        arg_entr.detach_()
        all_argmaxXentropy = torch.cat((all_argmaxXentropy, arg_entr.cpu()))

    loss_tr = all_losses.data.numpy()

    # outliers detection
    max_perc = np.percentile(loss_tr, 95)
    min_perc = np.percentile(loss_tr, 5)
    loss_tr = loss_tr[(loss_tr <= max_perc) & (loss_tr >= min_perc)]
    
    bmm_model_maxLoss = torch.FloatTensor([max_perc]).to('cuda' if use_cuda else 'cpu')
    bmm_model_minLoss = torch.FloatTensor([min_perc]).to('cuda' if use_cuda else 'cpu') + 10e-6

    loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / \
        (bmm_model_maxLoss.data.cpu().numpy() -
         bmm_model_minLoss.data.cpu().numpy() + 1e-6)

    loss_tr[loss_tr >= 1] = 1-10e-4
    loss_tr[loss_tr <= 0] = 10e-4

    bmm_model = BetaMixture1D(max_iters=10)
    bmm_model.fit(loss_tr)

    bmm_model.create_lookup(1)

    return bmm_model, bmm_model_maxLoss, bmm_model_minLoss
