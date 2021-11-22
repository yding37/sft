
import torch
import torch.nn.functional as F
import logging


def recursive_weighted_avg(prev_avg, prev_weight_sum, new_val, new_weight):
    new_weight_sum = prev_weight_sum + new_weight
    new_avg = prev_avg + (new_weight / new_weight_sum) * (new_val - prev_avg)
    return new_avg, new_weight_sum


class PerClassRunningAverage:
    def __init__(self, n_classes=10, h_dim=32, device=torch.device('cuda')):
        super(PerClassRunningAverage, self).__init__()
        self.n_classes = n_classes
        self.h_dim = h_dim
        self.device = device
        self.current_average = torch.zeros((n_classes, h_dim), dtype=torch.float32, device=device)
        self.current_weight = torch.zeros(n_classes, dtype=torch.float32, device=device)

        self.feature_queue = []

        self.pairwise_distances = torch.zeros((n_classes, n_classes), dtype=torch.float32, device=device)

        # self.mean_dists = torch.zeros((n_classes, n_classes), dtype=torch.float32, device=device)
        self.max_dist = torch.finfo(torch.float).min
        self.min_dist = torch.finfo(torch.float).max

    def reset(self):
        self.current_average = torch.zeros((self.n_classes, self.h_dim), dtype=torch.float32, device=self.device)
        self.current_weight = torch.zeros(self.n_classes, dtype=torch.float32, device=self.device)
        self.feature_queue = []

    def calc_min_max(self):
        self.max_dist = torch.finfo(torch.float).min
        self.min_dist = torch.finfo(torch.float).max

        for cl in range(self.current_average.shape[0]):
            for cl2 in range(self.current_average.shape[0]):
                if  cl != cl2:
                    # logging.debug(['d', self.current_average[cl], self.current_average[cl2]])
                    dst = F.pairwise_distance(self.current_average[cl].view(1, -1), self.current_average[cl2].view(1, -1))
                    # logging.debug([dst, self.min_dist])
                    if dst < self.min_dist:
                        self.min_dist = dst
                    
                    if dst > self.max_dist:
                        self.max_dist = dst

                    self.pairwise_distances[cl, cl2] = dst
                    self.pairwise_distances[cl2, cl] = dst

        # logging.debug(['min/max', self.min_dist, self.max_dist])

    def calc_dist_matrix(self, p=2):
        """
        Args:
            - p: int indicating which norm to use (L1, L2, etc.)
        """
        M = torch.zeros((self.n_classes, self.n_classes), dtype=torch.float32, device=self.device)
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                M[i, j] = torch.norm(self.current_average[i] - self.current_average[j], p=p, dim=0)
        return M

    def push_feature(self, feature):
        self.feature_queue.append(feature.clone().detach())

    def compute(self, targets_A, targets_B, lam):
        for i in range(len(self.feature_queue)):
            f = self.feature_queue[i]

            for j, t in enumerate(targets_A):
                t = t.long().item()
                if isinstance(lam, float):
                    l = lam
                else:
                    l = lam[j]
                self.current_average[t], self.current_weight[t] = recursive_weighted_avg(
                    self.current_average[t], self.current_weight[t], f[j], l)

            for j, t in enumerate(targets_B):
                t = t.long().item()
                if isinstance(lam, float):
                    l = lam
                else:
                    l = lam[j]
                self.current_average[t], self.current_weight[t] = recursive_weighted_avg(
                    self.current_average[t], self.current_weight[t], f[j], 1 - l)

        self.feature_queue = []


class PerClassPerModalityRunningAverage:
    def __init__(self, n_classes=10, n_modalities=3, h_dim=32, device=torch.device('cuda')):
        super(PerClassPerModalityRunningAverage, self).__init__()
        self.n_classes = n_classes
        self.n_modalities = n_modalities
        self.h_dim = h_dim
        self.device = device
        self.current_average = torch.zeros((n_modalities, n_classes, h_dim), dtype=torch.float32, device=device)
        self.current_weight = torch.zeros(n_modalities, n_classes, dtype=torch.float32, device=device)

        self.feature_queue = []

        self.pairwise_distances = torch.zeros((n_modalities, n_classes, n_classes), dtype=torch.float32, device=device)

        # self.mean_dists = torch.zeros((n_classes, n_classes), dtype=torch.float32, device=device)
        self.max_dist = [torch.finfo(torch.float).min for _ in range(self.n_modalities)]
        self.min_dist = [torch.finfo(torch.float).max for _ in range(self.n_modalities)]

    def reset(self):
        self.current_average = torch.zeros((self.n_modalities, self.n_classes, self.h_dim), dtype=torch.float32,
                                           device=self.device)
        self.current_weight = torch.zeros(self.n_modalities, self.n_classes, dtype=torch.float32, device=self.device)
        self.feature_queue = []

    def calc_min_max(self):
        self.max_dist = [torch.finfo(torch.float).min for _ in range(self.n_modalities)]
        self.min_dist = [torch.finfo(torch.float).max for _ in range(self.n_modalities)]

        for m in range(self.n_modalities):
            for cl in range(self.current_average.shape[1]):
                for cl2 in range(self.current_average.shape[1]):
                    if cl != cl2:
                        # logging.debug(['d', self.current_average[cl], self.current_average[cl2]])
                        dst = F.pairwise_distance(self.current_average[m, cl].view(1, -1),
                                                  self.current_average[m, cl2].view(1, -1))
                        # logging.debug([dst, self.min_dist])
                        if dst < self.min_dist[m]:
                            self.min_dist[m] = dst

                        if dst > self.max_dist[m]:
                            self.max_dist[m] = dst

                        
                        self.pairwise_distances[m, cl, cl2] = dst
                        self.pairwise_distances[m, cl2, cl] = dst

        # logging.debug(['min/max', self.min_dist, self.max_dist])

    def calc_dist_matrix(self, p=2):
        """
        Args:
            - p: int indicating which norm to use (L1, L2, etc.)
        """
        M = torch.zeros((self.n_modalities, self.n_classes, self.n_classes), dtype=torch.float32, device=self.device)
        for i in range(self.n_modalities):
            for j in range(self.n_classes):
                for k in range(self.n_classes):
                    M[i, j, k] = torch.norm(self.current_average[j] - self.current_average[k], p=p, dim=0)
        return M

    def push_feature(self, feature):
        self.feature_queue.append(feature.clone().detach())

    def compute(self, targets_A, targets_B, lam):
        for i in range(len(self.feature_queue)):
            f = self.feature_queue[i]

            for j, t in enumerate(targets_A):
                t = t.long().item()
                if isinstance(lam, float):
                    l = lam
                elif lam.shape[0] == targets_A.shape[0]:
                    l = lam[j]
                elif lam.shape[0] == self.n_modalities:
                    l = lam[i, j]
                else:
                    raise Exception
                self.current_average[i, t], self.current_weight[i, t] = recursive_weighted_avg(
                    self.current_average[i, t], self.current_weight[i, t], f[j], l)

            for j, t in enumerate(targets_B):
                t = t.long().item()
                if isinstance(lam, float):
                    l = lam
                elif lam.shape[0] == targets_A.shape[0]:
                    l = lam[j]
                elif lam.shape[0] == self.n_modalities:
                    l = lam[i, j]
                else:
                    raise Exception
                self.current_average[i, t], self.current_weight[i, t] = recursive_weighted_avg(
                    self.current_average[i, t], self.current_weight[i, t], f[j], 1 - l)

        self.feature_queue = []
