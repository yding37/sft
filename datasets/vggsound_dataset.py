import numpy as np
import torch
import h5py
import math
import logging


class VggH5pyDataset(torch.utils.data.Dataset):
    def __init__(self, h5py_path, ds_type='train', predefined_classes=[]):

        self.h5py_path = h5py_path

        with h5py.File(self.h5py_path, 'r') as f:
            self.n_samples = f['labels'].shape[0]

            # use all unless otherwise specified...
            ixes = list(range(self.n_samples))
            cls_names = set(f['class_name'][:])
            cls_names = sorted(cls_names)
            # logging.debug(['cls names', cls_names])
            if len(predefined_classes) > 0:
                ixes = [i for i, e in enumerate(
                    cls_names) if e in predefined_classes]
                self.label_dict = {e: i for i,
                                   e in enumerate(predefined_classes)}
            else:
                self.label_dict = {e: i for i, e in enumerate(cls_names)}

            logging.info([ds_type, 'label_dict', self.label_dict])
            self.n_samples = len(ixes)

            np.random.seed(1111)
            np.random.shuffle(ixes)

            # logging.debug(ixes)

            if ds_type == 'train':
                self.indices = ixes[:self.n_samples -
                                    math.floor(.2 * self.n_samples)]
            elif ds_type == 'valid':
                self.indices = ixes[-math.floor(.2 * self.n_samples):]
            else:
                self.indices = ixes

            self.n_samples = len(self.indices)

            # sequence lengths
            self.rgb_seq_length = f['rgb'][0].shape[0]
            self.spect_seq_length = f['spect'][0].shape[0]
            self.flow_seq_length = f['flow'][0].shape[0]

            # feature sizes
            self.rgb_feature_length = f['rgb'][0].shape[1]
            self.spect_feature_length = f['spect'][0].shape[1]
            self.flow_feature_length = f['flow'][0].shape[1]

    def __len__(self):
        return len(self.indices)

    def get_seq_lens(self):
        return self.rgb_seq_length, self.spect_seq_length, self.flow_seq_length

    def get_dims(self):
        return self.rgb_feature_length, self.spect_feature_length, self.flow_feature_length

    def __getitem__(self, index):

        h5ix = self.indices[index]

        sample = {
        }

        with h5py.File(self.h5py_path, 'r') as f:
            sample['rgb'] = f['rgb'][h5ix]
            sample['spect'] = f['spect'][h5ix]
            sample['flow'] = f['flow'][h5ix]
            # sample['label'] = torch.tensor(f['labels'][h5ix,:][0])
            sample['classname'] = f['class_name'][h5ix]
            sample['label'] = self.label_dict[sample['classname']]

        # logging.debug(['label', sample['label']])
        return sample


if __name__ == "__main__":
    fpath = '/media/scratch/datasets/vgg_train.h5py'

    ds = VggH5pyDataset(fpath)
    # print(len(ds), ds.get_dims(), ds.get_seq_lens())
    # 5748 (1024, 128, 1024) (38, 1200, 38)

    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    batch = next(iter(loader))

    for i in range(4):
        # shape of batch attributes:
        # rgb: [n, 38, 1024]
        # spect: [n, 1200, 128]
        # flow: [n, 38, 1024]
        # classname/label: [n]
        print(batch)
        break
