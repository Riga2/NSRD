import os

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, Sampler
import copy
from typing import Callable
import random

class SubsetSequenceSampler(Sampler):
    def __init__(self, indices):
        super(SubsetSequenceSampler, self).__init__(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class RenderingDataLoader(DataLoader):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.batch_size = 1 if args.test_only else args.batch_size
        self.number_previous_frames = args.num_pre_frames
        self.num_frames_samples = args.num_frames_samples
        self.test_every = args.test_every
        self.n_samples = args.total_folder_num * args.grain

        self.valid_range = []
        if args.test_only is False:
            valid_folders = np.random.choice(np.arange(0, args.total_folder_num), args.valid_folder_num, replace=False)
            for i in valid_folders:
                self.valid_range += range(i * args.grain, (i+1) * args.grain)

        self.test_range = []
        if args.test_folder is not None:
            for i in args.test_folder:
                self.test_range += range(i * args.grain, (i+1) * args.grain)

        self.idx_train = []
        for idx in range(self.n_samples):
            if (idx not in self.valid_range) and (idx not in self.test_range):
                idx_mod = idx % args.grain
                if (idx_mod >= self.number_previous_frames) and (idx_mod <= args.grain - self.num_frames_samples):
                    self.idx_train.append(idx)

        self.idx_valid = []
        for idx in self.valid_range:
            idx_mod = idx % args.grain
            if (idx_mod >= self.number_previous_frames) and (idx_mod <= args.grain - self.num_frames_samples):
                self.idx_valid.append(idx)

        self.idx_test = []
        for idx in self.test_range:
            idx_mod = idx % args.grain
            if (idx_mod >= self.number_previous_frames) and (idx_mod <= args.grain - self.num_frames_samples):
                self.idx_test.append(idx)

        self.sampler, self.valid_sampler = self._split_sampler()

        init_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'num_workers': args.n_threads
        }
        super().__init__(sampler=self.sampler, **init_kwargs, drop_last=True)

    def _split_sampler(self):
        # For test
        if len(self.valid_range) == 0:
            train_sampler = SubsetSequenceSampler(self.idx_test)
            return train_sampler, None

        repeat = (self.batch_size * self.test_every) // len(self.idx_train)
        train_idx = np.repeat(self.idx_train, repeat)

        np.random.seed(0)
        np.random.shuffle(train_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSequenceSampler(self.idx_valid)
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self, valid_dataset):
        if self.valid_sampler is None:
            return None
        else:
            valid_dataloader = DataLoader(sampler=self.valid_sampler, dataset=valid_dataset, batch_size=1, num_workers=self.num_workers, drop_last=True)
            return valid_dataloader
