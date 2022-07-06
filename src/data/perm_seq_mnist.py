"""This file contains utility functions to create the Sequential or Permutated MNIST problem."""
from toolz.functoolz import compose_left

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Lambda
import hydra
import os
from utils.run_utils import seed_worker, configure_reproducibility


def get_dataset(cfg):
    def _convert_to_float32(x: np.array):
        return x.astype(np.float32)

    def _flatten(x: np.array):
        return x.reshape(784, 1)

    def _zero_out(x: np.array):
        return np.zeros(x.shape)

    def _cut(x: np.array):
        return x[:28*10]  # accorcia la sequenza a 28*10 elementi

    def _rescale(x: np.array):
        return x / 255
    
    def _shuffle_digit_array(x):
        rng = np.random.default_rng(seed=cfg.run.seed)
        # ^ the permutation should be the same for all digits
        rng.shuffle(x)
        return x

    smnist_transforms = compose_left(
        np.array,
        _rescale,
        _flatten,
        _convert_to_float32,
    )
    
    pmnist_transforms = compose_left(
        np.array,
        _rescale,
        _flatten,
        _convert_to_float32,
        _shuffle_digit_array,
    )
    if cfg.data.permute:
        transforms = pmnist_transforms
    else:
        transforms = smnist_transforms

    train = MNIST(
        root=os.path.join(cfg.run.project_path, "data/external"),
        train=True,
        download=True,
        transform=Lambda(lambda x: transforms(x)),
    )

    test = MNIST(
        root=os.path.join(cfg.run.project_path, "data/external"),
        train=False,
        download=True,
        transform=Lambda(lambda x: transforms(x)),
    )
    return train, test


def get_dataloaders(cfg, rng):
    train, _ = get_dataset(cfg)
    train.data, train.targets = train.data[:cfg.data.num_train], train.targets[:cfg.data.num_train]

    train_idx, valid_idx = get_train_valid_indices(train, cfg)

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if cfg.run.device == 'cuda':
        pin_memory = True
    else:
        pin_memory = False

    train_data_loader = DataLoader(train,
                                   batch_size=cfg.train.batch_size,
                                   shuffle=False,
                                   worker_init_fn=seed_worker,
                                   sampler=train_sampler,
                                   num_workers=1,
                                   pin_memory=pin_memory,
                                   generator=rng)  # reproducibility

    valid_data_loader = DataLoader(train,
                                   batch_size=cfg.train.batch_size,
                                   shuffle=False,
                                   worker_init_fn=seed_worker,
                                   sampler=valid_sampler,
                                   num_workers=1,
                                   pin_memory=pin_memory,
                                   generator=rng)  # reproducibility

    return train_data_loader, valid_data_loader


def get_train_valid_indices(train, cfg):
    # obtain training indices that will be used for validation
    valid_size = cfg.train.perc_valid
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx


def get_test_dataloader(cfg, rng):
    _, test = get_dataset(cfg.data.permute, cfg.run.seed)
    test.data, test.targets = test.data[:cfg.data.num_test], test.targets[:cfg.data.num_test]

    test_data_loader = DataLoader(test,
                                  batch_size=cfg.train.batch_size,
                                  shuffle=False,
                                  worker_init_fn=seed_worker,
                                  num_workers=0,
                                  generator=rng)  # reproducibility
    return test_data_loader
