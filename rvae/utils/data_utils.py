import torch
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import sys

gen_fn_dir = os.path.abspath('..') + '/shared_scripts'
sys.path.append(gen_fn_dir)
from binned_spikes_class import spike_counts
import general_file_fns as gff

gen_params = gff.load_pickle_file('../general_params/general_params.p')
session = 'Mouse12-120810'
state = 'Wake'


class SpikeData(torch.utils.data.Dataset):
    def __init__(self, session):
        dt_kernel = 0.1  # 이 값에 따라서 sample 을 많이 추출할 수 있고, 적게 추출할 수 있다.(sub_sampling parameter)
        sigma = 0.1  # Kernel width => 100ms
        rate_params = {'dt': dt_kernel, 'sigma': sigma}
        session_rates = spike_counts(session, rate_params, count_type='rate',
                                     anat_region='ADn')
        counts, tmp_angles = session_rates.get_spike_matrix(
            state)
        print(counts.shape)
        tmp_angles = np.array(tmp_angles)
        self.feature_data = torch.from_numpy(counts).float()
        print(self.feature_data.shape)
        self.label_data = torch.from_numpy(tmp_angles).float()
        self.label_data = torch.reshape(self.label_data, (-1, 1))
        self.n_samples = counts.shape[0]
        self.dim = counts.shape[1]
        self.feature_data = torch.reshape(self.feature_data, (-1, self.dim))

    def __getitem__(self, item):
        return self.feature_data[item], self.label_data[item]

    def __len__(self):
        return self.n_samples


def get_spike_loaders(data_dir, batch_size, shuffle=True):
    data_session = 'Mouse12-120810'
    dataset = SpikeData(session=data_session)
    test_split = .2
    random_seed = 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset=SpikeData(session=data_session), sampler=train_sampler, batch_size=batch_size)
    test_loader = DataLoader(dataset=SpikeData(session=data_session), sampler=test_sampler, batch_size=batch_size)
    dim = dataset.dim

    return train_loader, test_loader, dim


class CircleData(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def get_fmnist_loaders(data_dir, batch_size, shuffle=True):
    """Helper function that deserializes FashionMNIST data 
    and returns the relevant data loaders.

    params:
        data_dir:    string - root directory where the data will be saved
        b_sz:        integer - the batch size
        shuffle:     boolean - whether to shuffle the training set or not
    """
    train_loader = DataLoader(
        FashionMNIST(data_dir, train=True, transform=ToTensor(), download=True),
        shuffle=shuffle, batch_size=batch_size)
    test_loader = DataLoader(
        FashionMNIST(data_dir, train=False, transform=ToTensor(), download=True),
        shuffle=False, batch_size=batch_size)

    return train_loader, test_loader


def get_mnist_loaders(data_dir, b_sz, shuffle=True):
    """Helper function that deserializes MNIST data 
    and returns the relevant data loaders.

    params:
        data_dir:    string - root directory where the data will be saved
        b_sz:        integer - the batch size
        shuffle:     boolean - whether to shuffle the training set or not
    """
    train_loader = DataLoader(
        MNIST(data_dir, train=True, transform=ToTensor(), download=True),
        shuffle=shuffle, batch_size=b_sz)
    test_loader = DataLoader(
        MNIST(data_dir, train=False, transform=ToTensor(), download=True),
        shuffle=False, batch_size=b_sz)

    return train_loader, test_loader


def get_kmnist_loaders(data_dir, b_sz, shuffle=True):
    """Helper function that deserializes KMNIST data 
    and returns the relevant data loaders.

    params:
        data_dir:    string - root directory where the data will be saved
        b_sz:        integer - the batch size
        shuffle:     boolean - whether to shuffle the training set or not
    """
    train_loader = DataLoader(
        KMNIST(data_dir, transform=ToTensor(), download=True),
        shuffle=shuffle, batch_size=b_sz)
    test_loader = DataLoader(
        KMNIST(data_dir, train=False, transform=ToTensor(), download=True),
        shuffle=False, batch_size=b_sz)

    return train_loader, test_loader


def get_circle_loaders(data_dir, b_sz, shuffle=True):
    train_loader = DataLoader(CircleData(data_dir + "circle_train.ptc"), batch_size=b_sz, shuffle=True)
    test_loader = DataLoader(CircleData(data_dir + "circle_test.ptc"), batch_size=b_sz, shuffle=False)

    return train_loader, test_loader
