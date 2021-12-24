import torch
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import math
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
        dt_kernel = 0.1  # 이 값에 따라서 sample을 많이 추출할 수 있고, 적게 추출할 수 있다.(sub_sampling parameter)
        sigma = 0.1  # Kernel width => 100ms
        rate_params = {'dt': dt_kernel, 'sigma': sigma}
        session_rates = spike_counts(session, rate_params, count_type='rate',
                                     anat_region='ADn')
        counts, tmp_angles = session_rates.get_spike_matrix(
            state)  # count 변수 중 desired_nSample 만큼 slice하여 Manifold Learning 수행함.
        self.data = torch.from_numpy(counts)
        self.n_samples = counts.shape[0]
        self.dim = counts.shape[1]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n_samples

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


def get_spike_loaders(data_dir, batch_size, shuffle=True):
    dataset = SpikeData('Mouse12-120810')
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    dim = dataset.shape[1]

    return train_loader, test_loader, dim
