#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from update import DatasetSplit



def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    assert args.num_classes == 2 or 10

    if args.dataset == 'cifar':
        data_dir = args.data_path+'cifar/'
        trf = []
        if args.model == 'vgg' or 'resnet': 
            trf = [transforms.Resize(size=(224, 224))]
        else:
            trf = []
        trf += [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        apply_transform = transforms.Compose(trf)

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = args.data_path+'mnist/'
        else:
            data_dir = args.data_path+'fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=False,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=False,
                                      transform=apply_transform)

    else:
        raise ValueError()

    N = args.num_samples*args.num_users
    if args.num_classes == 2:
        class1 = 1
        class2 = 6
        idxs = ((train_dataset.targets == class1) + (train_dataset.targets == class2)).nonzero().view(-1)
        idxs_test = ((test_dataset.targets == class1) + (test_dataset.targets == class2)).nonzero().view(-1) 
        train_dataset.targets[train_dataset.targets == class1] = 0
        train_dataset.targets[train_dataset.targets == class2] = 1
        test_dataset.targets[test_dataset.targets == class1] = 0
        test_dataset.targets[test_dataset.targets == class2] = 1

        if N < len(idxs):
            idxs = idxs[:N]
        elif N > len(idxs):
            raise ValueError()

    else:
        if N < len(train_dataset.targets):
            idxs = torch.randperm(len(train_dataset.targets))[:N]
            train_dataset = DatasetSplit(train_dataset, idxs)
        elif N > len(train_dataset.targets):
            raise ValueError()

    # if args.model == 'svm':
    #     # print(train_dataset.data.size())
    #     train_dataset.data = torch.flatten(train_dataset.data, start_dim=1)
    #     test_dataset.data = torch.flatten(test_dataset.data, start_dim=1)

    # Sample training data amongst users
    if args.iid: #Sample IID user data
        split_fn = mnist_iid if args.dataset == "mnist" else cifar_iid
    else: #Sample Non-IID user data from MNIST
        if args.unequal: #Choose unequal splits for every user
            split_fn = mnist_noniid_unequal
        else: #Choose equal splits for every user
            split_fn = mnist_noniid if args.dataset == "mnist" else cifar_noniid
    user_groups = split_fn(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w, boot_weights=None):
    """
    Returns the average of the weights.
    w: list of Tensor?
    """

    w_avg = copy.deepcopy(w[0])
    avg_weights = torch.arange(len(w)) if boot_weights is None else boot_weights
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[avg_weights[i].item()][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg


def generate_perturbations(model_list, num_init=10):
    init = average_weights(model_list)
    K = len(model_list)
    perturb_list = []
    for _ in range(num_init):
        weights = torch.multinomial(torch.Tensor([1./K]).repeat(K), K)
        boot_init = average_weights(model_list, weights)
        delta_0 = dict().fromkeys(init.keys())
        for key in init.keys():
            delta_0[key] = boot_init[key] - init[key] # boot_init - init 
        perturb_list.append(delta_0)

    return perturb_list


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
