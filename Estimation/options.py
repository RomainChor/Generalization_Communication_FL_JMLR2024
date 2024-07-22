#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--epochs', type=int, default=1)
    # parser.add_argument('--bs', type=int, default=1,
                        # help="batch size")


    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=2,
                        help="number of communication rounds: R")
    parser.add_argument('--rounds_values', nargs='+', type=int, default=None)
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--num_samples', type=int, default=100,
                        help="number of samples per user: n")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="number of local epochs: e")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--global_ep', type=int, default=1,
                        help='number of global epochs: E')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--ep_values', nargs='+', type=int, default=None)
    parser.add_argument('--total_ep', type=int, default=50)
    

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--svm_kernel', type=str, default='linear',
                        help='SVM kernel name')
    parser.add_argument('--gamma_init', type=float, default=0.01,
                        help='SVM Gaussian kernel parameter')
    parser.add_argument('--train_gamma', type=str2bool, default=True,
                        help='whether or not to train gamma parameter')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    parser.add_argument('--num_init', type=int, default=10)
    parser.add_argument('--bias', type=str2bool, default=True)
    # parser.add_argument('--reg', type=str2bool, default=False,
    #                     help="Whether to train with regularization")
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--beta_values', nargs='+', type=float, default=None)

    # other arguments
    parser.add_argument('--save_path', type=str, default='./save/', help="")
    parser.add_argument('--data_path', type=str, default='./data/', help="")
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--validation', type=str2bool, default=False)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--mc', type=int, default=3)
    # parser.add_argument('--gpu_id', default=None, help="To use cuda, set \
                        # to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--n_estim', type=int, default=5,
                        help='Number of estimations of expectation.')
    parser.add_argument('--info_eval_freq', type=int, default=0,
                        help="CMI evaluation frequency (epoch-wise). 0 is no evaluation, -1 is at last local epoch.")
    parser.add_argument('--gpu', type=str2bool, default=False, help="Whether or not to use cuda. \
                        Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')


    args = parser.parse_args()
    
    return args
