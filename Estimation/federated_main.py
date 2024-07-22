#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import copy
import time
import warnings 
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib
import torch.nn as nn
from torchvision import models
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, KernelSVM
from utils import get_dataset, average_weights, generate_perturbations, exp_details


matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def run_federated(args, data=None):
    # Load dataset and user groups
    if data is not None:
        train_dataset = data["train"]
        test_dataset = data["test"]
        user_groups = data["groups"]
    else:
        train_dataset, test_dataset, user_groups = get_dataset(args)

    idxs = set()
    for idx in user_groups.values():
        idxs.update(idx)
    idxs = list(idxs)

    # idxs = set()
    # for idx in user_groups.values():
    #     idxs.update(idx)
    # idxs = list(idxs)
    # print(len(idxs))

    # BUILD MODEL
    if args.model == 'cnn': #Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp': # Multi-layer perceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=256, 
                            dim_out=args.num_classes, bias=args.bias)
    elif args.model == 'svm':
        global_model = KernelSVM(train_data_x=train_dataset.data, 
                                 args=args)
    elif args.model == 'vgg':
        global_model = models.vgg11_bn(pretrained=True)
        # global_model = models.vgg16(pretrained=True)
        input_lastLayer = global_model.classifier[-1].in_features   
        global_model.classifier[6] = nn.Linear(input_lastLayer, args.num_classes)
    elif args.model == 'resnet':
        global_model = models.resnet18(pretrained=True)
        input_lastLayer = global_model.fc.in_features   
        global_model.fc = nn.Linear(input_lastLayer, args.num_classes)
    else:
        exit('Error: unrecognized model')

    if args.info_eval_freq != 0 and args.beta > 0:
        raise ValueError("info_eval_freq can not be != 0 while beta is > 0.")
    

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    print("===========================")
    
    global_weights = global_model.state_dict() # Copy weights

    # TRAINING ===============================================================
    print_every = 1
    # val_loss_pre, counter = 0, 0

    # Initialize client instances
    local_models = []
    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_dataset, 
                                client_idx=c, idxs=user_groups[c], logger=logger)
        local_models.append(local_model)

    local_states = []
    for _ in range(args.num_users):
        perturb = copy.deepcopy(global_weights)
        for key in perturb.keys(): #Generate a random initialization (round 0) for each local model
            perturb[key] += (torch.div(2*torch.rand(perturb[key].size()), args.num_users).long()).to(device)
        local_states.append(perturb)
    
    # Training FL model
    # info_values, bound_values = [], []
    # gen_values = []
    # train_acc_values, test_acc_values = [], []
    # train_loss_values, test_loss_values = [], []
    info_estim = 0
    for e in range(1, args.global_ep+1):
        print("Global epoch {}".format(e))

        # info_estim = 0 #RESET AFTER EACH GLOBAL EPOCH
        stream = tqdm(range(args.rounds), ncols=100)
        for round in stream:
            # stream.set_description("Communication round: {}".format(round+1))
            local_weights, local_losses = [], []

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)  

            perturb_list = generate_perturbations(local_states, args.num_init) #Initial perturbations

            for i in idxs_users:
                w, loss, info = local_models[i].update_weights(
                    model=copy.deepcopy(global_model), 
                    global_round=round, 
                    perturb_list=perturb_list
                )
                info_estim += info
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # Update global weights
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            local_states = copy.deepcopy(local_weights)

            # Compute avg train/val error and accuracy of global model at each round
            if args.verbose == 1:
                # list_acc, list_loss = [], []
                global_model.eval()
                train_accuracy, train_loss = test_inference(args, global_model, train_dataset, idxs)

                desc = f"Communication Round: {round+1} | "
                desc += "TRAIN Loss/Accuracy: {0:.2f}/{1:.2f}% | ".format(train_loss, 100*train_accuracy)
                if e % args.eval_freq == 0:
                    val_accuracy, val_loss = test_inference(args, global_model, test_dataset)
                    desc += "VAL Loss/Accuracy: {0:.2f}/{1:.2f}%".format(val_loss, 100*val_accuracy)

                    gen = train_accuracy - val_accuracy
                    bound_estim = np.sqrt(2*info_estim/(args.num_users*args.num_samples))
                    print("\n Gen. error: {0:.4f} | Bound estim.: {1:.4f}".format(gen, bound_estim))
                stream.set_description(desc)

    # ========================================================================
    # TEST INFERENCE after completion of training ============================
    train_accuracy, train_loss = test_inference(args, global_model, train_dataset, idxs)
    test_accuracy, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Inference results after {args.global_ep*args.rounds} communication rounds:')
    print("|---- Train Accuracy/Loss: {0:.2f}%/{1:.2f}".format(100*train_accuracy, train_loss))
    print("|---- Test Accuracy/Loss: {0:.2f}%/{1:.2f}".format(100*test_accuracy, test_loss))

    print('\n Total Run Time: {0:0.2f}s'.format(time.time()-start_time))
    # END OF INFERENCE =======================================================

    return train_accuracy, test_accuracy, train_loss, test_loss, bound_estim


def comparison_rounds(args):
    df = pd.DataFrame()
    df["ep_values"] = args.ep_values
    df["emp_risk"] = 0
    df["risk"] = 0
    df["bound"] = 0
    df["train_loss"] = 0
    df["test_loss"] = 0

    for m in range(args.mc):
        train_dataset, test_dataset, user_groups = get_dataset(args) #Expectation over S
        data = {"train":train_dataset, "test":test_dataset, "groups":user_groups}
        
        emp_risk_values, risk_values, bound_values = [], [], []
        train_loss_values, test_loss_values = [], []
        # for R in args.rounds_values:
        for ep in args.ep_values:
            args.local_ep = args.total_ep // ep
            args.global_ep = ep
            args.rounds = args.rounds_values[0] 
            print("\n Local ep = {} | Ep = {}".format(args.local_ep, ep))
            train_acc, test_acc, train_loss, test_loss, bound_estim = run_federated(args, data)
            emp_risk_values.append(1 - train_acc)
            risk_values.append(1 - test_acc)
            train_loss_values.append(train_loss)
            test_loss_values.append(test_loss)
            bound_values.append(bound_estim)
        df["emp_risk"] = emp_risk_values
        df["risk"] = risk_values
        df["bound"] = bound_values
        df["train_loss"] = train_loss_values
        df["test_loss"] = test_loss_values
        df.to_pickle(args.save_path+'values_{}_{}_K{}_n{}_b{}_E{}_eta{}_{}.pickle'.format(
            args.dataset, 
            args.model, 
            args.num_users, 
            args.num_samples, 
            args.local_bs, 
            args.global_ep,
            args.lr,
            m+1
        ))

def comparison_reg(args):
    df = pd.DataFrame()
    df["beta_values"] = args.beta_values
    df["emp_risk"] = 0
    df["risk"] = 0
    df["bound"] = 0
    df["train_loss"] = 0
    df["test_loss"] = 0

    for m in range(args.mc):
        train_dataset, test_dataset, user_groups = get_dataset(args) #Expectation over S
        data = {"train":train_dataset, "test":test_dataset, "groups":user_groups}
        
        emp_risk_values, risk_values, bound_values = [], [], []
        train_loss_values, test_loss_values = [], []
        for beta in args.beta_values:
            print("\n beta = {}".format(beta))
            args.beta = beta

            train_acc, test_acc, train_loss, test_loss, bound_estim = run_federated(args, data)
            emp_risk_values.append(1 - train_acc)
            risk_values.append(1 - test_acc)
            train_loss_values.append(train_loss)
            test_loss_values.append(test_loss)
            bound_values.append(bound_estim)
        df["emp_risk"] = emp_risk_values
        df["risk"] = risk_values
        df["bound"] = bound_values
        df["train_loss"] = train_loss_values
        df["test_loss"] = test_loss_values
        df.to_pickle(args.save_path+'comp_beta_values_{}_{}_K{}_n{}_e{}_b{}_E{}_eta{}_{}.pickle'.format(
            args.dataset, 
            args.model, 
            args.num_users, 
            args.num_samples, 
            args.local_ep, 
            args.local_bs, 
            args.global_ep,
            args.lr,
            m+1
        ))





if __name__ == '__main__':
    start_time = time.time()

    # define paths
    # project_path = os.path.abspath('..')
    logger = SummaryWriter('./logs/')

    args = args_parser()
    exp_details(args)

    device = 'cuda' if args.gpu else 'cpu'
    print("Using "+device)
    # torch.device(device)

    if (args.rounds_values is not None) and (args.beta_values is not None):
        raise ValueError

    if args.rounds_values is None:
        if args.beta_values is not None:
            comparison_reg(args)
        else:
            run_federated(args)

    else:
        comparison_rounds(args)

