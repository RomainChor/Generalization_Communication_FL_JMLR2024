#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import copy
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import HingeLoss
from nngeometry.object import PVector 


# torch.utils.data.Subset()!
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.

    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
    

class CMILoss(nn.Module):
    def __init__(self, beta=0, reduction='mean'):
        super(CMILoss, self).__init__()
        self.beta = beta
        self.base_loss = nn.CrossEntropyLoss(reduction=reduction)
        # self.base_loss = nn.functional.cross_entropy

    def forward(self, preds, targets, reg=torch.zeros(1)):
        loss = self.base_loss(preds, targets)
        if self.beta > 0:
            loss = loss + self.beta*reg.to(loss.device)

        return loss


def get_criterion(args, reduction='mean') :
    if args.model == 'svm':
        if args.num_classes == 2:
            return HingeLoss(task="binary", reduction=reduction)
        else:
            return HingeLoss(task="multiclass", reduction=reduction)
    # # return nn.NLLLoss()
    # return nn.CrossEntropyLoss(reduction=reduction)
    return CMILoss(beta=args.beta, reduction=reduction)


def get_pred_fn(args):
    if (args.num_classes == 2) and (args.model == 'svm'):
        return lambda x: x > 0
    
    return lambda x: torch.max(x, 1)[1]


class LocalUpdate(object):
    def __init__(self, args, dataset, client_idx, idxs, logger):
        self.args = args
        self.logger = logger
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(
        #     dataset, list(idxs))
        self.client_idx = client_idx
        self.trainloaders, self.validloader = self.train_val_test(
            dataset, list(idxs), valid=args.validation)
        
        # self.infoloaders, _ = self.train_val_test(dataset, list(idxs), valid=False, n_estim=self.args.n_estim)
        # self.fisherloaders, _ = self.train_val_test(dataset, list(idxs), valid=False, n_estim=-1)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = get_criterion(args).to(self.device) # Default criterion set to NLL loss function
        self.no_reduc_criterion = get_criterion(args, reduction='none').to(self.device)
        self.pred_fn = get_pred_fn(args)

        self.info_estim = 0
        self.regul = torch.zeros(1).to(self.device)

        if args.info_eval_freq == -1: 
            self.info_eval_freq = args.local_ep #Only at the end
        elif args.info_eval_freq == 0:
            self.info_eval_freq = 1e+9 #Arbitrary large value
        else:
            self.info_eval_freq = args.info_eval_freq
        


    def train_val_test(self, dataset, idxs, valid, n_estim=0):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # Split indexes for train, validation, and test (80, 10, 10)
        if valid:
            idxs_train = idxs[:int(0.8*len(idxs))]
            idxs_val = idxs[int(0.8*len(idxs)):]
        else:
            idxs_train = idxs

        idxs_rounds = np.array_split(np.array(idxs_train), self.args.rounds)
        if n_estim == 0:
            bs = self.args.local_bs
        elif n_estim == -1:
            bs = 1
        elif n_estim > len(idxs_rounds[0]):
            bs = len(idxs_rounds[0])
        else:
            bs = len(idxs_rounds[0]) // n_estim

        # samplers = [RandomSampler(DatasetSplit(dataset, idxs_rounds[r])) for r in range(self.args.rounds)]
        trainloaders_list = [DataLoader(DatasetSplit(dataset, idxs_rounds[r]),
                                batch_size=bs, shuffle=True) for r in range(self.args.rounds)]

        if valid:
            validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                    batch_size=int(len(idxs_val)/10), shuffle=False)
        else:
            validloader = None
        
        return trainloaders_list, validloader


    def update_weights(self, model, global_round, perturb_list=None, no_bp=True):
        if self.info_eval_freq != 0 and perturb_list is None:
            raise ValueError("'perturb_list' argument is None!")
        
        L = len(perturb_list)

        model.train() #Set mode to train model
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        init = nn.utils.parameters_to_vector(model.parameters())
        if self.args.beta == 0.0:
            init = init.detach()


        temp_model = copy.deepcopy(model)
        delta_0_list = []
        delta_list = []
        for state in perturb_list:
            temp_model.load_state_dict(state)
            # delta_0 = nn.utils.parameters_to_vector(temp_model.parameters()).detach()
            delta_0 = nn.utils.parameters_to_vector(temp_model.parameters())
            if self.args.beta == 0.0:
                delta_0 = delta_0.detach()
            if self.args.rounds == 1 and self.args.num_users == 1:
                delta_0 = torch.zeros_like(delta_0)
            delta_0_list.append(delta_0)
            delta_list.append(delta_0.clone())
        # dim_model = delta_0.size()[0]

        # mean_diff = torch.zeros(dim_model).to(self.device)
        # info = torch.zeros(1)
        for e in range(self.args.local_ep):
            # torch.manual_seed(e)
            # torch.cuda.manual_seed(e)

            batch_loss = 0
            tau = 0
            for batch_idx, (images, labels) in enumerate(self.trainloaders[global_round]):
                # TRAINING ===========================================
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels, self.regul)
                loss.backward(retain_graph=True)
                optimizer.step()
                
                tau += len(labels)
                # ====================================================
                # g = PVector.from_model_grad(model).detach().get_flat_representation()
                g = PVector.from_model_grad(model).get_flat_representation()
                if self.args.beta == 0.0:
                    g = g.detach()

                # mean_diff -= self.args.lr*g
                for l in range(L):
                    delta_list[l] = delta_list[l] - self.args.lr*(torch.mul(g, torch.dot(g, delta_list[l]).item()))
                # ====================================================
                if (self.args.verbose == 3) and (batch_idx % 10 == 0):
                    print('Client {}| Round: {} | Local Epoch: {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.client_idx, global_round+1, e+1, batch_idx * len(images),
                        len(self.trainloaders[global_round].dataset),
                        100. * batch_idx / len(self.trainloaders[global_round]), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss += loss.item()
            batch_loss /= len(self.trainloaders[global_round])
            epoch_loss.append(batch_loss)

            if self.args.verbose >= 2: # Print every epoch
                print('Client {}| Round: {} | Local Epoch: {} |\tLoss: {:.6f}'.format(
                    self.client_idx, global_round+1, e+1, batch_loss))
                
            # ====================================================
            # POST-epoch MI calculation/Regularizer update 
            if (e+1) % self.info_eval_freq == 0 or self.args.beta > 0:
                self.compute_CMI(model, global_round, init, delta_list, delta_0_list, tau, L)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), self.info_estim
    

    def compute_CMI(self, model, round, init, delta_list, delta_0_list, tau, L):
        # mean_diff = nn.utils.parameters_to_vector(model.parameters()).detach() - init
        mean_diff = nn.utils.parameters_to_vector(model.parameters()) - init
        if self.args.beta == 0.0:
            mean_diff = mean_diff.detach()

        info = torch.zeros(1).to(self.device) #Reset
        a, b, c = [0]*L, [0]*L, [0]*L

        for _, (images, labels) in enumerate(self.trainloaders[round]):  
            images, labels = images.to(self.device), labels.to(self.device)

            model.zero_grad()
            pred = model.forward(images)
            loss = self.criterion(pred, labels, self.regul)
            loss.backward(retain_graph=True)

            # g = PVector.from_model_grad(model).detach().get_flat_representation()
            g = PVector.from_model_grad(model).get_flat_representation()
            if self.args.beta == 0.0:
                g = g.detach()
            # grad_avg += g
            for l in range(L):
                v = mean_diff + delta_list[l] - delta_0_list[l] # \tilde{m}^{(l)}
                prod = torch.dot(v, g)
                a[l] += torch.mul(prod, prod)
                b[l] += torch.mul(prod, torch.dot(delta_list[l], g))
                c[l] += prod

        # info = torch.dot(mean_diff, grad_avg).item()**2
        for l in range(L):
            info += (a[l]*tau + 2*b[l]*c[l])
        info /= (L*len(self.trainloaders[round]))

        self.info_estim = info.item()
        
        if self.args.beta > 0:
            self.regul = info.clone()

        # for _, (images, labels) in enumerate(self.trainloaders[global_round]):  
        #     images, labels = images.to(self.device), labels.to(self.device)

        #     model.zero_grad()
        #     pred = model.forward(images)
        #     loss = self.criterion(pred, labels)
        #     loss.backward(retain_graph=True)

        #     g = PVector.from_model_grad(model).detach().get_flat_representation()
            
        #     for l in range(L):
        #         v = mean_diff + delta_list[l] - delta_0_list[l] # \tilde{m}^{(l)}
        #         c = torch.dot(g, v).item()
        #         info += (a[l]*tau + 2*b[l]*c)
        # info /= (L*len(self.trainloader
        
        # print(self.info_estim)


    def inference(self, model, round=None):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        dataloader = self.validloader if round is None else self.trainloaders[round]
        for _, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            pred_labels = self.pred_fn(outputs).view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct / total
        loss /= len(dataloader) # OK
        
        return accuracy, loss


def test_inference(args, model, test_dataset, idxs=None):
    """ Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = get_criterion(args)
    pred_fn = get_pred_fn(args)
    if idxs is not None:
        dataset = DatasetSplit(test_dataset, idxs)
    else:
        dataset = test_dataset
    testloader = DataLoader(dataset, batch_size=32,
                            shuffle=False)
                            

    for _, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        
        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        pred_labels = pred_fn(outputs).view(-1)

        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy = correct / total
    loss /= len(testloader)

    return accuracy, loss
