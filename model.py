import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import matplotlib.pyplot as plt
import dgl.function as fn
import dgl
import sympy
import scipy
import numpy as np
from torch.nn import init
from torch.nn import Sequential
from scipy import sparse
import random

class RHO_local(Module):
    def __init__(self, layer, in_features):
        super(RHO_local, self).__init__()
        self.in_features = in_features
        self.layer = layer
        self.temp_local = nn.ParameterList()
        self.fc = nn.ModuleList()
        for i in range(self.layer):
            self.temp_local.append(Parameter(torch.FloatTensor(in_features)))
            self.fc.append(nn.Linear(in_features, in_features))
 
        self.reset_parameters()

    def reset_parameters(self):
        for temp in self.temp_local:
            torch.nn.init.normal_(temp, mean=0, std=0)


    def forward(self, Lap, X):
        
        for i in range(self.layer):
            LX = torch.spmm(Lap, X) #LX
            k = self.temp_local[i] #k
            X = torch.sub(X,k*LX) # (I-k_iL)X_i
            X = F.relu(self.fc[i](X))
        return X



class RHO_global(Module):

    def __init__(self, layer, in_features):
        super(RHO_global, self).__init__()
        self.in_features = in_features
        self.layer = layer
        self.fc = nn.ModuleList()
        for i in range(self.layer):
            self.fc.append(nn.Linear(in_features, in_features))

        self.temp_global = Parameter(torch.FloatTensor(self.layer))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp_global.data.fill_(1)

    def forward(self, Lap, X):
        
        for i in range(self.layer):
            LX = torch.spmm(Lap, X) #LX
            k = self.temp_global[i] #k
            X = torch.sub(X,k*LX) # (I-k_iL)X
            X = F.relu(self.fc[i](X))
            
        return X



class MLPEncoder(nn.Module):
    def __init__(self, in_features, hidden1, hidden2):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden1)  # 第一层：in_features -> hidden1
        self.fc2 = nn.Linear(hidden1, hidden2)      # 第二层：hidden1 -> hidden2
        self.activation = nn.ReLU()                 # 激活函数
        

    def forward(self, x):
        x = self.activation(self.fc1(x))  # 第一层 + 激活
        x = self.fc2(x)  # 第二层 + 激活
        return x

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

class RHO(nn.Module):
    def __init__(self, in_features, hidden1, hid2, layers, batch_size, tau):
        super(RHO, self).__init__()

        self.rep_dim = hid2
        self.batch_size = batch_size
        self.tau = tau
        self.activation = nn.ReLU()     

        self.encoder = MLPEncoder(in_features, hidden1, hid2)  # 编码器

        self.ada_global = RHO_global(layers, hid2)
        self.ada_local = RHO_local(layers,hid2)

        #self.linear = nn.Linear(hid2*2, hid2)
        self.proj_head1 = Sequential(nn.Linear(hid2, hid2))
        self.proj_head2 = Sequential(nn.Linear(hid2, hid2))


    def forward(self, Lap, x):
        x = self.encoder(x)
        x = self.activation(x)
        x_global = x
        x_local = x
    
        x_global = self.ada_global(Lap,x_global)
        x_local = self.ada_local(Lap,x_local)


        embedding_g_p, embedding_l_p  = self.proj_head1(x_global), self.proj_head2(x_local) 
        loss = self.batch_nce_loss(embedding_g_p, embedding_l_p, self.tau)


        return x_global, x_local, loss
    


    def batch_nce_loss(self, z1, z2, temperature=0.2, pos_mask=None, neg_mask=None):
        #if pos_mask is None and neg_mask is None:
        #    pos_mask = self.pos_mask
        #    neg_mask = self.neg_mask
        

        nnodes = z1.shape[0]
        if (self.batch_size == 0) or (self.batch_size > nnodes):
            pos_mask = torch.eye(z1.shape[0])
            neg_mask = 1 - pos_mask
            loss_0 = self.infonce(z1, z2, pos_mask, neg_mask, temperature)
            loss_1 = self.infonce(z2, z1, pos_mask, neg_mask, temperature)
            loss = (loss_0 + loss_1) / 2.0
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.batch_size)
            loss = 0.0
            for b in batches:
                b = torch.tensor(b, device=z1.device)
                #print('11')
                batch_size = len(b)
                pos_mask = torch.eye(batch_size, device=z1.device)
                neg_mask = 1 - pos_mask
                weight = len(b) / nnodes
                loss_0 = self.infonce(z1[b], z2[b], pos_mask, neg_mask, temperature)
                loss_1 = self.infonce(z2[b], z1[b], pos_mask, neg_mask, temperature)
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss


    def infonce(self, anchor, sample, pos_mask, neg_mask, tau):
        pos_mask = pos_mask.to(anchor.device)
        neg_mask = neg_mask.to(anchor.device)
        sim_anchor = self.similarity(anchor, anchor) / tau
        exp_sim_anchor = torch.exp(sim_anchor) * neg_mask
        sim = self.similarity(anchor, sample) / tau
        exp_sim = torch.exp(sim) * neg_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True)) - torch.log(exp_sim_anchor.sum(dim=1, keepdim=True))
        #log_prob = sim - torch.log(exp_sim_anchor.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()


    def similarity(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()
