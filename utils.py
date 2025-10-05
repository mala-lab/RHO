import time
import math
import random
import numpy as np
import scipy as sp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from collections import Counter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False


def get_split(num_node, label, train_rate=0.3, val_rate =0.1):

    all_labels = np.squeeze(np.array(label))
    #num_node = index
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    print('Training', Counter(np.squeeze(all_labels[idx_train])))
    print('Test', Counter(np.squeeze(all_labels[idx_test])))
    # 分离正常节点和异常节点
    #normal_idx = [i for i in index if all_labels[i] == 0]
    #anomal_idx = [i for i in index if all_labels[i] == 1]

    all_normal_label_idx = [i for i in idx_train if all_labels[i] == 0]
    rate = 0.5  #  change train_rate to 0.3 0.5 0.6  0.8
    normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * rate)]
    print('Training rate', rate*train_rate)

    
    return normal_label_idx, idx_val, idx_test
