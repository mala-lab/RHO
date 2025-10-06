import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

from model import RHO
from utils import count_parameters, init_params, get_split
from dataset import Dataset
import warnings
warnings.filterwarnings("ignore")

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main_worker(args):

    fix_seed(args.seed)
    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    time_run=[]    

    def init_center_c(adj, inputs, net, device, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c_global = torch.zeros(net.rep_dim).to(device)
        c_local = torch.zeros(net.rep_dim).to(device)
        net.eval()
        with torch.no_grad():
            outputs_global, outputs_local,  _ = net(adj, inputs)

            n_samples = outputs_global.shape[0]
            c_global =torch.sum(outputs_global, dim=0)
            c_local =torch.sum(outputs_local, dim=0)

        c_global /= n_samples
        c_local /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c_local[(abs(c_local) < eps) & (c_local < 0)] = -eps
        c_local[(abs(c_local) < eps) & (c_local > 0)] = eps

        c_global[(abs(c_global) < eps) & (c_global < 0)] = -eps
        c_global[(abs(c_global) < eps) & (c_global > 0)] = eps

        return c_local, c_global

    def train(model, optimizer, adj, idx_train, center_local, center_global):
        
        t_st=time.time()
        model.train()
        optimizer.zero_grad()
        outputs_global, outputs_local, nce_loss = model(adj, features)
        dist_global = torch.sum((outputs_global[idx_train] - center_global) ** 2, dim=1)
        dist_local = torch.sum((outputs_local[idx_train] - center_local) ** 2, dim=1) 
        dist = 0.5*dist_global + 0.5*dist_local

        loss =  torch.mean(dist) + args.alpha * nce_loss
        loss.backward()
        optimizer.step()
        time_epoch = time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)
        del outputs_global, outputs_local
        torch.cuda.empty_cache()
        return time_run, loss
    
    def test(model, adj, labels, idx, center_local, center_global):
        with torch.no_grad():
            model.eval()
            outputs_global, outputs_local, nce_loss = model(adj, features)

            scores = ((torch.sum((outputs_global[idx] - center_global) ** 2, dim=1))+\
                      (torch.sum((outputs_local[idx] - center_local) ** 2, dim=1)))/2
            
            labels = np.array(labels.cpu().data.numpy())
            scores = np.array(scores.cpu().data.numpy())

            precision, recall, _ = precision_recall_curve(labels[idx], scores)
            auprc = auc(recall, precision)
            #print(" Test set  AUPRC: {:.4f}".format(100. * auprc))
            # 计算 AUROC
            auroc = roc_auc_score(labels[idx], scores)
            #print('Test set AUROC: {:.2f}%'.format(100. * auroc))
            return auprc, auroc
        
    graph = Dataset(args.dataset).graph
    Lap = Dataset(args.dataset).Lap
    labels = graph.ndata['label']
    features = graph.ndata['feature']
    in_feats = graph.ndata['feature'].shape[1]

    num_node = features.shape[0]
    idx_train, idx_val, idx_test = get_split(num_node, labels, args.train_ratio)

    net = RHO(in_feats, args.hidden1, args.hidden2, args.nlayers, args.batch_size, args.tau).to(device) 
    #graph = graph.to(device)
    Lap = Lap.to(device)
    features = features.to(device) 
    labels = labels.to(device) 
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    


    for epoch in range(args.epochs):
        c_local, c_global = init_center_c(Lap, features, net, device)
        timerun,losses_train = train(net, optimizer, Lap, idx_train, c_local, c_global)
        
        auprc_test, auroc_test = test(net, Lap, labels, idx_test, c_local, c_global)
        print(f"{epoch:>4} epochs trained. "
        f"Current auprc: {round(auprc_test, 4):>7} "
        f"Current auroc:  {round(auroc_test, 4):>7} "
        f"Total loss: {round(float(losses_train), 4):>7} ")
            
            
        if epoch == args.epochs-1:
            print('AUROC:{}'.format(auroc_test))
            print('AUPRC:{}'.format(auprc_test))

    #torch.save(net.state_dict(), './checkpoint/{}_{}_{}_best'.format(args.dataset,auroc_test,auprc_test))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='elliptic',
                        choices=['amazon', 'tfinance','reddit','photo','elliptic','tolokers','questions'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument("--train_ratio", type=float, default=0.3, help="Training ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val ratio")
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden1', type=int, default=1024)
    parser.add_argument('--hidden2', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.2)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    main_worker(args)
    
