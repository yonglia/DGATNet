import os
import numpy as np
import argparse
import time
import copy

import matplotlib.pyplot as plt
import deepdish as dd

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split

from torch_geometric.data import DataLoader
from net.brain_networks import NNGAT_Net


torch.manual_seed(123)

EPS = 1e-15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=30000, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/AD_NC_137', help='root directory of the dataset')
parser.add_argument('--matroot', type=str, default='MAT/ADNI/twoClassification/AD_NC_70_label.mat', help='root directory of the subject ID')
parser.add_argument('--fold', type=int, default=1, help='training which fold')
parser.add_argument('--lr', type = float, default=0.0001, help='learning rate')
parser.add_argument('--rep', type=int, default=1, help='augmentation times')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
parser.add_argument('--poolmethod', type=str, default='topk', help='topk || sag')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam || SGD')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--nodes', type=int, default=116, help='number of nodes')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--net', type=str, default='NNGAT', help='model name: NNGAT || LI_NET || DGATNet')
parser.add_argument('--indim', type=int, default=137, help='feature dim')
parser.add_argument('--nclass', type=int, default=2, help='feature dim') # 2分类 or 4分类
parser.set_defaults(save_model=True)
parser.set_defaults(normalization=True)
opt = parser.parse_args()

#################### Parameter Initialization #######################

from torch_geometric.datasets import FakeDataset 

train_dataset = FakeDataset(153, 116, 15, 137, 1, 2, "graph", True)
val_dataset = FakeDataset(51, 116, 15, 137, 1, 2, "graph", True)
test_dataset = FakeDataset(51, 116, 15, 137, 1, 2, "graph", True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


############### Define Graph Deep Learning Network ##########################

model = NNGAT_Net(opt.ratio, indim=opt.indim, poolmethod = opt.poolmethod).to(device)

print(model)
print('ground_truth: ', test_dataset.data.y, 'total: ', len(test_dataset.data.y), 'positive: ',sum(test_dataset.data.y.bool()))

optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

###################### Network Training Function#####################################
def train(epoch):
    print('train...........')
    model.train()

    loss_all = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_weight, 18)
        loss = F.nll_loss(output, data.y) # classification loss
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        # scheduler.step()

    return loss_all / len(train_dataset)

def test_acc(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output, _, _ = model(data.x, data.edge_index, data.batch, data.edge_weight, 18)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def test_loss(loader,epoch):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_weight, 18)
        loss_c = F.nll_loss(output, data.y)
        loss = loss_c
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

for epoch in range(0, opt.n_epochs):
    
    since  = time.time()
    tr_loss = train(epoch)
    tr_acc = test_acc(train_loader)
    val_acc = test_acc(val_loader)
    val_loss = test_loss(val_loader,epoch)
    time_elapsed = time.time() - since
    
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss, tr_acc, val_loss, val_acc))
    
def load_model(path):
    model_dict=model.load_state_dict(torch.load(path))
    return model_dict