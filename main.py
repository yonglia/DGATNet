from copyreg import pickle
import os
import numpy as np
import argparse
import time
import copy
import pickle
import json
import random

import matplotlib.pyplot as plt
import deepdish as dd

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split

from AdniData import AdniDataset
from torch_geometric.data import DataLoader
from net.brain_networks import DGATNet

from utils.utils import normal_transform_train,normal_transform_test,train_val_test_split, train_val_test_split_4
from utils.mmd_loss import MMD_loss

torch.manual_seed(123)

EPS = 1e-15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=30000, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/AD_NC_130_LXX', help='root directory of the dataset')
parser.add_argument('--matroot', type=str, default='MAT/ADNI/twoClassification/AD_NC_130_LXX_label.mat', help='root directory of the subject ID')
parser.add_argument('--fold', type=int, default=1, help='training which fold')
parser.add_argument('--lr', type = float, default=0.0001, help='learning rate')
parser.add_argument('--rep', type=int, default=1, help='augmentation times')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
parser.add_argument('--poolmethod', type=str, default='topk', help='topk || sag')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--nodes', type=int, default=200, help='number of nodes')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--net', type=str, default='DGATNet', help='model name:DGATNet')
parser.add_argument('--indim', type=int, default=200, help='feature dim')
parser.add_argument('--nclass', type=int, default=2, help='classification number') # 2分类 or 4分类
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--normalization', action='store_true') # 不用标准化
parser.add_argument('--slide_win_num', type=int, default=2, help='滑动窗口数')
parser.set_defaults(save_model=True)
parser.set_defaults(normalization=True)
opt = parser.parse_args()

#################### Parameter Initialization #######################
name = 'Adni'
str_time = time.strftime("%y_%m_%d_%H_%M_%S", time.localtime())
writer = SummaryWriter(os.path.join('./log/log_{}/{}_fold{}'.format(str_time, opt.net,opt.fold)))

os.mkdir('models/{}'.format(str_time))

dataset = AdniDataset(opt.dataroot, name)
data_num = len(dataset)

np.random.seed(0)
idx = np.random.permutation(data_num)

dataset = dataset[list(idx)]

############### split train, val, and test set -- need costumize########################
data_index = np.random.randn(len(dataset))
tr_index,te_index,val_index = train_val_test_split(mat_dir=opt.matroot, fold=opt.fold, rep = opt.rep)

train_mask = torch.zeros(len(dataset)*opt.rep, dtype=torch.bool)
val_mask = torch.zeros(len(dataset)*opt.rep, dtype=torch.bool)
test_mask = torch.zeros(len(dataset)*opt.rep, dtype=torch.bool)

tr_index_tensor = torch.tensor(np.floor(tr_index/opt.rep), dtype=torch.long)
val_index_tensor = torch.tensor(np.floor(val_index/opt.rep), dtype=torch.long)
te_index_tensor = torch.tensor(np.floor(te_index/opt.rep), dtype=torch.long)

train_dataset = dataset[tr_index_tensor]
val_dataset = dataset[val_index_tensor]
test_dataset = dataset[te_index_tensor]

# 标准化数据(已经在数据预处理阶段对roi time series数据标准化了)
if opt.normalization:
    for i in range(train_dataset.data.x.shape[1]):
        train_dataset.data.x[:, i], lamb, xmean, xstd = normal_transform_train(train_dataset.data.x[:, i])
        test_dataset.data.x[:, i] = normal_transform_test(test_dataset.data.x[:, i],lamb, xmean, xstd)
        val_dataset.data.x[:, i] = normal_transform_test(val_dataset.data.x[:, i], lamb, xmean, xstd)

test_loader = DataLoader(test_dataset,batch_size=opt.batchSize,shuffle = False)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)

############### Define Graph Deep Learning Network ##########################

model = DGATNet(opt.ratio, indim=opt.indim, poolmethod = opt.poolmethod).to(device)

print(model)
print('ground_truth: ', test_dataset.data.y, 'total: ', len(test_dataset.data.y), 'positive: ',sum(test_dataset.data.y.bool()))

optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

###################### Network Training Function#####################################
def train(epoch):
    print('train...........')
    model.train()
    
    loss_all = 0
    
    i = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, local1, local2 = model(data.x, data.edge_index, data.batch, data.edge_attr, opt.slide_win_num)
        loss = F.nll_loss(output, data.y) # classification loss
        # loss1 = F.nll_loss(local1, data.y)
        # loss2 = F.nll_loss(local2, data.y)
        
        #consistence_loss = torch.sum(torch.norm((output - local1), dim=1))+ torch.sum(torch.norm((output-local2), dim=1)) + torch.sum(torch.norm((local1-local2), dim=1))

        writer.add_scalar('train/classification_loss', loss, epoch * len(train_loader) + i)

        # loss_consist = loss + loss1 + loss2 
        loss_consist = loss
        loss_consist.backward()
        loss_all += loss_consist.item() * data.num_graphs
        optimizer.step()
        # scheduler.step()
        
        i = i + 1
        if not os.path.exists('outputs/'):
            os.makedirs('outputs/')

    return loss_all / len(train_dataset)

###################### Network Testing Function#####################################
def test_acc(loader):
    model.eval()
    correct = 0
    TP, FN, FP, TN =0, 0, 0, 0
    result = torch.zeros((2,2), dtype=torch.float32 ,device='cuda:0')
    for data in loader:
        data = data.to(device)
        
        output, _, _ = model(data.x, data.edge_index, data.batch, data.edge_attr, opt.slide_win_num)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        batch_size = len(output)
        batch_pred = torch.zeros((batch_size, 2, 2), device='cuda:0')
        for i in range(batch_size):
            batch_pred[i,pred[i], data.y[i]] = 1
        result += torch.sum(batch_pred, dim=0)
    return correct / len(loader.dataset), (result[1,1]/(result[1,0]+result[1,1])).item(), (result[1,1]/(result[0,1]+result[1,1])).item()

def test_loss(loader,epoch):
    print('testing...........')
    model.eval()
    loss_all = 0

    i=0
    for data in loader:
        data = data.to(device)
        output, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, opt.slide_win_num)
        loss_c = F.nll_loss(output, data.y)
        loss = loss_c
        writer.add_scalar('val/classification_loss', loss_c, epoch * len(loader) + i)
        i = i + 1

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

#######################################################################################
############################   Model Training #########################################
#######################################################################################
# 载入以前的模型
#raw_state_dict = torch.load('models/22_03_18_16_05_39/rep1_biopoint_1_DGATNet_0.pth')
# model.load_state_dict(torch.load('models/22_03_17_23_43_02/rep1_biopoint_1_DGATNet_0.pth'))

best_model_wts = copy.deepcopy(model.state_dict())
# best_model_wts.update(raw_state_dict)
# model.load_state_dict(best_model_wts)
best_loss = 1e10
for epoch in range(0, opt.n_epochs):
    
    since  = time.time()
    tr_loss = train(epoch)
    tr_acc, tr_prec, tr_rec = test_acc(train_loader)
    
    val_loss = test_loss(val_loader,epoch)
    val_acc, val_prec, val_rec = test_acc(val_loader)
    time_elapsed = time.time() - since
    
    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Train Precision: {:.7f}, Train Recall: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}, Test Precision: {:.7f}, Test Recall: {:.7f}'.format(epoch, tr_loss,
                                                       tr_acc, tr_prec, tr_rec, val_loss, val_acc, val_prec, val_rec), )

    writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
    writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
    # writer.add_scalar('Precision', {'train_precision': tr_prec, 'val_precision':val_prec}, epoch)
    # writer.add_scalar('Recall', {'train_recall': tr_rec, 'val_recall':val_rec}, epoch)

    if val_loss < best_loss and epoch > 5:
        print("saving best model")
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        if not os.path.exists('models/'):
            os.makedirs('models/')
        if opt.save_model:
            torch.save(best_model_wts,
                       'models/{}/rep{}_biopoint_{}_{}.pth'.format(str_time, opt.rep, opt.fold, opt.net))

#######################################################################################
######################### Testing on testing set ######################################
#######################################################################################
model.load_state_dict(best_model_wts)
model.eval()
test_accuracy = test_acc(test_loader)
test_l= test_loss(test_loader,0)
print("===========================")
print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
print(opt)
