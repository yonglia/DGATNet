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

from AdniData import AdniDataset
from torch_geometric.data import DataLoader
from net.brain_networks import DGATNet, LI_Net,NNGAT_Net

from utils.utils import normal_transform_train,normal_transform_test,train_val_test_split, train_val_test_split_4
from utils.mmd_loss import MMD_loss

torch.manual_seed(123)

EPS = 1e-15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=30000, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/ADNI2Classification', help='root directory of the dataset')
parser.add_argument('--matroot', type=str, default='MAT/ADNI/twoClassification/twoclassification_label.mat', help='root directory of the subject ID')
parser.add_argument('--fold', type=int, default=1, help='training which fold')
parser.add_argument('--lr', type = float, default=0.0001, help='learning rate')
parser.add_argument('--rep', type=int, default=1, help='augmentation times')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=1, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=1, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0.1, help='s1 distance regularization')
parser.add_argument('--lamb4', type=float, default=0.1, help='s2 distance regularization')
parser.add_argument('--lamb5', type=float, default=0, help='s1 consistence regularization')
parser.add_argument('--lamb6', type=float, default=0, help='s2 consistence regularization')
parser.add_argument('--distL', type=str, default='bce', help='bce || mmd')
parser.add_argument('--poolmethod', type=str, default='topk', help='topk || sag')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam || SGD')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--nodes', type=int, default=116, help='number of nodes')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--net', type=str, default='DGATNet', help='model name: NNGAT || LI_NET || DGATNet')
parser.add_argument('--indim', type=int, default=70, help='feature dim')
parser.add_argument('--nclass', type=int, default=2, help='feature dim') # 2分类 or 4分类
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--normalization', action='store_true')
parser.set_defaults(save_model=True)
parser.set_defaults(normalization=True)
opt = parser.parse_args()

#################### Parameter Initialization #######################
name = 'Adni'
writer = SummaryWriter(os.path.join('./log/{}_fold{}_consis{}'.format(opt.net,opt.fold,opt.lamb5)))

############# Define Dataloader -- need costumize#####################
dataset = AdniDataset(opt.dataroot, name)

############### split train, val, and test set -- need costumize########################
data_index = np.random.randn(len(dataset))
tr_index,te_index,val_index = train_val_test_split(mat_dir=opt.matroot,fold=opt.fold,rep = opt.rep)

########### skip these two lines for cv.. ############################
# val_index = np.concatenate([te_index,val_index])
# te_index = val_index
######################################################################

test_dataset = dataset[te_index.tolist()]
train_dataset = dataset[tr_index.tolist()]
val_dataset = dataset[val_index.tolist()]

# ######################## Data Preprocessing ########################
# ###################### Normalize features ##########################
if opt.normalization:
    for i in range(train_dataset.data.x.shape[1]):
        train_dataset.data.x[:, i], lamb, xmean, xstd = normal_transform_train(train_dataset.data.x[:, i])
        test_dataset.data.x[:, i] = normal_transform_test(test_dataset.data.x[:, i],lamb, xmean, xstd)
        val_dataset.data.x[:, i] = normal_transform_test(val_dataset.data.x[:, i], lamb, xmean, xstd)

test_loader = DataLoader(test_dataset,batch_size=opt.batchSize,shuffle = False)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)

############### Define Graph Deep Learning Network ##########################
if opt.net =='LI_NET':
    model = LI_Net(opt.ratio).to(device)
elif opt.net == 'NNGAT':
    model = NNGAT_Net(opt.ratio, indim=opt.indim, poolmethod = opt.poolmethod).to(device)
elif opt.net == 'DGATNet':
    model = DGATNet(opt.ratio, indim=opt.indim, poolmethod = opt.poolmethod).to(device)

print(model)
print('ground_truth: ', test_dataset.data.y, 'total: ', len(test_dataset.data.y), 'positive: ',sum(test_dataset.data.y.bool()))
if opt.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
elif opt.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

###################### Network Training Function#####################################
def train(epoch):
    print('train...........')
    model.train()

    loss_all = 0
    
    i = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, 18)

        loss = F.nll_loss(output, data.y) # classification loss

        writer.add_scalar('train/classification_loss', loss, epoch * len(train_loader) + i)

        i = i + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs

        optimizer.step()
        scheduler.step()

        if not os.path.exists('outputs/'):
            os.makedirs('outputs/')

    return loss_all / len(train_dataset)

###################### Network Testing Function#####################################
def test_acc(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output, _, _ = model(data.x, data.edge_index, data.batch, data.edge_attr)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def test_loss(loader,epoch):
    print('testing...........')
    model.eval()
    loss_all = 0

    i=0
    for data in loader:
        data = data.to(device)
        output, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss_c = F.nll_loss(output, data.y)

        loss = loss_c
        writer.add_scalar('val/classification_loss', loss_c, epoch * len(loader) + i)
        i = i + 1

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

#######################################################################################
############################   Model Training #########################################
#######################################################################################
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10
for epoch in range(0, opt.n_epochs):
    
    since  = time.time()
    tr_loss = train(epoch)
    tr_acc = test_acc(train_loader)
    val_acc = test_acc(val_loader)
    val_loss = test_loss(val_loader,epoch)
    time_elapsed = time.time() - since
    
    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                       tr_acc, val_loss, val_acc))

    writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
    writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)

    if val_loss < best_loss and epoch > 5:
        print("saving best model")
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        if not os.path.exists('models/'):
            os.makedirs('models/')
        if opt.save_model:
            torch.save(best_model_wts,
                       'models/rep{}_biopoint_{}_{}_{}.pth'.format(opt.rep,opt.fold,opt.net,opt.lamb5))

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
