from distutils.command.check import HAS_DOCUTILS
from platform import node
import torch
from torch import jit
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv,NNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_sort_pool as gsp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm

from net.MyTopK import TopKPooling
from net.MySAG import SAGPooling

import numpy as np



########################### PRGNN MICCAI Li et al. 2020  ###################################################
class NNGAT_Net(torch.nn.Module):
    def __init__(self, ratio, indim, poolmethod = 'topk'):
        super(NNGAT_Net, self).__init__()

        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 8
        self.indim = indim
        self.poolmethod = poolmethod

        self.conv1 = GATConv( self.indim, self.dim1)
        self.bn1 = torch.nn.BatchNorm1d(self.dim1)
        if self.poolmethod == 'topk':
            self.pool1 = TopKPooling(self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        elif self.poolmethod == 'sag':
            self.pool1 = SAGPooling(self.dim1, ratio=ratio, GNN=GATConv,nonlinearity=torch.sigmoid) #0.4 data1 10 fold

        self.conv2 = GATConv(self.dim1, self.dim2)
        self.bn2 = torch.nn.BatchNorm1d(self.dim2)
        if self.poolmethod == 'topk':
            self.pool2 = TopKPooling(self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        elif self.poolmethod == 'sag':
            self.pool2 = SAGPooling(self.dim2, ratio=ratio, GNN=GATConv,nonlinearity=torch.sigmoid)

        self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2)
        self.bn4 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn5 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, 2)

    def forward(self, x, edge_index, batch, edge_attr):
        # edge_attr = edge_attr.squeeze()
        # edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        x = self.conv1(x, edge_index)
        if x.norm(p=2, dim=-1).min() == 0:
            print('x is zeros')
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        x = self.conv2(x, edge_index)
        x, edge_index, edge_attr, batch, perm, score2  = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1,x2], dim=1) #concate

        x = self.bn4(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn5(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x, score1, score2

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        return edge_index, edge_weight


###########################  MICCAI Li et al. 2019  ###################################################
class LI_Net(torch.nn.Module):
    def __init__(self, indim):
        super(LI_Net, self).__init__()

        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 8
        self.indim = indim
        n1 = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, self.dim1 * self.indim))
        self.conv1 = NNConv(self.indim, self.dim1, n1, aggr="add")
        self.bn1 = torch.nn.BatchNorm1d(self.dim1)
        self.pool1 = TopKPooling(self.dim1, ratio=0.5)
        n2 = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, self.dim1 * self.dim2))
        self.conv2 = NNConv(self.dim1, self.dim2, n2, aggr="add")
        self.bn2 = torch.nn.BatchNorm1d(self.dim2)
        self.pool2 = TopKPooling(self.dim2, ratio=0.5)

        self.fc1 = torch.nn.Linear((self.dim1 + self.dim2) * 2, self.dim2)
        self.bn4 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn5 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, 2)

    def forward(self, x, edge_index, batch, edge_attr):
        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        x = self.conv1(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        x = self.conv2(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1, x2], dim=1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x,score1, score2

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        return edge_index, edge_weight

class DGATNet(nn.Module):
    def __init__(self, ratio, indim, poolmethod = 'topk'):
        super(DGATNet, self).__init__()
        self.dim1 = 64
        self.dim2 = 32
        self.dim3 = 8
        self.indim = indim
        self.poolmethod = poolmethod

        # 编码过程
        self.conv1 = GATConv( self.indim, self.dim1)
        self.bn1 = torch.nn.BatchNorm1d(self.dim1)
        if self.poolmethod == 'topk':
            self.pool1 = TopKPooling(self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        elif self.poolmethod == 'sag':
            self.pool1 = SAGPooling(self.dim1, ratio=ratio, GNN=GATConv,nonlinearity=torch.sigmoid) #0.4 data1 10 fold

        self.conv2 = GATConv(self.dim1, self.dim2)
        self.bn2 = torch.nn.BatchNorm1d(self.dim2)
        if self.poolmethod == 'topk':
            self.pool2 = TopKPooling(self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        elif self.poolmethod == 'sag':
            self.pool2 = SAGPooling(self.dim2, ratio=ratio, GNN=GATConv,nonlinearity=torch.sigmoid)
            
        self.conv3 = GATConv(self.dim2, self.dim2) # 特征维数不改变
        
        # 添加注意力模块
        self.vtam = VTAM(self.dim2, 1, 1)
        
        # 解码过程
        self.unpool1 = DiffusioUnpool(self.dim2)
        self.deconv1 = GATConv(self.dim2, self.dim1)
        self.vtam1 = VTAM(self.dim1, 1, 1)
        
        self.unpool2 = DiffusioUnpool(self.dim2)
        self.deconv2 = GATConv(self.dim1, self.indim)
        self.vtam2 = VTAM(self.indim, 1, 1)

        # Readout
        self.fc1 = torch.nn.Linear(self.indim, self.dim1)
        self.bn4 = torch.nn.BatchNorm1d(self.dim1)
        self.fc2 = torch.nn.Linear(self.dim1, self.dim3)
        self.bn5 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, 2) # 2分类or 4分类
        
    def forward(self, x, edge_index, batch, edge_attr, win_num):
        # edge_attr = edge_attr.squeeze()
        # edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        y = self.conv1(x, edge_index)
        if y.norm(p=2, dim=-1).min() == 0:
            print('x is zeros')
        real_batch = batch
        real_batch_size = torch.max(real_batch).item() + 1
        batch = generator_batch(batch, win_num)
        x1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(y, edge_index, edge_attr, batch)

        edge_attr1 = edge_attr1.squeeze()
        #edge_index1, edge_attr1 = self.augment_adj(edge_index1, edge_attr1, x1.size(0))

        y1 = self.conv2(x1, edge_index1)
        x2, edge_index2, edge_attr2, batch2, perm2, score2  = self.pool2(y1, edge_index1, edge_attr1, batch1)

        y2 = self.conv3(x2, edge_index2)
        xx2, h_out, time_coff, node_coff = self.vtam(y2, batch2, win_num) # 注意batch
        batch_size = torch.max(batch2).item() + 1
        
        # 下面是注意力解码器模块
        xx2 = xx2.squeeze()
        # Unpooling 参数应该有perm, batch, 输入维数, 还有要恢复的节点数目个数, 
        node_num, _ = y1.shape
        yy1, node_coff = self.unpool1(xx2, perm2, edge_index1, edge_attr1, node_coff, node_num)
        temp = torch.matmul(y1.new_ones(y1.shape[1], 1), node_coff.t()) + y1.new_ones((y1.t().shape)) 
        temp = torch.mul(y1, temp.t()).reshape(batch_size, -1, self.dim2)
        yy1_res = yy1 + torch.mul(time_coff[0].reshape(batch_size, 1, 1), temp).reshape(-1, self.dim2)# 融合y1(编码器)和yy1(解码器)的信息
        xx1 = self.deconv1(yy1_res, edge_index1)
        xx1, h_out2, time_coff1, node_coff1 = self.vtam1(xx1, batch1, win_num)
        
        xx1 = xx1.squeeze()
        node_num , _= y.shape
        yy, node_coff1 = self.unpool2(xx1, perm1, edge_index1, edge_attr1, node_coff1, node_num)
        temp = torch.matmul(y.new_ones(y.shape[1], 1), node_coff1.t()) + y.new_ones((y.t().shape))
        temp = torch.mul(y, temp.t()).reshape(batch_size, -1, self.dim1)
        yy_res = yy + torch.mul(time_coff1[0].reshape(batch_size, 1, 1), temp).reshape(-1, self.dim1)
        xx = self.deconv2(yy_res, edge_index)
        xx, h_out3, time_coff2, node_coff2 = self.vtam2(xx, batch, win_num)

        if batch_size == 1:
            x = F.relu(self.fc1(h_out3))
        else:

            x = self.bn4(F.relu(self.fc1(h_out3)))
        x = F.dropout(x, p=0.5, training=self.training)
        
        if batch_size == 1:
            x = F.relu(self.fc2(x))
        else:
            x = self.bn5(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1) 
        
        return x, score1, score2
    
    def augment_adj(self, edge_index, edge_weight, num_nodes):
        if not isinstance(num_nodes, int):
            num_nodes = num_nodes.item()
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                    num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index, edge_weight, num_nodes, num_nodes, num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        return edge_index, edge_weight
        
class VTAM(nn.Module):
    def __init__(self, in_dim, out_dim, time_num=1):
        super(VTAM, self).__init__()
        self.time_num = time_num
        self.proj = nn.Linear(in_dim, out_dim).cuda()

    def forward(self, g_feature_time, batch, win_num):
        tol_len, fea_dim = g_feature_time.shape
        node_num = int(tol_len/(torch.max(batch).item() + 1))
        g_feature_time = g_feature_time.reshape(-1, win_num, node_num, fea_dim)
        # g_feature_array = torch.sum(g_feature_time, dim=1) # 按时间维求和
        node_atten_coffe_time = self.proj(g_feature_time)  # 节点注意力系数
        batch_size, _, _, _ = node_atten_coffe_time.shape
        node_atten_coffe = torch.sum(node_atten_coffe_time, dim=1) # 按时间维度求和
        
        temp_ones = node_atten_coffe.new_ones((node_num, 1))
        time_atten_coffe = torch.matmul(node_atten_coffe_time.reshape(-1, win_num, 1, node_num), temp_ones).squeeze() # 时间注意力系数
        
        other = torch.matmul(node_atten_coffe.new_ones((batch_size, win_num, 1)), (node_atten_coffe @ node_atten_coffe.new_ones((1, fea_dim))).reshape(-1, 1, node_num * fea_dim)).reshape(-1, win_num, node_num, fea_dim)
        
        g_feature_update = torch.mul(g_feature_time, other)

        # 全局最大池化操作
        h_output, _ = torch.max(g_feature_update, 1)
        h_output, _ = torch.max(h_output, 1)

        return (
            g_feature_update.reshape(-1, fea_dim),
            h_output.reshape(-1, fea_dim),
            time_atten_coffe.reshape(-1),
            node_atten_coffe.reshape(-1),
        )  # 注意力系数也要返回，需要传递给解码器层
        
class DiffusioUnpool(nn.Module):
    def __init__(self, indim):
        super(DiffusioUnpool, self).__init__()
        self.indim = indim
        
    def forward(self, fea, perm, encoder_edge_index, encoder_edge_attr, node_atte_coffe, node_num):
        _, node_dim = fea.shape
        x_zero = fea.new_zeros(node_num, node_dim)
        atte_coffe_zero = node_atte_coffe.new_zeros(node_num,1)
        x_zero[perm] = fea
        atte_coffe_zero[perm] = node_atte_coffe
        value = torch.ones(node_num).cuda()
        index = torch.tensor([range(node_num), range(node_num)]).cuda()
        A_sparse = torch.sparse_coo_tensor(encoder_edge_index, encoder_edge_attr, size=(node_num, node_num)) + torch.sparse_coo_tensor(index, value, (node_num, node_num))
        A = A_sparse.to_dense()
        D = torch.diag(1/torch.sqrt(torch.sum(A, dim=1)))
        x = torch.matmul(torch.matmul(torch.matmul(D, A), D), x_zero)
        
        atte_updata = torch.matmul(torch.matmul(torch.matmul(D, A), D), atte_coffe_zero)
        
        return x, atte_updata
    
def generator_batch(batch, win_num):
    batch_size = int((torch.max(batch)+1)*win_num)
    node_num = int(len(batch)/batch_size)
    col = torch.range(0,batch_size-1, dtype=torch.int64).reshape(batch_size, 1)
    row = torch.ones(1, node_num, dtype=torch.int64)
    batch = torch.matmul(col, row).reshape(-1).to('cuda')
    
    return batch