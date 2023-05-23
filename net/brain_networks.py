from distutils.command.check import HAS_DOCUTILS
from platform import node
import torch
from torch import jit
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv, TopKPooling, TransformerConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_sort_pool as gsp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm
from torch.nn import init

# from net.SelfDefinitionConv.use_conv import GATConv

#from net.MyTopK import TopKPooling
from net.MySAG import SAGPooling

import numpy as np


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
        # self.bn3 = torch.nn.BatchNorm1d(self.dim2)
        # self.fc_out_1 = torch.nn.Linear(2*self.dim2, self.dim3)
        # self.fc_out_2 = torch.nn.Linear(self.dim3, 2) # 2分类
        
        # # 解码过程
        self.unpool1 = DiffusioUnpool(self.dim2)
        self.deconv1 = GATConv(2*self.dim2, self.dim1)
        self.bn_d1 = torch.nn.BatchNorm1d(self.dim1)
        self.vtam1 = VTAM(self.dim1, 1, 1)
        self.bn_v1 = torch.nn.BatchNorm1d(self.dim1)
        # self.fc_out2_1 = torch.nn.Linear(self.dim1, self.dim2)
        # self.fc_out2_2 = torch.nn.Linear(self.dim2, self.dim3)
        # self.fc_out2_3 = torch.nn.Linear(self.dim3, 2) # 2分类
        
        self.unpool2 = DiffusioUnpool(self.dim2)
        self.deconv2 = GATConv(2*self.dim1, self.indim)
        self.bn_d2 = torch.nn.BatchNorm1d(self.indim)
        self.vtam2 = VTAM(self.indim, 1, 1)
        self.bn_v2 = torch.nn.BatchNorm1d(self.indim)

        # # Readout
        # #self.fc1 = torch.nn.Linear(self.indim, self.dim2)
        # self.fc1 = torch.nn.Linear(self.dim2 + self.dim1, self.dim2)
        self.fc1 = torch.nn.Linear(self.dim2 + self.dim1 + self.indim, self.dim2)
        self.bn4 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn5 = torch.nn.BatchNorm1d(self.dim3)
        
        self.fc3 = torch.nn.Linear(self.dim3, 2) # 2分类or 4分类
        
        # 初始化参数
        #self.weights_init()
        
    def forward(self, x, edge_index, batch, edge_attr, win_num):
        # with torch.no_grad():
        y = self.conv1(x, edge_index)
        # y = self.bn1(y)
        if y.norm(p=2, dim=-1).min() == 0:
            print('x is zeros')
        real_batch = batch
        real_batch_size = torch.max(real_batch).item() + 1
        batch = generator_batch(batch, win_num)
        x1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(y, edge_index, edge_attr, batch)
        edge_index1, edge_attr1 = self.augment_adj(edge_index1, edge_attr1, x.size(0))

        # 第二层编码器
        y1 = self.conv2(x1, edge_index1)
        # y1 = self.bn2(y1)
        x2, edge_index2, edge_attr2, batch2, perm2, score2  = self.pool2(y1, edge_index1, edge_attr1, batch1)
        
        # h_out = torch.cat([gmp(x2, batch2),gap(x2, batch2)], dim=1)
        # h_out = torch.sum(h_out.view(real_batch_size, win_num, -1), dim=1)

        y2 = self.conv3(x2, edge_index2)
        xx2, h_out, time_coff, node_coff = self.vtam(y2, batch2, win_num) # 注意batch
        # # xx2 = self.bn3(xx2)
        # h_out = torch.cat([gmp(xx2,batch2), gap(xx2, batch2)], dim=1)
        # h_out = torch.sum(h_out.view(real_batch_size, win_num, -1), dim=1)
        
        # # 输出判别性结果
        # s1 = F.relu(self.fc_out_1(h_out))
        # s1 = F.log_softmax(self.fc_out_2(s1), dim=-1)
        
        
        # # 下面是注意力解码器模块
        # Unpooling 参数应该有perm, batch, 输入维数, 还有要恢复的节点数目个数, 
        node_num, fea_dim = y1.shape
        yy1, node_coff = self.unpool1(xx2, perm2, edge_index1, edge_attr1, node_coff, node_num, real_batch_size)
        
        # 利用注意力系数重新更新相应编码器层的节点信息
        y1_n = y1.reshape(real_batch_size, win_num, -1) * time_coff.reshape(real_batch_size, -1, 1)
        other = node_coff.reshape(real_batch_size, win_num, -1, 1) @ torch.ones((1, fea_dim), device='cuda') + torch.ones((int(len(node_coff)/(real_batch_size*win_num)), fea_dim), device='cuda')
        y1_res = (y1_n.reshape(real_batch_size, win_num, -1, fea_dim) * other).reshape(-1, fea_dim)

        #yy1_res = yy1 + y1_res # 融合y1(编码器)和yy1(解码器)的信息, 此处用的是加法(也可以维度拼接来融合)
        yy1_res = torch.concat([yy1, y1_res], dim=1) # 融合y1(编码器)和yy1(解码器)的信息, 此处是拼接二者
        xx1 = self.deconv1(yy1_res, edge_index1)
        xx1 = self.bn_d1(xx1)
        xx1, h_out2, time_coff1, node_coff1 = self.vtam1(xx1, batch1, win_num)
        # xx1 = self.bn_v1(xx1)
        
        # s2 = F.relu(self.fc_out2_1(h_out2))
        # s2 = F.relu(self.fc_out2_2(s2))
        # s2 = F.log_softmax(self.fc_out2_3(s2), dim=-1)
              
        
        # 解码器第二层
        node_num, fea_dim = y.shape
        yy, node_coff1 = self.unpool2(xx1, perm1, edge_index, edge_attr, node_coff1, node_num, real_batch_size)
        
        y_n = y.reshape(real_batch_size, win_num, -1) * time_coff1.reshape(real_batch_size, -1, 1)
        other = node_coff1.reshape(real_batch_size, win_num, -1, 1) @ torch.ones((1, fea_dim), device='cuda') + torch.ones((int(len(node_coff1)/(real_batch_size*win_num)), fea_dim), device='cuda')
        y_res = (y_n.reshape(real_batch_size, win_num, -1, fea_dim) * other).reshape(-1, fea_dim)
        
        #yy_res = yy + y_res
        yy_res = torch.concat([yy, y_res], dim=1)
        xx = self.deconv2(yy_res, edge_index)
        xx = self.bn_d2(xx)
        xx, h_out3, time_coff2, node_coff2 = self.vtam2(xx, batch, win_num)
        xx = self.bn_v2(xx)
        
        h_out4 = torch.concat([h_out, h_out2, h_out3], dim=1)

        if real_batch_size == 1:
            x = F.relu(self.fc1(h_out4))
        else:
            x = self.bn4(F.relu(self.fc1(h_out4)))
        x = F.dropout(x, p=0.5, training=self.training)
        
        if real_batch_size == 1:
            x = F.relu(self.fc2(x))
        else:
            x = self.bn5(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1) 
        
        return x, x, x # x是最后的特征输出, s1, s2是低分辨率下的特征
    
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
    
    def weights_init(self):
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight' and self.state_dict()[key].ndim > 1:
                init.kaiming_normal(self.state_dict()[key], mode='fan_out')
            elif key.split('.')[-1] == 'bias' and self.state_dict()[key].ndim > 1:
                self.state_dict()[key][...] = 0
        
class VTAM(nn.Module):
    def __init__(self, in_dim, out_dim, time_num=1):
        super(VTAM, self).__init__()
        self.time_num = time_num
        self.proj = nn.Linear(in_dim, out_dim).cuda()
        # self.bn = nn.BatchNorm1d(in_dim)

    def forward(self, g_feature_time, batch, win_num):
        tol_len, fea_dim = g_feature_time.shape
        node_num = int(tol_len/(torch.max(batch).item() + 1))
        g_feature_time = g_feature_time.reshape(-1, win_num, node_num, fea_dim)
        # g_feature_array = torch.sum(g_feature_time, dim=1) # 按时间维求和
        node_atten_coffe_time = self.proj(g_feature_time)  # 节点注意力系数
        batch_size, _, _, _ = node_atten_coffe_time.shape
        node_atten_coffe = torch.sum(node_atten_coffe_time, dim=1) # 按时间维度求和
        node_atten_coffe = torch.softmax(node_atten_coffe, dim=1) # 节点系数要归一下化处理
        
        temp_ones = node_atten_coffe.new_ones((node_num, 1))
        time_atten_coffe = torch.matmul(node_atten_coffe_time.reshape(-1, win_num, 1, node_num), temp_ones).reshape(batch_size, -1) # 时间注意力系数
        time_atten_coffe = torch.softmax(time_atten_coffe, dim=1) # 时间点系数也要归一化处理
        
        other = torch.matmul(node_atten_coffe.new_ones((batch_size, win_num, 1)), (node_atten_coffe @ node_atten_coffe.new_ones((1, fea_dim))).reshape(-1, 1, node_num * fea_dim)).reshape(-1, win_num, node_num, fea_dim)
        
        g_feature_update = (torch.mul(g_feature_time, other) * node_num + g_feature_time).reshape(batch_size, win_num, 1, -1)
        time_atten_coffe = time_atten_coffe.reshape(batch_size, win_num, 1, 1)
        g_feature_update = torch.matmul(time_atten_coffe, g_feature_update)
        
        
        # batch normalization操作
        x = g_feature_update.reshape(-1, node_num, fea_dim)
        # x_mean = x.mean(dim=1).reshape(-1, 1, fea_dim)
        # x_std = x.std(dim=1).reshape(-1, 1, fea_dim)
        # x = ((x-x_mean)/x_std).reshape(batch_size, -1, node_num, fea_dim)
        # # x = self.bn(x)
        x = x.reshape(batch_size, -1, node_num, fea_dim)

        # 全局最大池化操作
        h_output, _ = torch.max(x, 1)
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
        
    def forward(self, fea, perm, encoder_edge_index, encoder_edge_attr, node_atte_coffe, all_node_num, batch_size):
        # 需要知道batch_size, slide_win_num, node_num, fea_dim四个参数
        slide_win_num = int(len(perm)/len(node_atte_coffe))
        _, fea_dim = fea.shape
        node_num = int(len(node_atte_coffe)/batch_size)
        next_node_num = int(all_node_num/(batch_size * slide_win_num))
        
        # 重新复制一下同一个数据不同时间窗口的节点注意力系数
        node_atte_coffe = node_atte_coffe.reshape(batch_size, 1, node_num)
        win_node_atte_coffe = torch.matmul(torch.ones((batch_size, slide_win_num, 1), device='cuda'), node_atte_coffe).reshape(-1)
        
        # 上采样扩散：feature
        x_zero = fea.new_zeros(all_node_num, fea_dim)
        x_zero[perm] = fea
        
        # 上采样扩散: 节点注意力系数
        atte_coffe_zero = node_atte_coffe.new_zeros(all_node_num)
        atte_coffe_zero[perm] = win_node_atte_coffe
        
        value = torch.ones(all_node_num).cuda()
        index = torch.tensor([range(all_node_num), range(all_node_num)]).cuda()
        A_sparse = torch.sparse_coo_tensor(encoder_edge_index, encoder_edge_attr, size=(all_node_num, all_node_num)) + torch.sparse_coo_tensor(index, value, (all_node_num, all_node_num))
        A = A_sparse.to_dense()
        D = torch.diag(1/torch.sqrt(torch.sum(A, dim=1)))
        # D_T = torch.transpose(D.view(72, 58, 4176), 1, 2).view(72, 72, 58, 58).sum(dim=1) 
        x = torch.matmul(torch.matmul(torch.matmul(D, A), D), x_zero)
        
        atte_updata = torch.matmul(torch.matmul(torch.matmul(D, A), D), atte_coffe_zero)
        
        return x, atte_updata
    
def generator_batch(batch, win_num):
    batch_size = int((torch.max(batch)+1)*win_num)
    node_num = int(len(batch)/batch_size)
    col = torch.arange(0,batch_size, dtype=torch.int64).reshape(batch_size, 1)
    row = torch.ones(1, node_num, dtype=torch.int64)
    batch = torch.matmul(col, row).reshape(-1).to('cuda')
    
    return batch