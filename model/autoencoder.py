from .gat import GAT
from utils.utils import create_norm
from functools import partial
from itertools import chain
from .loss_func import sce_loss
import torch
import torch.nn as nn
import dgl
import random
import torch.nn.functional as F
import dgl.function as fn

def build_model(ndim_in,ndim_out,edim,num_layer,n,p):

    model =  DGI(ndim_in,ndim_out,edim,num_layer,n,p)
    return model

class DGI(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim,num_layer,n,p):
      super(DGI, self).__init__()
      self.encoder = SAGE(ndim_in, ndim_out, edim,num_layer)
      self.loss = nn.BCEWithLogitsLoss()
      self.n = n
      self.p = p

    def forward(self, g):
      n_features = g.ndata['h']
      e_features = g.edata['h']
      positive = self.encoder(g, n_features, e_features, corrupt=False)
      negative = self.encoder(g, n_features, e_features, corrupt=True)
      # positive = torch.mean(positive, dim=0)
      # negative = torch.mean(negative, dim=0)
      l1 = self.loss(positive,  torch.full_like(positive, self.p))
      l2 = self.loss(negative,  torch.full_like(negative, self.n))

      return l1 + l2

    def embed(self, g):
        n_features = g.ndata['h']
        e_features = g.edata['h']
        rep = self.encoder(g, n_features,e_features,corrupt=False)
        return rep

class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim,num_layer):
      super(SAGE, self).__init__()
      self.layers = nn.ModuleList()
      self.layers.append(SAGELayer(ndim_in, edim, ndim_out))
      if num_layer > 1 :
          for i in range(num_layer-1):
              self.layers.append(SAGELayer(ndim_out, edim, ndim_out))
    def forward(self, g, nfeats, efeats, corrupt=False):
      if corrupt:
        e_perm = torch.randperm(g.number_of_edges())
        #n_perm = torch.randperm(g.number_of_nodes())
        efeats = efeats[e_perm]
        #nfeats = nfeats[n_perm]
      for i, layer in enumerate(self.layers):
        nfeats = layer(g, nfeats, efeats)
      return nfeats

class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out):
      super(SAGELayer, self).__init__()
      self.W_apply = nn.Linear(ndim_in + ndim_out , ndim_out)
      self.activation = F.relu
      self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
      self.W_edge = nn.Linear(ndim_in + ndim_out ,ndim_out )
      self.reset_parameters()

    def reset_parameters(self):
      gain = nn.init.calculate_gain('relu')
      nn.init.xavier_uniform_(self.W_apply.weight, gain=gain)

    def message_func(self, edges):
      # return {'m':  edges.data['h']}
        return {'m': self.W_msg(torch.cat([edges.src['h'],edges.data['h']], 1))}

    def forward(self, g_dgl, nfeats, efeats):
      with g_dgl.local_scope():
        g = g_dgl
        g.ndata['h'] = nfeats
        g.edata['h'] = efeats
        g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
        g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1)))

        node_h = g.ndata['h']

        # 拼接
        node_fin = torch.cat((node_h,nfeats), dim=1)

        # （如果你要喂进一个线性层，比如 self.W_edge）
        node = self.W_edge(node_fin)
        # u, v = g.edges()
        # edge = self.W_edge(torch.cat((g.srcdata['h'][u], g.dstdata['h'][v]), 2))
        return g.ndata['h']











        