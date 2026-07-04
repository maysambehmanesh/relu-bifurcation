
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyFilter(nn.Module):
    """Learnable polynomial P(A) = sum_k c_k A^k"""
    def __init__(self, A_norm, K=5, init='adjacency'):
        super().__init__()
        self.A = A_norm
        self.K = K
        coeffs = torch.zeros(K + 1)
        if init == 'adjacency':
            coeffs[1] = 1.0
        elif init == 'laplacian':
            coeffs[0] = 1.0
            coeffs[1] = -1.0
        self.coeffs = nn.Parameter(coeffs)
    
    def forward(self, X):
        result = self.coeffs[0] * X
        Ak_X = X
        for k in range(1, self.K + 1):
            Ak_X = torch.sparse.mm(self.A, Ak_X)
            result = result + self.coeffs[k] * Ak_X
        return result
    


class ReLU_PolyGCN(nn.Module):
    """ReLU + Polynomial filter"""
    def __init__(self, in_c, hid_c, out_c, n_layers, A_norm, K=5, 
                 dropout=0.0, share_poly=True):
        super().__init__()
        self.dropout = dropout
        self.share_poly = share_poly
        sizes = [in_c] + [hid_c]*(n_layers-1) + [out_c]
        self.lins = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(n_layers)])
        
        if share_poly:
            self.poly = PolyFilter(A_norm, K, 'adjacency')
        else:
            self.polys = nn.ModuleList([PolyFilter(A_norm, K, 'adjacency') 
                                       for _ in range(n_layers-1)])
    
    def forward(self, x, _ei=None):
        for i, lin in enumerate(self.lins[:-1]):
            poly = self.poly if self.share_poly else self.polys[i]
            x = poly(x)
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.lins[-1](x), dim=1)


class Sine_PolyGCN(nn.Module):
    """Sine + Polynomial filter"""
    def __init__(self, in_c, hid_c, out_c, n_layers, A_norm, K=5,
                 dropout=0.0, w0=1.0, share_poly=True):
        super().__init__()
        self.dropout = dropout
        self.w0 = w0
        self.share_poly = share_poly
        sizes = [in_c] + [hid_c]*(n_layers-1) + [out_c]
        self.lins = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(n_layers)])
        
        if share_poly:
            self.poly = PolyFilter(A_norm, K, 'adjacency')
        else:
            self.polys = nn.ModuleList([PolyFilter(A_norm, K, 'adjacency') 
                                       for _ in range(n_layers-1)])
    
    def forward(self, x, _ei=None):
        for i, lin in enumerate(self.lins[:-1]):
            poly = self.poly if self.share_poly else self.polys[i]
            x = poly(x)
            x = lin(x)
            x = torch.sin(self.w0 * x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.lins[-1](x), dim=1)
