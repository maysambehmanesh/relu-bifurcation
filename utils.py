import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomNodeSplit


# Reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Hybrid activation
class HybridActivation(nn.Module):
    def __init__(self, a=math.pi/6, w0=1.0, negative_slope=0.05):
        super().__init__()
        self.a, self.w0, self.neg = a, w0, negative_slope
    
    def forward(self, x):
        in_range = (x >= -self.a) & (x <= self.a)
        s = torch.sin(self.w0 * x)
        r = F.relu(x)
        l = F.leaky_relu(x, self.neg)
        return torch.where(in_range, s, torch.where(x > self.a, r, l))


# Build normalized adjacency + top eigenvector
def build_A_norm_sparse(data, device):
    num_nodes = data.num_nodes
    row, col = data.edge_index
    idx = torch.stack([row, col], dim=0)
    vals = torch.ones(idx.size(1), device=device)
    # add self-loops
    self_idx = torch.arange(num_nodes, device=device)
    self_edges = torch.stack([self_idx, self_idx], dim=0)
    idx = torch.cat([idx, self_edges], dim=1)
    vals = torch.cat([vals, torch.ones(num_nodes, device=device)], dim=0)
    deg = torch.zeros(num_nodes, device=device).scatter_add_(0, idx[0], vals)
    dinv_sqrt = (deg + 1e-12).pow(-0.5)
    norm_vals = dinv_sqrt[idx[0]] * vals * dinv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, norm_vals, (num_nodes, num_nodes)).coalesce()

@torch.no_grad()
def top_eigvec_power(A_norm, iters=100):
    N = A_norm.size(0)
    u = torch.randn(N, device=A_norm.device)
    u = u / (u.norm() + 1e-12)
    for _ in range(iters):
        u = torch.sparse.mm(A_norm, u.unsqueeze(1)).squeeze(1)
        u = u / (u.norm() + 1e-12)
    return u



# Perturbation Operator (A_norm + eps * P)
class NodeRank2AdjPerturb(nn.Module):
    def __init__(self, A_norm, eps_init=0.05, power_iters=100):
        super().__init__()
        self.A = A_norm
        u = top_eigvec_power(self.A, iters=power_iters)
        self.register_buffer('u', u, persistent=False)
        N = self.A.size(0)
        w0 = torch.zeros(N, device=u.device)
        nn.init.normal_(w0, mean=0.0, std=0.02)
        w0 = w0 - (u @ w0) * u
        self.w = nn.Parameter(w0)
        self.eps = nn.Parameter(torch.tensor(eps_init, device=u.device))
    
    def forward(self, X):
        AX = torch.sparse.mm(self.A, X)
        w = self.w - (self.u @ self.w) * self.u
        wTX = (w.unsqueeze(0) @ X).squeeze(0)
        uTX = (self.u.unsqueeze(0) @ X).squeeze(0)
        PX = self.u.unsqueeze(1) * wTX.unsqueeze(0) + w.unsqueeze(1) * uTX.unsqueeze(0)
        return AX + self.eps * PX



def siren_init(model, w0=1.0, c=6):
    first = True
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "weight") and m.weight is not None:
                fan_in = m.weight.size(1)
                bound = (1.0/fan_in) if first else math.sqrt(c/fan_in)/w0
                first = False
                m.weight.uniform_(-bound, bound)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.uniform_(-math.pi, math.pi)

def tempered_siren_init(model, w0=1.0, c=2):
    first = True
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "weight") and m.weight is not None:
                fan_in = m.weight.size(1)
                bound = math.sqrt(c/fan_in)/(w0*(2 if not first else 4))
                first = False
                m.weight.uniform_(-bound, bound)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.uniform_(-math.pi/4, math.pi/4)



def ginibre_graph_aware_init(model, data, eps=0.0):
    device = data.edge_index.device
    num_nodes = data.num_nodes
    
    # Build sparse normalized adjacency
    from utils import build_A_norm_sparse
    A_norm = build_A_norm_sparse(data, device)

    v = torch.randn(num_nodes, device=device)
    v = v / v.norm()
    
    for _ in range(100):
        v = torch.sparse.mm(A_norm, v.unsqueeze(1)).squeeze(1)
        v = v / (v.norm() + 1e-12)
    
    Av = torch.sparse.mm(A_norm, v.unsqueeze(1)).squeeze(1)
    lam_max = abs((v @ Av).item())
    
    # Initialize
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "weight") and m.weight is not None:
                n_in = m.weight.size(1)
                std = (1.0 + eps) / (lam_max * math.sqrt(n_in))
                m.weight.normal_(0.0, std)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.normal_(0.0, 0.02)
    
    print(f"Ginibre init: λ_max={lam_max:.3f}, ε={eps:.2f}")
    


def train_epoch(model, data, opt):
    model.train()
    opt.zero_grad(set_to_none=True)
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()

@torch.no_grad()
def eval_epoch(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = out[mask].argmax(dim=1)
        accs.append((pred == data.y[mask]).float().mean().item())
    return accs






class NodeRank2AdjPerturb(nn.Module):
    def __init__(self, A_norm, eps_init=0.05, power_iters=100):
        super().__init__()
        self.A = A_norm
        
        u = top_eigvec_power(self.A, iters=power_iters)
        self.register_buffer('u', u, persistent=False)
        
        N = self.A.size(0)
        
        w0 = torch.zeros(N, device=u.device)
        nn.init.normal_(w0, mean=0.0, std=0.02)
        w0 = w0 - (u @ w0) * u  
        self.w = nn.Parameter(w0)
        

        self.eps = nn.Parameter(torch.tensor(eps_init, device=u.device))
    
    def forward(self, X):
        AX = torch.sparse.mm(self.A, X)
        
        # Keep w orthogonal to u
        w = self.w - (self.u @ self.w) * self.u
        
        # Compute perturbation: 
        wTX = (w.unsqueeze(0) @ X).squeeze(0)  
        uTX = (self.u.unsqueeze(0) @ X).squeeze(0)  
        PX = self.u.unsqueeze(1) * wTX.unsqueeze(0) + w.unsqueeze(1) * uTX.unsqueeze(0)
        
        # Return: (A + ε·P)·X
        return AX + self.eps * PX