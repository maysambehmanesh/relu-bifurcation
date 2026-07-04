import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork, Actor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import add_remaining_self_loops, to_undirected



try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage, EdgeStorage, NodeStorage
    torch.serialization.add_safe_globals([
        DataEdgeAttr, 
        DataTensorAttr,
        GlobalStorage,
        EdgeStorage,
        NodeStorage
    ])
except (ImportError, AttributeError):
    pass

# Create train/val/test splits based on ratios
def create_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    
    num_nodes = data.num_nodes
    num_classes = data.y.max().item() + 1
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        class_indices = (data.y == c).nonzero(as_tuple=True)[0]
        n = len(class_indices)
        
        perm = torch.randperm(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_mask[class_indices[perm[:n_train]]] = True
        val_mask[class_indices[perm[n_train:n_train + n_val]]] = True
        test_mask[class_indices[perm[n_train + n_val:]]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


## Load graph benchmark datasets.
def load_dataset(dataset_name, data_root='data', use_public_split=True, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):

    os.makedirs(data_root, exist_ok=True)
    dataset_path = os.path.join(data_root, dataset_name)
    
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=dataset_path, name=dataset_name, split='public')
        
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root=dataset_path, name=dataset_name)
        
    elif dataset_name in ['CS', 'CoauthorCS']:
        dataset = Coauthor(root=dataset_path, name='CS')
        
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=dataset_path)
        data = dataset[0]
        
        if use_public_split:
            split_idx = dataset.get_idx_split()
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[split_idx['train']] = True
            data.val_mask[split_idx['valid']] = True
            data.test_mask[split_idx['test']] = True
        else:
            data = create_split(data, train_ratio, val_ratio, test_ratio)
        
        if data.y.dim() > 1:
            data.y = data.y.squeeze(1)
        
        return data, dataset
        
    elif dataset_name in ['Texas', 'Wisconsin', 'Cornell']:
        dataset = WebKB(root=dataset_path, name=dataset_name)
        
    elif dataset_name in ['Film', 'Actor']:
        dataset = Actor(root=dataset_path)
        
    elif dataset_name in ['Squirrel', 'Chameleon']:
        dataset = WikipediaNetwork(root=dataset_path, name=dataset_name.lower())
        # ## control self loops for Chameleon/Squirrel
        # dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index)
        # ## control undirected for Chameleon/Squirrel
        # dataset.data.edge_index = to_undirected(dataset.data.edge_index)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Apply custom split if needed
    if not use_public_split:
        data = dataset[0]
        data = create_split(data, train_ratio, val_ratio, test_ratio)
    else:
        data = dataset[0]
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            data = create_split(data, train_ratio, val_ratio, test_ratio)
        elif data.train_mask.dim() > 1:
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]
    
        
    return data, dataset
