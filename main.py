"""
Train ReLU and Sine models with Polynomial(A) filters.
Includes: Shared and Per-Layer variants
"""

import os
import math
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import yaml
from utils import set_seed, build_A_norm_sparse, ginibre_graph_aware_init, train_epoch, eval_epoch
from load_data import load_dataset 
from model import PolyFilter, ReLU_PolyGCN,Sine_PolyGCN



def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def get_device(gpu_id=None):
    if gpu_id is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif gpu_id < 0:
        return "cpu"
    else:
        if torch.cuda.is_available():
            if gpu_id < torch.cuda.device_count():
                return f"cuda:{gpu_id}"
            else:
                print(f"GPU {gpu_id} not available. Using cuda:0")
                return "cuda:0"
        else:
            print(f"CUDA not available. Using CPU")
            return "cpu"


## model constructor
def get_model_constructor(m_name, poly_k):
    constructors = {
        "ReLU-Poly(A)-sh": lambda h, d: ReLU_PolyGCN(in_dim, h, out_dim, d, A_norm_sparse, poly_k, cfg['model']['dropout'], share_poly=True),
        "ReLU-Poly(A)-pl": lambda h, d: ReLU_PolyGCN(in_dim, h, out_dim, d, A_norm_sparse, poly_k, cfg['model']['dropout'], share_poly=False),
        "Sine-Poly(A)-sh": lambda h, d: Sine_PolyGCN(in_dim, h, out_dim, d, A_norm_sparse, poly_k,cfg['model']['dropout'], cfg['model']['w0'], share_poly=True),
        "Sine-Poly(A)-pl": lambda h, d: Sine_PolyGCN(in_dim, h, out_dim, d, A_norm_sparse, poly_k,cfg['model']['dropout'], cfg['model']['w0'], share_poly=False),
    }
    return constructors.get(m_name)



def load_best_params_for_dataset(dataset_name, params_dir="hyperparam_poly"):
    params_file = os.path.join(params_dir, f"best_params_{dataset_name}.json")
    
    if os.path.exists(params_file):
        print(f"Loading best parameters from: {params_file}")
        with open(params_file, 'r') as f:
            model_params = json.load(f)
        return model_params
    else:
        print(f"Warning: {params_file} not found. using default parameters.")
        return None


def get_model_params(model_name, dataset_name, init_name, model_params_dict, default_params):
    if model_params_dict and model_name in model_params_dict:
        if init_name in model_params_dict[model_name]:
            params = model_params_dict[model_name][init_name].copy()
            print(f"Using optimized parameters for {model_name} ({init_name})")
            return params
        else:
            available_inits = list(model_params_dict[model_name].keys())
            if available_inits:
                params = model_params_dict[model_name][available_inits[0]].copy()
                print(f"Using {available_inits[0]} parameters for {model_name} ({init_name})")
                return params
    
    print(f"Using default parameters for {model_name} ({init_name})")
    return default_params



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train BRELU')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--datasets', nargs='+', help='Datasets to train on')
    parser.add_argument('--use_best_params', action='store_true', help='Use optimized hyperparameters for each model', default=True)
    parser.add_argument('--params_dir', type=str, default='hyperparam_poly', help='Directory containing best_params_{dataset}.json files')
    parser.add_argument('--poly_k', type=int, help='Polynomial order K')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension', default=64)
    parser.add_argument('--depth', type=int, help='Number of layers')
    args = parser.parse_args()
    
    cfg = load_config(args.config)

    datasets = args.datasets or cfg['datasets']
    model_names = cfg['model_names']
    poly_k = args.poly_k if args.poly_k is not None else cfg['model']['poly_k']
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else cfg['model']['hidden_dim']
    depth = args.depth if args.depth is not None else cfg['model']['depth']

    OUT_DIR = cfg['out_dir']
    os.makedirs(OUT_DIR, exist_ok=True)

    torch.set_default_dtype(torch.float32) 

    device = get_device(args.gpu)
    print(f"Using device: {device}")
    
    results = {}
    
    for DATASET in datasets:
        print(f"\n{'-'*90}\nDataset: {DATASET}\n{'-'*90}")
        
        # Load model specific best parameters
        model_params_dict = None
        if args.use_best_params:
            model_params_dict = load_best_params_for_dataset(DATASET, args.params_dir)
        
        default_params = {
            'hidden_dim': hidden_dim,
            'dropout': cfg['model']['dropout'],
            'lr': cfg['training']['lr'],
            'weight_decay': cfg['training']['weight_decay'],
            'epochs': cfg['training']['epochs'],
            'w0': cfg['model']['w0'],
            'depth': depth,
            'patience': cfg['training']['patience'],
            'val_every': cfg['training']['val_every'],
            'poly_k': poly_k,
        }
        # Load data
        data, dataset = load_dataset(DATASET, use_public_split=cfg['split']['use_public_split'],
            train_ratio=cfg['split']['train_ratio'],
            val_ratio=cfg['split']['val_ratio'],
            test_ratio=cfg['split']['test_ratio'])
        data = data.to(device)
        in_dim = dataset.num_node_features
        out_dim = dataset.num_classes
        
        # Build adjacency matrix
        A_norm_sparse = build_A_norm_sparse(data, device)
        
               
        # Initialization 
        results[DATASET] = {}
        INITS = {
            "Ginibre (subcritical)": lambda m: ginibre_graph_aware_init(m, data, -0.5),
            "Ginibre (critical)": lambda m: ginibre_graph_aware_init(m, data, 0.0),
            "Ginibre (supercritical)": lambda m: ginibre_graph_aware_init(m, data, 0.1),
        }
               
        # ** Train model **
        for m_name in model_names:
            results[DATASET][m_name] = {}
            
            for init_name, init_func in INITS.items():
                print(f"\n{'-'*80}\nTraining {m_name} with {init_name}\n{'-'*80}")
                
                # Get model-specific parameters
                params = get_model_params(m_name, DATASET, init_name, model_params_dict, default_params)
                
                hidden = params['hidden_dim']
                dropout = params['dropout']
                depth = params['depth']
                w0 = params.get('w0', cfg['model']['w0'])
                lr = params['lr']
                weight_decay = params['weight_decay']
                epochs = params.get('epochs', cfg['training']['epochs'])
                patience = params.get('patience', cfg['training']['patience'])
                val_every = params.get('val_every', cfg['training']['val_every'])
                poly_k = params.get('poly_k', args.poly_k)

                
                print(f"Parameters: hidden={hidden}, dropout={dropout:.3f}, depth={depth}, "
                      f"poly_k={poly_k}, lr={lr:.6f}, weight_decay={weight_decay:.6f}")

                acc_list = []

                for run_idx, seed in enumerate(cfg['experiment']['seeds']):
                    print(f"\nRun {run_idx+1}/{cfg['experiment']['n_runs']} | Seed {seed}")
                    set_seed(seed)

                    constructor = get_model_constructor(m_name, poly_k)
                    if constructor is None:
                        print(f"Warning: No constructor for {m_name}, skipping")
                        continue
                    
                    model = constructor(hidden, depth).to(device)

                    init_func(model)
                        

                    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    
                    ## Training 
                    best_val, best_test, wait = 0, 0, 0
                    
                    for ep in tqdm(range(epochs), 
                                  desc=f"{DATASET} | {m_name} [{init_name}] | run {run_idx+1}"):
                        _ = train_epoch(model, data, opt)
                        
                        if (ep + 1) % val_every == 0 or ep == epochs - 1:
                            tr, va, te = eval_epoch(model, data)
                            
                            if va > best_val:
                                best_val, best_test, wait = va, te, 0
                            else:
                                wait += val_every
                            
                            if wait >= patience:
                                break
                    
                    acc_list.append(best_test)
                    print(f"Run {run_idx+1} test acc: {best_test:.4f}")
                
                # Compute results
                acc_array = np.array(acc_list)
                mean_acc = acc_array.mean()
                std_acc = acc_array.std()

                results[DATASET][m_name][init_name] = {
                    "mean": float(mean_acc),
                    "std": float(std_acc),
                    "runs": [float(a) for a in acc_array]
                }

                print(f"\n{DATASET} | {m_name} ({init_name}) FINAL: {mean_acc:.4f} ± {std_acc:.4f}")


    print("\n" + "-"*90)
    print(f"FINAL ACCURACY COMPARISON (mean ± std over {cfg['experiment']['n_runs']} runs)")
    print("-"*90)

    for dataset_name, model_dict in results.items():
        print(f"\n{dataset_name}")
        for model_name, init_dict in model_dict.items():
            print(f"\n{model_name}")
            for init_name, stats in init_dict.items():
                mean = stats["mean"]
                std = stats["std"]
                print(f"  {init_name:30s}: {mean:.4f} ± {std:.4f}")


    # Save results
    output_file = os.path.join(OUT_DIR, "results_BRELU.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")