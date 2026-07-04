

# Beyond ReLU: Bifurcation, Oversmoothing, and Topological Priors

PyTorch implementation of the paper:

**[Beyond ReLU: Bifurcation, Oversmoothing, and Topological Priors](https://arxiv.org/pdf/2602.15634)**  
*Erkan Turan, Gaspard Abel, Maysam Behmanesh, Emery Pierson, Maks Ovsjanikov*

LIX, École Polytechnique, IP Paris, Université Paris-Saclay, Université Paris Cité, ENS Paris-Saclay, CNRS, INSERM, Centre Borelli, Centre d’Analyse et de Mathématique Sociales, CNRS

## Overview

Graph Neural Networks (GNNs) learn node representations through iterative network-based message-passing. While powerful, deep GNNs suffer from oversmoothing, where node features converge to a homogeneous, non-informative state. We re-frame this problem of representational collapse from a bifurcation theory perspective, characterizing oversmoothing as convergence to a stable “homogeneous fixed point.” Our central contribution is the theoretical discovery that this undesired stability can be broken by replacing standard monotone activations (e.g., ReLU) with a class of functions. Using LyapunovSchmidt reduction, we analytically prove that this substitution induces a bifurcation that destabilizes the homogeneous state and creates a new pair of stable, non-homogeneous patterns that provably resist oversmoothing. Our theory predicts a precise, nontrivial scaling law for the amplitude of these emergent patterns, which we quantitatively validate in experiments. Finally, we demonstrate the practical utility of our theory by deriving a closed-form, bifurcation-aware initialization and showing its utility in real benchmark experiments

<img width="1168" height="392" alt="image" src="https://github.com/user-attachments/assets/fe697cb3-8f91-4332-b325-fe06804d3099" />


## 📁 Structure
```text
├── data/                        # Downloaded automatically on first run
├── hyperparam_poly/             # Best hyperparameters per dataset (JSON)
│   └── best_params_{dataset}.json
├── results_BRELU/               
├── config.yaml                  
├── main.py                      
├── model.py                     
├── load_data.py                 
├── utils.py                     
└── requirements.txt         
```

## ⚙️ Installation
Create a virtual environment (recommended):
```text
python -m venv env
source env/bin/activate      # Linux / Mac
# env\Scripts\activate       # Windows
```
Install dependencies:
```text
pip install -r requirements.txt
```

## 🧪 Configuration

All experiment settings are controlled via ```config.yaml ```



## 📊 Data

Datasets are downloaded automatically on the first run via [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) and [OGB](https://ogb.stanford.edu/) and saved to the `data/` directory. No manual download is required.

Supported datasets: `Texas`, `Wisconsin`, `Cornell`, `Film`, `Squirrel`, `Chameleon`, `Cora`, `CiteSeer`, `PubMed`, `Computers`, `Photo`, `CoauthorCS`, `ogbn-arxiv`.

## 🚀 Usage

Train with default config:

```text
python main.py --config config.yaml --gpu 0
```

Train on specific datasets:

```text
python main.py --gpu 0 --datasets Texas Wisconsin Cornell
```

Use optimised hyperparameters:

```text
python main.py --gpu 0 --use_best_params --params_dir hyperparam_poly_gcn_opt
```



## Citation
If you find this work useful, please cite:
```bash
@article{turan2026brelu,
  title={Beyond ReLU: Bifurcation, Oversmoothing, and Topological Priors},
  author={Turan, Erkan and Abel, Gaspard and Behmanesh, Maysam and Pierson, Emery and Ovsjanikov, Maks},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  month = {6-11 Jul},
  publisher = {PMLR},
  year={2026}
}
```


