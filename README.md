
# [ICLR 2024] Long-Short-Range Message-Passing

A comprehensive implementation of Long-Short-Range Message-Passing and state-of-the-art models for molecular dynamics simulation, optimized for Multi-GPU training.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [LSR-MP Illustration](#lsr-mp-illustration)
- [BRICS Algorithm](#brics-algorithm)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Step 1: Create Environment](#step-1-create-environment)
  - [Step 2: Activate Environment](#step-2-activate-environment)
  - [Step 3: Install Dependencies](#step-3-install-dependencies)
  - [Quick Start](#quick-start)
  - [Detailed Installation](#detailed-installation)
  - [Environment Verification](#environment-verification)
- [Configuration](#configuration)
  - [Model Parameters](#model-parameters)
  - [Training Parameters](#training-parameters)
  - [Dataset Configuration](#dataset-configuration)
  - [Cutoff Parameters](#cutoff-parameters)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Supported Components](#supported-components)
  - [Single GPU Training](#single-gpu-training)
  - [Multi-GPU Training](#multi-gpu-training)
  - [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
  - [Command Line Arguments](#command-line-arguments)
  - [Model APIs](#model-apis)
  - [Data Loading APIs](#data-loading-apis)
  - [Performance Optimization](#performance-optimization)
- [Development](#development)
  - [Contributing](#contributing)
  - [Code Structure](#code-structure)
- [Enhanced Troubleshooting](#enhanced-troubleshooting)
  - [Common Issues](#common-issues)
  - [Performance Issues](#performance-issues)
- [Citation](#citation)
- [Credits](#credits)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This repository provides:
- Implementation of Long-Short-Range Message-Passing (LSR-MP)
- Multiple state-of-the-art models for molecular dynamics simulation
- Multi-GPU training optimization
- Support for various fragmentation methods

**Note:** Preprocessing steps may take longer than expected, especially for larger molecular systems. Please be patient during the initial data preparation phase.

## Architecture

### LSR-MP Illustration

![LSR-MP architecture diagram showing long-short range message passing between molecular fragments](./plots/LSR-MP.png)

## BRICS Algorithm

The Breaking of Retrosynthetically Interesting Chemical Substructures (BRICS) method is widely used in quantum chemistry, chemical retrosynthesis, and drug discovery.

### Key Features:

1. **Bond Dissection**: Compounds are dissected at 16 predefined bond types selected by organic chemists
2. **Environmental Awareness**: Considers chemical environment (atom types) to maintain reasonable fragment sizes
3. **Filtering**: Removes extremely small fragments, duplicates, and overlapping structures
4. **Stabilization**: Adds supplementary atoms (mainly hydrogen) at bond-breaking points for chemical stability

### Visual Representation:

![BRICS algorithm supplementary figure showing molecular fragmentation process](./plots/supplementary-fig.png)

![BRICS algorithm pseudocode and workflow diagram](./plots/BRICS_algorithm.png)

## Project Structure

```
LSR-MP/
├── README.md                    # Project documentation
├── build_env.sh                 # Environment setup script
├── run_ddp.py                   # Main training script with DDP support
├── prepare_lmdb.py              # Data preprocessing utilities
├── graph.py                     # Graph construction utilities
├── xyz2mol.py                   # XYZ to molecule conversion
├── lightnp/                     # Core framework package
│   ├── LSRM/                    # Long-Short Range Message Passing implementation
│   │   ├── models/              # Model architectures
│   │   │   ├── lsrm_modules.py           # LSR-MP model implementations
│   │   │   ├── e2former_lsrmp.py         # E2Former integration
│   │   │   ├── long_short_interact_modules.py # Interaction modules
│   │   │   ├── nets/                     # Neural network components
│   │   │   └── torchmdnet/               # TorchMD-Net integration
│   │   ├── data/                # Data handling and loading
│   │   │   ├── lmdb_dataset.py          # LMDB dataset implementation
│   │   │   ├── atoms_loader.py          # Atomic data loading
│   │   │   └── pyg_wrapper.py           # PyTorch Geometric wrapper
│   │   └── utils/               # Utility functions
│   │       ├── build_group_graph.py     # Graph construction
│   │       ├── transforms.py            # Data transformations
│   │       └── model_utils.py           # Model utilities
│   ├── data/                    # General data processing
│   ├── train/                   # Training infrastructure
│   │   ├── ddp_trainer.py       # Distributed training
│   │   ├── hooks/               # Training hooks and callbacks
│   │   └── loss.py              # Loss functions
│   └── utils/                   # General utilities
├── checkpoints/                 # Model checkpoints and training logs
├── MD22/                        # MD22 dataset storage
└── plots/                       # Visualization assets
    ├── LSR-MP.png              # Architecture diagram
    ├── BRICS_algorithm.png     # BRICS workflow
    └── supplementary-fig.png   # Additional figures
```

## Prerequisites

Before installation, ensure you have:
- Python 3.7+
- CUDA-compatible GPU(s) for training
- Conda or Miniconda installed
- Git for cloning the repository

### Additional Requirements for E2Former Model

If you plan to use the E2Former model, you must first clone the official E2Former repository:

```bash
git clone https://github.com/2023huang6385/E2Former.git
```

Then modify the path in `lightnp/LSRM/models/e2former_lsrmp.py` (line 14):

```python
# Change this line to point to your E2Former installation
sys.path.append('/path/to/your/E2Former')
```

Replace `/path/to/your/E2Former` with the actual path where you cloned the E2Former repository.

## Installation

### Step 1: Create Environment

Create a new conda environment:
```bash
conda create -n LSR-MP python=3.8
```

### Step 2: Activate Environment

```bash
conda activate LSR-MP
```

### Step 3: Install Dependencies

The main packages required:
- `torch` - PyTorch framework (v1.9.0+cu111)
- `torch-geometric` - Graph neural network library (v2.0.3)
- `torch-scatter` - Scatter operations for PyTorch (v2.0.8)
- `ase` - Atomic Simulation Environment
- `rdkit` - Chemical informatics toolkit
- `wandb` - Experiment tracking
- `pytorch_lightning` - Training framework (v1.5.0)
- \

#### Quick Start

Install using the provided script:
```bash
chmod +x build_env.sh
./build_env.sh
```

#### Detailed Installation

For manual installation or customization:

```bash
# Install PyTorch with CUDA support
pip install torch==1.9.0+cu111 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# Install PyTorch Geometric and related packages
pip install torch-scatter==2.0.8 torch-sparse==0.6.10 torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric==2.0.3

# Install additional dependencies
pip install pytorch_lightning==1.5.0
pip install wandb torch-ema ase sympy
pip install opencv-python-headless
conda install yaml -y

# Install framework-specific requirements
pip install -r lightnp_env_requirements.txt
```

#### Environment Verification

Verify your installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Configuration

### Model Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--model` | Model architecture | `Visnorm_shared_LSRMNorm2_2branchSerial` | See [Available Models](#available-models) |
| `--hidden_channels` | Hidden layer dimensions | 128 | 64-512 |
| `--num_interactions` | Number of interaction layers | 6 | 1-12 |
| `--long_num_layers` | Long-range interaction layers | 2 | 1-6 |
| `--dropout` | Dropout rate | 0.1 | 0.0-0.5 |

### Training Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--lr` | Learning rate | 0.0004 | 1e-5 to 1e-2 |
| `--batch_size` | Training batch size | 16 | 1-128 |
| `--ema_decay` | Exponential moving average decay | 0.999 | 0.9-0.9999 |
| `--early_stop_patience` | Early stopping patience | 500 | 100-1000 |
| `--rho_criteria` | Convergence criteria | 0.001 | 1e-4 to 1e-2 |

### Dataset Configuration

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--molecule` | Target molecule system | `Ac_Ala3_NHMe`, `DHA`, `stachyose`, `AT_AT`, `AT_AT_CG_CG`, `double_walled_nanotube`, `buckyball_catcher` |
| `--group_builder` | Fragmentation method | `rdkit` (BRICS), `k-means`, `spectral` |
| `--dataset` | Dataset name | `my_dataset` | Custom dataset identifier |

### Cutoff Parameters

| Parameter | Description | Default | Typical Range |
|-----------|-------------|---------|---------------|
| `--otfcutoff` | On-the-fly cutoff (short-range graph) | 4.0 Å | 3.0-6.0 Å |
| `--short_cutoff_upper` | Short-range upper cutoff | 4.0 Å | 3.0-6.0 Å |
| `--long_cutoff_lower` | Long-range lower cutoff | 0.0 Å | 0.0-2.0 Å |
| `--long_cutoff_upper` | Long-range upper cutoff | 9.0 Å | 6.0-12.0 Å |

**Note**: `--otfcutoff` must equal `--short_cutoff_upper` for proper graph construction.

## Usage

### Data Preparation

Before training, prepare your molecular dynamics data:

1. **Convert XYZ to LMDB format**:
   ```bash
   python prepare_lmdb.py --input_path /path/to/xyz/files --output_path ./MD22/
   ```

2. **Build molecular graphs**:
   ```bash
   python graph.py --molecule AT_AT_CG_CG --fragmentation rdkit
   ```

3. **Verify data integrity**:
   ```bash
   python -c "from lightnp.LSRM.data import LmdbDataset; ds = LmdbDataset('./MD22/AT_AT_CG_CG/my_dataset'); print(f'Dataset size: {len(ds)}')"
   ```

### Supported Components

#### Fragment Assignment Methods:
- **BRICS** (`rdkit`) - Chemical substructure-based fragmentation
- **K-Means Clustering** (`k-means`) - Distance-based clustering
- **Spectral Clustering** (`spectral`) - Graph-based clustering

#### Supported Molecules:
- `Ac_Ala3_NHMe` - Alanine tripeptide
- `DHA` - Docosahexaenoic acid
- `stachyose` - Tetrasaccharide
- `AT_AT` - Adenine-thymine base pairs
- `AT_AT_CG_CG` - Mixed DNA base pairs
- `double_walled_nanotube` - Carbon nanotube structure
- `buckyball_catcher` - Fullerene host-guest complex

#### Available Models:
- **VisNet-LSRM** (`Visnorm_shared_LSRMNorm2_2branchSerial`) - Long-Short Range Message Passing
- **Equivariant Transformer** (`TorchMD_ET`) - Transformer-based approach
- **PaiNN** - Polarizable Atom Interaction Neural Network
- **Equiformer-LSRM** (`dot_product_attention_transformer_exp_l2_md17_lsrmserial`) - E(3)-equivariant transformer with LSRM
- **E2Former** (`E2Former`) - State-of-the-art E(3)-equivariant transformer with direct force prediction

### Single GPU Training

Train LSR-MP on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1230 \
  run_ddp.py \
    --datapath ./ \
    --model=Visnorm_shared_LSRMNorm2_2branchSerial \
    --molecule AT_AT_CG_CG \
    --dataset=my_dataset \
    --group_builder rdkit \
    --num_interactions=6 --long_num_layers=2 \
    --lr=0.0004 --rho_criteria=0.001 \
    --dropout=0 --hidden_channels=128 \
    --calculate_meanstd --otfcutoff=4 \
    --short_cutoff_upper=4 --long_cutoff_lower=0 --long_cutoff_upper=9 \
    --early_stop --early_stop_patience=500 \
    --no_broadcast --batch_size=16 \
    --ema_decay=0.999 --dropout=0.1 \
    --wandb --api_key [YOUR_WANDB_API_KEY]
```

For Equiformer-LSRM model, use:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1230 \
  run_ddp.py \
    --datapath ./ \
    --model=dot_product_attention_transformer_exp_l2_md17_lsrmserial \
    --molecule AT_AT_CG_CG \
    --dataset=my_dataset \
    --group_builder rdkit \
    --num_interactions=6 --long_num_layers=2 \
    --lr=0.0004 --rho_criteria=0.001 \
    --dropout=0 --hidden_channels=128 \
    --calculate_meanstd --otfcutoff=4 \
    --short_cutoff_upper=4 --long_cutoff_lower=0 --long_cutoff_upper=9 \
    --early_stop --early_stop_patience=500 \
    --no_broadcast --batch_size=16 \
    --ema_decay=0.999 --dropout=0.1 \
    --wandb --api_key [YOUR_WANDB_API_KEY]
```

For E2Former model (state-of-the-art E(3)-equivariant transformer), use:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1230 \
  run_ddp.py \
    --datapath ./ \
    --model=E2Former \
    --molecule AT_AT_CG_CG \
    --dataset=my_dataset \
    --group_builder rdkit \
    --num_interactions=6 \
    --lr=0.0004 --rho_criteria=0.001 \
    --dropout=0 --hidden_channels=128 \
    --calculate_meanstd --otfcutoff=4 \
    --short_cutoff_upper=4 \
    --early_stop --early_stop_patience=500 \
    --no_broadcast --batch_size=16 \
    --ema_decay=0.999 --dropout=0.1 \
    --wandb --api_key [YOUR_WANDB_API_KEY]
```

**Note**: E2Former uses direct force prediction via eSCN force blocks and does not utilize long-short range message passing. The model is based on the vanilla E2Former architecture without LSR-MP extensions.

**Important**: Before using E2Former, ensure you have:
1. Cloned the official E2Former repository: `git clone https://github.com/2023huang6385/E2Former.git`
2. Updated the path in `lightnp/LSRM/models/e2former_lsrmp.py` line 14 to point to your E2Former installation

#### Key Parameters:
- `--datapath`: Path to your dataset directory
- `--model`: Model architecture to use
- `--molecule`: Target molecule system
- `--group_builder`: Fragmentation method (rdkit, k-means, spectral)
- `--num_interactions`: Number of interaction layers
- `--lr`: Learning rate
- `--hidden_channels`: Hidden layer dimensions
- `--batch_size`: Training batch size

### Multi-GPU Training

Train using Distributed Data Parallel across multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1230 \
  run_ddp.py \
    --datapath ./ \
    --model=Visnorm_shared_LSRMNorm2_2branchSerial \
    --molecule AT_AT_CG_CG \
    --dataset=my_dataset \
    --group_builder rdkit \
    --num_interactions=6 --long_num_layers=2 \
    --lr=0.0004 --rho_criteria=0.001 \
    --dropout=0 --hidden_channels=128 \
    --calculate_meanstd --otfcutoff=4 \
    --short_cutoff_upper=4 --long_cutoff_lower=0 --long_cutoff_upper=9 \
    --early_stop --early_stop_patience=500 \
    --no_broadcast --batch_size=16 \
    --ema_decay=0.999 --dropout=0.1 \
    --wandb --api_key [YOUR_WANDB_API_KEY]
```

#### Important Notes:
- `--nproc_per_node` must equal the number of GPUs in `CUDA_VISIBLE_DEVICES`
- `--otfcutoff` must equal `--short_cutoff_upper` (defines short-range graph radius)
- Use `--wandb` with your API key for experiment tracking
- Recommended settings provide good MD22 benchmark performance

### Testing

Evaluate a trained model:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1230 run_ddp.py \
  --datapath [YOUR_DATA_PATH] \
  --model=Visnorm_shared_LSRMNorm2_2branchSerial \
  --molecule AT_AT_CG_CG \
  --dataset=[DATASET_NAME] \
  --test --restore_run [PATH_TO_TRAINED_MODEL] \
  --wandb --api_key [YOUR_WANDB_API_KEY]
```

#### Evaluation Metrics:
- **Force MAE** - Mean absolute error for atomic forces
- **Energy MAE** - Mean absolute error for system energy

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce `--batch_size` or use fewer `--hidden_channels`
2. **Preprocessing Delays**: Large molecular systems may require extended preprocessing time
3. **Missing Dependencies**: Ensure all packages are installed via `build_env.sh`

### Performance Tips:

- Use multiple GPUs for faster training on large datasets
- Adjust cutoff parameters based on your molecular system size
- Monitor training with wandb for optimal hyperparameter tuning

## API Reference

### Command Line Arguments

#### Core Arguments

```bash
# Data and Model Configuration
--datapath          # Path to dataset directory
--model             # Model architecture name
--molecule          # Target molecule system
--dataset           # Dataset identifier
--group_builder     # Fragmentation method

# Training Configuration
--lr                # Learning rate
--batch_size        # Training batch size
--num_interactions  # Number of interaction layers
--hidden_channels   # Hidden layer dimensions
--dropout           # Dropout rate

# Distributed Training  
--nproc_per_node    # Number of processes per node
--master_port       # Master port for communication

# Monitoring and Logging
--wandb             # Enable Weights & Biases logging
--api_key           # W&B API key
--log_dir           # Log directory path
```

#### Advanced Arguments

```bash
# Cutoff Parameters
--otfcutoff         # On-the-fly cutoff radius
--short_cutoff_upper # Short-range interaction cutoff
--long_cutoff_lower # Long-range interaction lower bound
--long_cutoff_upper # Long-range interaction upper bound

# Optimization
--ema_decay         # Exponential moving average decay
--early_stop        # Enable early stopping
--early_stop_patience # Early stopping patience
--rho_criteria      # Convergence criteria

# Model-Specific
--long_num_layers   # Long-range interaction layers (LSR-MP)
--calculate_meanstd # Calculate dataset statistics
--no_broadcast      # Disable parameter broadcasting
```

### Model APIs

#### LSR-MP Model

```python
from lightnp.LSRM.models.lsrm_modules import Visnorm_shared_LSRMNorm2_2branchSerial

# Initialize model
model = Visnorm_shared_LSRMNorm2_2branchSerial(
    hidden_channels=128,
    num_interactions=6,
    long_num_layers=2,
    dropout=0.1
)

# Forward pass
output = model(data)
```

#### E2Former Integration

```python
from lightnp.LSRM.models.e2former_lsrmp import E2Former

# Initialize E2Former model
model = E2Former(
    hidden_channels=128,
    num_interactions=6
)
```

### Data Loading APIs

#### LMDB Dataset

```python
from lightnp.LSRM.data import LmdbDataset, collate_fn
from torch.utils.data import DataLoader

# Load dataset
dataset = LmdbDataset('./MD22/AT_AT_CG_CG/my_dataset')

# Create data loader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)
```




### Performance Optimization

#### GPU Memory Optimization

```bash
# For limited GPU memory
--batch_size 8 --hidden_channels 64 --gradient_checkpointing

# For high-memory GPUs (>24GB)
--batch_size 32 --hidden_channels 256 --mixed_precision
```


## Development

### Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Commit** changes: `git commit -am 'Add feature'`
4. **Test** your changes
5. **Push** to branch: `git push origin feature-name`
6. **Submit** a pull request



### Code Structure

#### Core Components

```
lightnp/LSRM/
├── models/
│   ├── lsrm_modules.py      # Main LSR-MP implementations
│   ├── e2former_lsrmp.py    # E2Former integration
│   └── nets/                # Neural network primitives
├── data/
│   ├── lmdb_dataset.py      # LMDB data handling
│   └── atoms_loader.py      # Atomic structure loading
└── utils/
    ├── build_group_graph.py # Graph construction
    └── transforms.py        # Data transformations
```

#### Design Patterns

- **Factory Pattern**: Model creation via `create_model()`
- **Strategy Pattern**: Fragmentation methods in `group_builder`
- **Observer Pattern**: Training hooks and callbacks
- **Template Method**: Base classes for models and datasets


4. **Update documentation** with model parameters and usage

## Enhanced Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: CUDA version mismatch
```bash
# Error: RuntimeError: CUDA error: no kernel image is available
# Solution: Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==1.9.0+cu111 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
```

**Issue**: PyTorch Geometric compilation errors
```bash
pip install --no-cache-dir --no-binary :all: torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```

#### Training Issues

**Issue**: CUDA Out of Memory
```bash
# Error: RuntimeError: CUDA out of memory
# Solutions:
1. Reduce batch size: --batch_size 8
2. Reduce model size: --hidden_channels 64
```


#### Data Processing Issues


**Issue**: Graph construction errors
```bash
# Error: Invalid molecular graph
# Solutions:
1. Adjust cutoff radius: --otfcutoff 5.0
2. Use alternative fragmentation: --group_builder spectral
3. delete your processed lmdb/file path
```




### Performance Issues

#### Slow Training

1. **Check GPU utilization**: `nvidia-smi`
2. **Profile data loading**: Add `--profile_data_loading`
3. **Optimize batch size**: Use largest batch that fits in memory
4. **Enable mixed precision**: `--mixed_precision`
5. **Use multiple GPUs**: Scale with `--nproc_per_node`

#### Memory Optimization

```bash
# Memory-efficient settings
--batch_size 4 \
--hidden_channels 64 \
--gradient_checkpointing \
--mixed_precision \
--cpu_offload
```

#### I/O Bottlenecks

```bash
# Optimize data loading
--num_workers 4 \
--pin_memory \
--prefetch_factor 2 \
--cache_data
```

## Citation

If you use LSR-MP in your research, please consider citing:

```bibtex
@inproceedings{
li2024longshortrange,
title={Long-Short-Range Message-Passing: A Physics-Informed Framework to Capture Non-Local Interaction for Scalable Molecular Dynamics Simulation},
author={Yunyang Li and Yusong Wang and Lin Huang and Han Yang and Xinran Wei and Jia Zhang and Tong Wang and Zun Wang and Bin Shao and Tie-Yan Liu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=rvDQtdMnOl}
}
```

## Credits

This work builds upon several foundational contributions:

Equiformer
```bibtex
@inproceedings{
    liao2023equiformer,
    title={Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs},
    author={Yi-Lun Liao and Tess Smidt},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=KwmPfARgOTD}
}
```

ViSNet
```bibtex
@article{wang2024enhancing,
  title={Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing},
  author={Wang, Yusong and Wang, Tong and Li, Shaoning and He, Xinheng and Li, Mingyu and Wang, Zun and Zheng, Nanning and Shao, Bin and Liu, Tie-Yan},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={313},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

E2Former
```bibtex
@article{li2025e2former,
  title={E2Former: A Linear-time Efficient and Equivariant Transformer for Scalable Molecular Modeling},
  author={Li, Yunyang and Huang, Lin and Ding, Zhihao and Wang, Chu and Wei, Xinran and Yang, Han and Wang, Zun and Liu, Chang and Shi, Yu and Jin, Peiran and others},
  journal={arXiv preprint arXiv:2501.19216},
  year={2025}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **TorchMD-Net** team for the molecular dynamics foundation
- **PyTorch Geometric** developers for graph neural network utilities  
- **E2Former** authors for the equivariant transformer architecture
- **RDKit** community for chemical informatics tools
- **MD22** dataset contributors for benchmark data
- Research computing facilities for computational resources



