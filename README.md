
# Long-Short-Range Message-Passing: A Physics-Informed Framework to Capture Non-Local Interaction for Scalable Molecular Dynamics Simulation

* ICML 2023 Submission 4999 Authors
* This repository is a comprehensive code base which implements Long-Short-Range Message-Passing as well as a spectrum of state-of-the-art models for molecular dynamics simulation



## Install Pacakges 

* Main Pacakges used in this repo:
```
  torch
  torch-geometric
  torch-scatter
  ase
  rdkit
```

* Installation using pip:

```bash
pip install -r requirements.txt
```

## Run Model


For fragments assignments, supported methods include: 
* BRICS [rdkit], 
* K-Means Clustering [k-means] 
* Distance-based Spectral Clustering [spectral]

Supported Molecules:
* Ac_Ala3_NHMe 
* DHA 
* stachyose 
* AT_AT
* AT_AT_CG_CG 
* double_walled_nanotube 
* buckyball_catcher 

See supported arguments:

```bash
python run_ddp.py -h
```

* Run LSRM on a Single GPU 

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1230 run_ddp.py \
--datapath [YOUR_DATA_PATH] \
--model=Visnorm_shared_LSRMNorm2_2branchSerial \
--molecule AT_AT_CG_CG \
--dataset=[DATASET_ID]  \
--group_builder rdkit \
--num_interactions=6  --long_num_layers=2 \
--learning_rate=0.0004 --rho_tradeoff 0.001 \
--dropout=0 --hidden_channels 128 \
--gradient_clip \
--calculate_meanstd --otfcutoff 4 \
--short_cutoff_upper 4 --long_cutoff_lower 0 --long_cutoff_upper 9 \
--early_stop --early_stop_patience 500 \
--no_broadcast  --batch_size 16 \
--ema_decay 0.999 --dropout 0.1 \
--wandb --api_key [YOUR API KEY IN WANDB]
```

* Run LSRM using Distributed Data Parallel Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1230 run_ddp.py \
--datapath [YOUR_DATA_PATH] \
--model=Visnorm_shared_LSRMNorm2_2branchSerial \
--molecule AT_AT_CG_CG  \
--dataset=[DATASET_NAME]  \
--group_builder rdkit \
--num_interactions=6  --long_num_layers=2 \
--learning_rate=0.0004 --rho_tradeoff 0.001 \
--dropout=0 --hidden_channels 128 \
--gradient_clip \
--calculate_meanstd --otfcutoff 4 \
--short_cutoff_upper 4 --long_cutoff_lower 0 --long_cutoff_upper 9 \
--early_stop --early_stop_patience 500 \
--no_broadcast  --batch_size 16 \
--ema_decay 0.999 --dropout 0.1 \
--wandb --api_key [YOUR API KEY IN WANDB]
```

Notes:
- The above setting is a good start for a fair performance on MD22
- --nproc_per_node=4 must equal to the number of CUDA_VISIBLE_DEVICES
- --otfcutoff must equal to short_cutoff_upper, this is the radius for short-range graph
- --wandb toggle on wandb, just input your api key
- You can specify a unique id for each dataset in [DATASET_ID]. This is mainly used when comparing different fragmentation scheme under the same molecules.

* Test using a single gpu

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1230 run_ddp.py \
--datapath [YOUR_DATA_PATH] \
--model=Visnorm_shared_LSRMNorm2_2branchSerial \
--molecule AT_AT_CG_CG  \
--dataset=[DATASET_NAME]  \
--test --restore_run [PATH_TO_TRAINED_MODEL] \
--wandb --api_key [YOUR API KEY IN WANDB]
```
The evaluation metrics includes: 
- MAE for force
- MAE for energy



