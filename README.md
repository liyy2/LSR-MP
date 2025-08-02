
# [ICLR 2024] Long-Short-Range Message-Passing


* This repository is a comprehensive code base that implements Long-Short-Range Message-Passing as well as a spectrum of state-of-the-art models for molecular dynamics simulation

* This code base is designed and optimized for Multi-GPU training

  

## Illustration of LSR-MP

![image-20230319125927129](./plots/LSR-MP.png)



## BRICS Algorithm Introduction

The Breaking of Retrosynthetically Interesting Chemical Substructures (BRICS) method is one of the most widely employed strategies in the communities of quantum chemistry, chemical retrosynthesis, and drug discovery. We summarize the key points of BRICS as follows:

   * A compound is first dissected into multiple substructures at predefined 16 types of bonds that are selected by organic chemists. In addition, BRICS also takes into account the chemical environment near the bonds, e.g. the types of atoms, to make sure that the size of each fragment is reasonable and the characteristics of the compounds are  kept as much as possible.
   * BRICS method then applies substructure filters to remove extremely small fragments (for example single atoms), duplicate fragments, and fragments with overlaps.
   *  Finally, BRICS concludes the fragmentation procedure by adding supplementary atoms (mostly hydrogen atoms) to the fragments at the bond-breaking points and makes them chemically stable. 

We included a visual representation and pseudocode of the BRICS algorithm as follows:

<img src="./plots/supplementary-fig.png"  />



<img src="./plots/BRICS_algorithm.png"  />



## Install Pacakges 

* Main Pacakges used in this repo:
```
  torch
  torch-geometric
  torch-scatter
  ase
  rdkit
```

* Make a new conda environments:

```bash
conda create -n LSR-MP
```


* Activate the new conda environments:

```bash
conda activate LSR-MP
```

* Installation using pip:

```bash
chmod +X build_env.sh
./buil_env.sh
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

Currently Supported Models:

* Visnorm_shared_LSRMNorm2_2branchSerial (VisNet-LSRM)
* TorchMD_ET (Equivariant Transformer)
* PaiNN




### Run LSRM on a Single GPU 

```bash
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=1230 \
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
    --ema_decay=0.999 --dropout=0.1
```

### Run LSRM using Distributed Data Parallel Training

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

### Test using a single gpu

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



