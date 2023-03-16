# Install Pacakges 
```bash
pip install -r requirements.txt
```

# Run Model

You can specify a unique id for each dataset in [DATASET_ID].
For fragments assignments, supported methods include rdkit[BRICS], kmeans, spectral.
For molecule, Ac_Ala3_NHMe, DHA, stachyose, AT_AT, AT_AT_CG_CG, double_walled_nanotube, buckyball_catcher are supported.

* Single GPU Version
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1230 run_ddp.py \
--datapath [YOUR_DATA_PATH] \
--model=Visnorm_shared_LSRMNorm2_2branchSerial \
--molecule AT_AT_CG_CG --test_interval 100  \
--dataset=[DATASET_ID]  \
----group_builder rdkit \
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

* Distributed Data Parallel Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1230 run_ddp.py \
--datapath [YOUR_DATA_PATH] \
--model=Visnorm_shared_LSRMNorm2_2branchSerial \
--molecule AT_AT_CG_CG --test_interval 100  \
--dataset=[DATASET_NAME]  \
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
- --nproc_per_node=4 must equal to the number of CUDA_VISIBLE_DEVICES
- --otfcutoff must equal to short_cutoff_upper, this is the radius for short-range graph
