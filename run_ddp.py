
import random
import os
import datetime
import logging
import torch
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json
from lightnp.data.datasets.qm9 import get_statistics
from lightnp.LSRM.utils.transforms import *
from lightnp.LSRM.data import collate_fn, LmdbDataset
from torch.utils.data import DataLoader
from lightnp.LSRM.models.torchmdnet.models.model import create_model
from lightnp.LSRM.models.lsrm_modules import Visnorm_shared_LSRMNorm2_2branchSerial
import lightnp as ltnp
from lightnp.utils.ltnp_utils import Logger_Lin
import argparse
from tensorboardX import SummaryWriter
import wandb



from ase.units import kcal,mol


SEED = 48

# kcal/mol 0.4

def cal_num_trainable_parm(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # cudnn.benchmark = True
    # cudnn.deterministic = True

def parse_args(jupyter=False):
    parser = argparse.ArgumentParser()
    ######################## DATA Loading ########################
    parser.add_argument('--datapath', type=str, default='./dataset', help='path to the dataset')
    parser.add_argument('--molecule', type=str, default='chig', help = 'molecule name, supported [chig, Ac_Ala3_NHM, DHA, AT_AT_CG_CG, AT_AT, stachyose]')
    parser.add_argument('--dataset', type=str, default='radius3_broadcast', help='Name of the precalculated dataset, should be consistent with the name of the folder in prepare_lmdb.py')
    parser.add_argument('--amlt', action='store_true', default=False, help='deprecated')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--model_path', type=str,default='./checkpoints', help='path to the model')
    parser.add_argument('--mean', type=str,default='./logs', help = 'path to the mean and std of the dataset')
    parser.add_argument('--calculate_meanstd', action='store_true', default=False, help='calculate mean and std of the dataset, must toggle on')
    parser.add_argument('--group_builder', type=str, default='kmeans', choices=['rdkit', 'kmeans','spectral','spectral_two'], help='which group builder to use, BRICS method is rdkit')
    ######################## Model ########################
    parser.add_argument('--model', type=str, default="TorchMD_Norm")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_interactions', type=int, default=6)
    parser.add_argument('--long_num_layers', type=int, default=3)
    parser.add_argument('--adaptive_cutoff', action = 'store_true', default=False)
    parser.add_argument('--short_cutoff_lower', type=float, default=0.0)
    parser.add_argument('--short_cutoff_upper', type=float, default=8.0)
    parser.add_argument('--long_cutoff_lower', type=float, default=0.0)
    parser.add_argument('--long_cutoff_upper', type=float, default=9.0)
    parser.add_argument('--otfcutoff', type=float, default=5.0)
    parser.add_argument('--group_center', type=str, default='center_of_mass')
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--not_otf_graph', action='store_true', default = False, 
                        help = 'on the fly graph construction, only for TorchMD_Norm')
    parser.add_argument('--no_broadcast', action='store_true', default=False)
    parser.add_argument('--num_rbf', type=int, default=50)
    ######################## Optimizer ########################
    # parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--lr_patience', type=int, default=30)
    parser.add_argument('--fp16', default=False, action='store_true')
    # parser.add_argument('--gradient_clip', default=False, action='store_true')
    # parser.add_argument('--AMSGrad', default=False, action='store_true')
    # parser.add_argument('--warmup_steps', type = int, default = 1000)
    parser.add_argument('--test_interval', type= int, default = 600)
    parser.add_argument('--loss', type=str, default="MSE", choices=["MSE", "L2MAE"])
    parser.add_argument('--lr_std',  type=int, default=0,
                        help='whether to use learning rate diveded by std (default: 0)')
    
    ######################## Training ########################
    parser.add_argument('--not_regress_forces', action='store_true', default=False, help='whether to use forces in training')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='load checkpoint')
    parser.add_argument('--inductive', action='store_true', default=False, help='inductive learning, only implemented for HUT Dataset')
    parser.add_argument('--early_stop', action='store_true', default=False, help='early stopping, default patience is 30')
    parser.add_argument('--early_stop_patience', type=int, default=500, help='early stopping, default patience is 30')
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--energy_weight', type=float, default=0.2)
    parser.add_argument('--force_weight', type=float, default=0.8)
    parser.add_argument('--rho_criteria', type=float, default=.1)
    parser.add_argument('--sample_size', type=int, default=-1)
    parser.add_argument('--train_prop', type=float, default=0.95)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=24) ## gpu device count
    parser.add_argument('--local_rank', type=int, default=0) ## gpu device countpu device count
    parser.add_argument('--master_port', type=str, default="0000") ## gpu device countpu device count
    ######################## Optimizer ########################
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-3,
                        help='weight decay (default: 5e-3)')
    # learning rate schedule parameters (timm)
    ######################## Schedulers ########################
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--epochs', type=int, default=2500, metavar='EPOCHS',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler -default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--denoise', '--dn', type=int, default=0, metavar='DN',
                        help='Noisy node training')
    parser.add_argument('--noise-levels', '--nl', type=float, default=0.1, metavar='NL')
    ######################## Logging ########################
    parser.add_argument('--wandb', action='store_true', default=False) ## gpu device count
    parser.add_argument('--restore_run', type=str, default=None, help="restore run, timestamp_config['model']")
    parser.add_argument('--restore_run_wandb', type=str, default=None, help='restore wandb run from clouds')
    parser.add_argument('--name', type=str, default=None, help='name of the experiment')
    parser.add_argument('--notes', type=str, default=None, help='description of the experiment')
    parser.add_argument('--tags', type=str, default=None, help='tags of the experiment')
    parser.add_argument('--group', type=str, default=None, help='project name of the experiment')
    parser.add_argument('--api_key', type=str, default=None, help='wandb api key')
    parser.add_argument('--one_label', type = int, default=0, help='one label for all the data')
    



    if(jupyter):
        args = parser.parse_args(args = [])
    else:
        args = parser.parse_args()
    
    args.num_workers = min(args.num_workers,args.batch_size)

    # if os.environ["LOCAL_RANK"] is not None:
    #     args.local_rank = int(os.environ["LOCAL_RANK"])
    config = {}
    for key, value in vars(args).items():
        if key.startswith('not_'):
            config[key[4:]] = not value
        config[key] = value
    
    print(config)
    return config, args

def prepare_dataset(config):
    if os.path.exists(os.path.join(config['datapath'], f'MD22/{config["molecule"]}', config['dataset'])):
        print('Dataset already exists')
        return
    
    os.system(f'python prepare_lmdb.py --datapath {config["datapath"]} --broadcast_radius {config["otfcutoff"]} \
              --out_path {config["datapath"]}  --molecule {config["molecule"]} --group_builder {config["group_builder"]} \
              --dataset_identifier {config["dataset"]}')


def amlt_config(config):
    r'''
    Change Configuration File According to the setting of amlt
    '''
    config['datapath'] = os.path.join(os.environ.get('AMLT_DATA_DIR'),'dataset')
    config['model_path'] = os.path.join(os.environ.get('AMLT_OUTPUT_DIR'), 'checkpoints')
    config['name'] = os.environ.get('AMLT_JOB_NAME') if config['name'] is None else config['name']
    config['group'] = os.environ.get('AMLT_EXPERIMENT_NAME') if config['group'] is None else config['group']
    config['notes'] = os.environ.get('AMLT_DESCRIPTION') if config['notes'] is None else config['notes']
    
    return config


def get_model(config,mean,std,regress_forces,atomref):
    if config['model'] in ["TorchMD_Norm", "TorchMD_ET", "PaiNN"]:
        model = create_model(config, mean=mean, std=std, atomref=atomref)
    elif config["model"].startswith("TorchMD_NeurIPS_LSRM") or config["model"].startswith("TorchMD_NeurIPS_PointTransformer_LSRM") or \
        config["model"].startswith("Visnorm"):
        model = eval(config["model"])(regress_forces = regress_forces,
                 hidden_channels=config["hidden_channels"],
                 num_layers=config['num_interactions'],
                 num_rbf=50,
                 rbf_type="expnorm",
                 trainable_rbf=False,
                 activation="silu",
                 attn_activation="silu",
                 neighbor_embedding=True,
                 num_heads=8,
                 distance_influence="both",
                 short_cutoff_lower=config["short_cutoff_lower"],
                 short_cutoff_upper=config["short_cutoff_upper"], ###10
                 long_cutoff_lower=config["long_cutoff_lower"],
                 long_cutoff_upper=config["long_cutoff_upper"],
                 mean = mean,
                 std = std,
                 atom_ref = atomref,
                 max_z=100,
                 max_num_neighbors=32,
                 group_center='center_of_mass',
                 tf_writer = None,
                 config=config)
    else:
        raise NotImplementedError
    
    config['params_num'] = cal_num_trainable_parm(model)

    return model


def get_stats(config, unit, train_set):
    print(config['molecule'])
    if config['molecule'] == 'chig':
        config["meanstd_level"] = "molecule_level"
        atomref = None
        mean = torch.Tensor([0])
        std = torch.Tensor([127.8126])/unit
    elif config['molecule'] in [
        'Ac_Ala3_NHMe',
        'DHA',
        "stachyose",
        'AT_AT',
        'AT_AT_CG_CG',
        'buckyball_catcher',
        'double_walled_nanotube'
        ]:
        # Now every process will calculate the mean and std of the training set
        # Low priority TODO: this is not efficient, we should only calculate it once using distributed data parallel
        train_loader = DataLoader(train_set, batch_size=1, 
                                  collate_fn = collate_fn(unit = unit, with_force = config['regress_forces']), 
                                  shuffle=False, num_workers = config['num_workers'])
        atomref = None #get_atomref(train_set, "energy", data_len = None, atomic_number_max = 60)
        mean, std = get_statistics(train_set, train_loader, 'energy', False, atomref=None) #prop_divide_by_atoms: False. As in visnet setting, its energy mean and std is not divided by atoms.
        config["meanstd_level"] = "molecule_level"
        print(config["local_rank"], mean, std)
        print(atomref,mean,std)
    else:
        assert(False)
    return mean, std, atomref, config

def get_dataset(config):
    if config['molecule'] in [
        'buckyball_catcher',
        'double_walled_nanotube']:
        if (config['dataset']!='radius3_broadcast_kmeans_rmMean'):
            print("Please use radius3_broadcast_kmeans_rmMean for ", config['molecule'])
        dataset = LmdbDataset(os.path.join(config['datapath'], f'MD22/{config["molecule"]}', config['dataset']))
        unit = 1    
    
    elif config['molecule'] in [
        'Ac_Ala3_NHMe',
        'DHA',
        'AT_AT_CG_CG',
        'AT_AT',
        "stachyose"]:
        dataset = LmdbDataset(os.path.join(config['datapath'], f'MD22/{config["molecule"]}', config['dataset']))
        unit = 1
    
    elif config['molecule'] == 'chig':
        #unit eV  eV/A
        if config['dataset'] in ['radius3_broadcast']:
            dataset = LmdbDataset(os.path.join(config['datapath'], 'chignolin_dft_2'))
        unit = mol/kcal ##23.06
    else:
        raise NotImplementedError
    
    return dataset, unit

def main(world_size, config, args):
    print("config['master_port'] = {}, os.environ['MASTER_PORT'] = {}".format(config['master_port'],os.environ['MASTER_PORT']))
    if config["amlt"]:
        config["master_port"] = os.environ['MASTER_PORT']
        dist.init_process_group(backend='nccl', world_size=world_size, rank=config["local_rank"])
    else:
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:"+os.environ['MASTER_PORT'], world_size=world_size, rank=config["local_rank"])

        

    torch.cuda.set_device(config['local_rank']) # important! solved imbalanced memory at GPU0, ref: https://discuss.pytorch.org/t/how-to-balance-gpu-memories-in-ddp/93170
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    rank = config['local_rank']
    seed_everything(SEED)
    logging.info("get dataset")
    regress_forces = config['regress_forces']
    dataset, unit = get_dataset(config)
    if config['sample_size'] > len(dataset):
        raise ValueError("sample_size is larger than dataset size")




    sample_size_dic = {'buckyball_catcher':600,
            'double_walled_nanotube':800,
            'AT_AT': 3000,
            'AT_AT_CG_CG': 2000,
            'stachyose': 8000,
            'DHA': 8000,
            'Ac_Ala3_NHMe': 6000,
            'chig': 8000,
        }
    
    if config['sample_size'] == -1:
        if config['molecule'] in sample_size_dic:
            config['sample_size'] = sample_size_dic[config['molecule']]
        else:
            raise NotImplementedError
    
    if config['no_broadcast']:
            dataset.transforms = [convert_to_neighbor(r = config['otfcutoff']), reconstruct_group_with_threshold()]

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
                                                                            int(config['sample_size']* config['train_prop']),
                                                                            config['sample_size']-int(config['sample_size']* config['train_prop']), 
                                                                            len(dataset) - config['sample_size']],
                                                                            generator=torch.Generator().manual_seed(SEED))
    if config["local_rank"]==0:
        np.savez(config["model_path"]+"/data.npz", train_indices = train_set.indices, val_indices = val_set.indices,test_indices = test_set.indices)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, seed=SEED, drop_last=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False, drop_last=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False, drop_last=True)


    # Test for dataloader
    # Shuffling is mutually exclusive with sampler, so we disable shuffling if a sampler is provided.
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], collate_fn = collate_fn(unit = unit, with_force = regress_forces), shuffle=False,num_workers = config['num_workers'], sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=config['batch_size']//2, collate_fn = collate_fn(unit = unit, with_force = regress_forces), shuffle=False,num_workers = config['num_workers'], sampler=valid_sampler)
    test_loader = DataLoader(test_set, batch_size=config['batch_size']//2, collate_fn = collate_fn(unit = unit, with_force = regress_forces), shuffle=False,num_workers = config['num_workers'], sampler=test_sampler)
    mean, std, atomref, config = get_stats(config, unit, train_set)
    # if config["lr_std"]:
    config["mean"] = mean.item()
    config["std"] = std.item()
    # config["lr"] = config["lr"]/std.item()
    # args.lr = args.lr/std.item()
    config['atomref'] = atomref.cuda() if atomref is not None else None
        
    model = get_model(config,mean,std,regress_forces,atomref)
    properties = ['energy', 'forces'] if regress_forces else ['energy'] # diff_U0_group, group_energy

    hooks = []
    #cp_hook
    tf_writer = None
    csv_writer = None
    if config['local_rank'] == 0:
        tf_writer = SummaryWriter(log_dir=config['model_path'], flush_secs=1) 
        tf_writer.add_text('config', json.dumps(config, indent=4, sort_keys=True))
        csv_writer = Logger_Lin(config['model_path'], flush_secs=60) if rank == 0 else None
        if config['wandb']:
            wandb.login(key=config['api_key']) if config['api_key'] else None
            wandb.init(project="LightNP", dir = config['model_path'], group = config['group'], 
                       name = config['name'], 
                       tags = config['tags'], 
                       notes = config['notes'])
            wandb.config.update(config)
            config['model_path'] = wandb.run.dir

        
    
    model.tf_writer = tf_writer
    if config["loss"] == 'L2MAE':
        loss_fn = ltnp.train.loss.build_l2maeloss(energy_weight = config["energy_weight"],force_weight = config["force_weight"],with_forces=config["regress_forces"])
    else:
        loss_fn = ltnp.train.loss.build_mse_loss_with_forces(energy_weight = config["energy_weight"],force_weight = config["force_weight"],with_forces=config["regress_forces"])
    trainer = ltnp.train.ddp_trainer.DDPTrainer(model_path = config['model_path'],
                                    local_rank = rank,
                                    world_size=world_size,
                                    model = model,
                                    properties=properties,
                                    loss_fn = loss_fn,
                                    train_loader = train_loader,
                                    validation_loader = val_loader,
                                    test_loader=test_loader,
                                    test_interval=config["test_interval"],
                                    # metrics_name = ['MAE', 'R2', 'Spearman'], #,'MSE'],
                                    device = device,
                                    hooks = hooks, train_force = regress_forces, config = config,
                                    csv_writer = csv_writer,
                                    tf_writer = tf_writer, 
                                    use_wandb=config['wandb'], 
                                    early_stop=config['early_stop'], 
                                    early_stop_patience=config['early_stop_patience'], 
                                    ema_decay = config['ema_decay'], 
                                    args = args)
   
   
   # Restore checkpoint
    if config['restore_run']: # used for resuming training
        trainer.restore_checkpoint()
    elif config['restore_run_wandb']:
        trainer.restore_best_checkpoint_wandb(config['restore_run_wandb'])
    else:
        pass
    
    if config["test"]:
        # trainer.restore_checkpoint()
        try: # some previous runs may not have best checkpoint
            trainer.restore_best_checkpoint() if not config['restore_run_wandb'] else None
        except:
            print("No best checkpoint found, use the latest checkpoint instead.")
            pass
        trainer.test()
    else:
        trainer.train(config['max_epochs'])
        trainer.test()
    dist.destroy_process_group()

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
    
    
if __name__ == '__main__':
    config, args = parse_args()
    WORLD_SIZE = torch.cuda.device_count()
    config["batch_size"] = config["batch_size"]//WORLD_SIZE
    config["ngpus"] = WORLD_SIZE
    if config['amlt']:config = amlt_config(config)
    if not config['restore_run']: 
        timestamp = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y_%m_%d_%H_%M_%S_")    
        config['model_path'] = os.path.join(config['model_path'], f"{config['molecule']}_{config['dataset']}/{timestamp}_{config['model']}") 
        if config["local_rank"] ==0:
            if not os.path.exists(config['model_path']):
                os.makedirs(config['model_path'])
            prepare_dataset(config)
    else:
        print('Restore run: ', config["restore_run"])
        config['model_path'] = config["restore_run"]
    if(config["local_rank"] == -1):assert(False)
    main(WORLD_SIZE, config, args)