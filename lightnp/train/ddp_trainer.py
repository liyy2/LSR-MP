from collections import defaultdict
import gc
import os
import wandb
import yaml
import sys
from typing import Type
from unittest import result
from torch_ema import ExponentialMovingAverage
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from ..utils.dist_utils import reduce_cat,reduce_mean,reduce_sum
from .hooks.logging_metric import MeanAbsoluteError,MeanSquaredError, SpearmanCorr, R2
# from .loss_analysis import update_analysis
#from torch.utils.tensorboard import SummaryWriter

class DDPTrainer:
    r"""Class to train a model.

    This contains an internal training loop which takes care of validation and can be
    extended with custom functionality using hooks.

    Args:
         model_path (str): path to the model directory.
         model (torch.Module): model to be trained.
         loss_fn (callable): training loss function.
         optimizer (torch.optim.optimizer.Optimizer): training optimizer.
         train_loader (torch.utils.data.DataLoader): data loader for training set.
         validation_loader (torch.utils.data.DataLoader): data loader for validation set.
         keep_n_checkpoints (int, optional): number of saved checkpoints.
         checkpoint_interval (int, optional): intervals after which checkpoints is saved.
         hooks (list, optional): hooks to customize training process.
        ## ##loss_is_normalized (bool, optional): if True, the loss per data point will be
             reported. Otherwise, the accumulated loss is reported.

    """

    def __init__(
        self,
        model_path,
        model,
        loss_fn,
        properties,
        train_loader,
        validation_loader,
        test_loader=None,
        test_interval = 1000,
        metrics_name = ['MAE'], #,'MSE'
        device = torch.device("cuda"),
        local_rank=0, # which GPU am I
        world_size=1, # how many GPUs are there
        keep_n_checkpoints=1,
        checkpoint_interval=10,
        hooks=[],
        do_analysis = True,
        train_force = False,
        config  = None,
        csv_writer = None,
        tf_writer = None,
        use_wandb = False,
        early_stop = False,
        early_stop_patience = 500,
        ema_decay = 0.9999,
    ):
        self.nprocs = torch.cuda.device_count()
        self.model_path = model_path
        self.world_size = world_size
        self.device = device
        self.loss_fn = loss_fn
        self.local_rank = local_rank
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.test_interval = test_interval
        self.keep_n_checkpoints = keep_n_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.hooks = hooks
        self.config = config
        self.do_analysis = do_analysis        
        self.checkpoint_path = os.path.join(self.model_path, "checkpoints")
        self.best_model = os.path.join(self.model_path, "best_model")
        self.train_force = train_force
        self._stop = False
        self.wandb = use_wandb
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.early_stop_counter = 0
        if 'forces' in properties:
            self.scheduler_criteria = { 'val/MAE_energy':config["rho_criteria"],'val/MAE_forces': 1-config["rho_criteria"]}  ##rho_criteria default .1
        else:
            self.scheduler_criteria = {'val/MAE_energy':1}

        model.to(self.device)
        torch.cuda.empty_cache()
        if world_size==1:
            self._model = model
        else:
            self._model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            
        if self.local_rank == 0 and self.wandb:
            wandb.watch(self._model)
        self.gradient_clip = config['gradient_clip']
        self.fp16 = config['fp16']
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        self.epoch = 0
        self.best_epoch = 0
        self.step = 0
        self.best_loss = float("inf")
        self.best_logger = defaultdict(lambda: float("inf"))
        
        # trainable_params = filter(lambda p: p.requires_grad, self._model.parameters())
        # self.optimizer = optim.Adam(trainable_params, lr=0.001, weight_decay=5e-4) # TODO: Change this to a config option
        # self.optimizer  = torch.optim.AdamW([p for p in self._model.parameters() if p.requires_grad],
        #                         lr = config['learning_rate'], weight_decay=5e-4)
                                # amsgrad = config['AMSGrad'])
        self.optimizer  = torch.optim.AdamW([p for p in self._model.parameters() if p.requires_grad],
                                lr = config['learning_rate'], weight_decay=0)
        self.ema = ExponentialMovingAverage(self._model.parameters(), decay=ema_decay)

        
        self.properties = properties
        self.metrics_name = metrics_name
        self.metrics = []
        for p in self.properties:
            for m in self.metrics_name:
                if m == 'MAE':
                    self.metrics.append(MeanAbsoluteError(p,p))
                elif m == 'MSE':
                    self.metrics.append(MeanSquaredError(p,p))
                else:
                    self.metrics.append(eval(m)(p,p))

        
        # self.save_config(config,model_path)
        #### for CSV logger
        self.csv_str = []
        self.csv_writer = csv_writer
        ### for tensorboar Logger
        self.tf_writer = tf_writer
        for h in self.hooks:
            h.on_init_end(self)

    def save_config(self,config,model_path):
        if config is None:
            print("Config file is None, save failed.")
            return
        if config["local_rank"] == 0:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            with open(model_path+'/config.yaml', 'w') as f:
                yaml.dump(config,f) 
        
    def _check_is_parallel(self):
        return True if isinstance(self._model, 
                                    torch.nn.DataParallel) or isinstance(self._model,
                                    torch.nn.parallel.DistributedDataParallel) else False

    def _load_model_state_dict(self, state_dict):
        if self._check_is_parallel():
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)

    
    @property
    def state_dict(self):
        state_dict = {
            "epoch": self.epoch,
            "step": self.step,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "hooks": [h.state_dict for h in self.hooks],
            "early_stop_counter": self.early_stop_counter,
        }
        if self._check_is_parallel():
            state_dict["model"] = self._model.module.state_dict()
        else:
            state_dict["model"] = self._model.state_dict()
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        try:
            self._load_model_state_dict(state_dict["model"])
            self.epoch = state_dict["epoch"]
            self.best_epoch = state_dict["best_epoch"]
            self.early_stop_counter = state_dict["early_stop_counter"]
            self.step = state_dict["step"]
            self.best_loss = state_dict["best_loss"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            
            self.ema.load_state_dict(state_dict["ema"])
            for h, s in zip(self.hooks, state_dict["hooks"]):
                h.state_dict = s
        except KeyError:
            pass
            

    def store_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):os.makedirs(self.checkpoint_path)
        
        chkpt = os.path.join(
            self.checkpoint_path, "checkpoint-" + str(self.epoch) + ".pth.tar"
        )
        torch.save(self.state_dict, chkpt)
        # if self.config['wandb']:
        #     wandb.save(os.path.join(self.checkpoint_path, "checkpoint*"))
        chpts = [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pth.tar")]
        if len(chpts) > self.keep_n_checkpoints:
            chpt_epochs = [int(f.split(".")[0].split("-")[-1]) for f in chpts]
            sidx = np.argsort(chpt_epochs)
            for i in sidx[: -self.keep_n_checkpoints]:
                os.remove(os.path.join(self.checkpoint_path, chpts[i]))
    
    def restore_checkpoint(self, epoch=None, path = None):
        if epoch is None:
            epoch = max(
                [
                    int(f.split(".")[0].split("-")[-1])
                    for f in os.listdir(self.checkpoint_path if path==None else path)
                    if f.startswith("checkpoint")
                ]
            )

        chkpt = os.path.join(
            self.checkpoint_path if path ==None else path, "checkpoint-" + str(epoch) + ".pth.tar"
        )
        self.state_dict = torch.load(chkpt)
    
    def store_best_checkpoint(self):
        if not os.path.exists(self.best_model):os.makedirs(self.best_model)
        
        chkpt = os.path.join(
            self.best_model, "checkpoint-" + 'best' + ".pth.tar"
        )
        torch.save(self.state_dict, chkpt)

    def restore_best_checkpoint(self, path = None):
        self.state_dict = torch.load(os.path.join(
            self.best_model if path==None else path , "checkpoint-" + 'best' + ".pth.tar"
        ))
    
    def restore_best_checkpoint_wandb(self, run_id):
        weights = wandb.restore(os.path.join("best_model","checkpoint-best.pth.tar"), run_path=run_id)
        weights = torch.load(weights.name)
        self.state_dict = weights

    def forward_batch(self, batch_data):
        if isinstance(batch_data,dict):
            # Vanilla Dictionary object
            for key in batch_data:
                batch_data[key] = batch_data[key].to(self.device)
        else:
            # Exception for PyG data object
            for key in batch_data.keys:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)

        result = self._model(batch_data)
        loss = self.loss_fn(batch_data, result)
        
        # metrics statistics for batch data.
        for m in self.metrics:
            m.add_batch(result,batch_data)
        return result, loss
    
    def evaluate_epoch(self, loader, loader_name = "val_loader"):
        # loader name can be ["train_loader","val_loader","test_loader"]
        self._model.eval()
        with self.ema.average_parameters():
            eval_loss = 0.0
            if self.local_rank == 0:
                loop = tqdm(total=len(loader)) # * loader.batch_size)
            self.local_step = 0
            for eval_batch in loader:
                # append batch_size
                self.local_step += 1

                if not self.train_force:
                    with torch.no_grad():
                        result, loss = self.forward_batch(eval_batch)
                        eval_loss += loss.detach().cpu().numpy() 
                else:
                    result, loss = self.forward_batch(eval_batch)
                    eval_loss += loss.detach().cpu().numpy()
                    # self.optimizer.zero_grad()????
                if self.local_rank == 0 and not self.config['amlt']:
                    loop.set_description(f'Epoch [{self.epoch}], [{loader_name}]')
                    loop.set_postfix(loss=loss.detach().cpu().numpy())
                    loop.update(1)
            
            if self.world_size>1:
                # reduce loss from each childern.
                eval_loss = torch.Tensor([eval_loss]).to(self.device)
                dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
                eval_loss = eval_loss.cpu().numpy().item()
                eval_loss = eval_loss / self.world_size / len(loader)
                
                dist.barrier()
                
            if self.local_rank == 0:
                tqdm.write("epoch : {} , {} loss average is : {}".format(self.epoch, loader_name, eval_loss))
            return eval_loss
    
    def _get_params(self):
        return self._model.parameters() if not self._check_is_parallel() else self._model.module.parameters()
    
    def train_epoch(self, loader):
        
        self._model.train()
        train_loss = 0
        loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:loop = tqdm(total=len(loader)) # * loader.batch_size)
        self.local_step = 0
        for train_batch in loader:
            self.optimizer.zero_grad()
            
            if self.fp16:
                with torch.cuda.amp.autocast():
                    result, loss = self.forward_batch(train_batch)
            else:
                result, loss = self.forward_batch(train_batch) 
            if torch.any(torch.isnan(loss)):raise ValueError("loss is nan")

            # SYNC multiple process!!!!
            torch.distributed.barrier()

            # for the loss part, when backward, torch will aumatically to reduce them and then backward.
            if self.fp16:
                self.scaler.scale(loss).backward()
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self._get_params(), max_norm=1.0, norm_type=2.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self._get_params(), max_norm=1.0, norm_type=2.0)## loss is default reduced.
                self.optimizer.step()
                self.ema.update()
            
            train_loss += loss.detach().cpu().numpy() # TODO: write a loss accumulator to do this
            
            self.local_step += 1
            self.step += 1

            for h in self.hooks: # TODO: Separate hooks for rank 0 or not
                h.on_train_batch_end(self,train_batch,result,train_loss)
            if self._stop:break
            
            if self.local_rank == 0 and not self.config['amlt']:
                loop.set_description(f'Epoch [{self.epoch}]')
                loop.set_postfix(loss=loss.detach().cpu().numpy())
                loop.update(1)
                
        train_loss = torch.Tensor([train_loss]).to(self.device)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_loss = train_loss.cpu().numpy().item()
        train_loss = train_loss / self.world_size / len(loader)
        
        dist.barrier()
        if self.local_rank == 0:
            tqdm.write("epoch : {} train loss average is : {}".format(self.epoch, train_loss/len(loader)))
        

        return train_loss
        
                
    def train(self, n_epochs = 100000,loader = None):
        """Train the model for the given number of epochs on a specified device.

        Note: Depending on the `hooks`, training can stop earlier than `n_epochs`.

        """
        self._stop = False

        for h in self.hooks:
            h.on_train_begin(self)
        
        try:
            for _ in range(n_epochs):
                # increase number of epochs by 1
                self.epoch += 1

                if self._stop:
                    # decrease self.epoch if training is aborted on epoch begin
                    self.epoch -= 1
                    break
                
                gc.collect()
                torch.cuda.empty_cache()
                log_name=["epoch_idx","learnig_rate"]
                log_val = [self.epoch,self.optimizer.param_groups[0]["lr"]]
                
                #---------------train--------------------
                train_loss = self.train_epoch(self.train_loader)
                log_name.append("train/train_loss")
                log_val.append(train_loss)
                self.update_metrics2log(log_name,log_val,name_prefix = "train",clear_metrics=True)
                
                #---------------val--------------------
                val_loss = self.evaluate_epoch(self.validation_loader,loader_name="validation_loader")
                log_name.append("val/val_loss")
                log_val.append(val_loss)
                self.update_metrics2log(log_name,log_val,name_prefix = "val",clear_metrics=True)

                #self.epoch == 10 : check oom in test process.
                if self.epoch == 10 or self._stop or (self.epoch % self.test_interval == 0 and self.test_loader is not None):
                    self.restore_best_checkpoint() if self._stop else None #only restore when training is stopped
                    torch.cuda.empty_cache()
                    test_loss = self.evaluate_epoch(self.test_loader,loader_name="test_loader")
                    torch.cuda.empty_cache()
                    log_name.append("test/test_loss")
                    log_val.append(test_loss)
                    self.update_metrics2log(log_name,log_val,name_prefix = "test",clear_metrics=True)
                    if self._stop:
                        for i in range(len(log_name)):
                            if self.wandb:
                                wandb.run.summary['final/' + log_name[i]] = log_val[i]

                                      
                if (self.epoch % self.checkpoint_interval == 0) and (self.local_rank == 0):
                    self.store_checkpoint()
                
                scheduler_loss = 0
                for name in self.scheduler_criteria: # scheduler loss, normally we do not use MSE as scheduler loss
                    scheduler_loss += log_val[log_name.index(name)]*self.scheduler_criteria[name] # scheduler loss is the sum of loss with weight
                scheduler_loss = torch.Tensor([scheduler_loss]).to(self.device)
                scheduler_loss = reduce_mean(scheduler_loss.to(self.device),self.world_size).cpu().item()

                
                if scheduler_loss <= self.best_loss:
                    self.early_stop_counter = 0
                    if (self.local_rank == 0): self.store_best_checkpoint()
                    self.best_loss = scheduler_loss
                    self.best_epoch = self.epoch 
                else:
                    self.early_stop_counter += 1     
                # print(self.local_rank,"val_loss:",val_loss,"scheduler_loss",scheduler_loss,self.optimizer.param_groups[0]["lr"])
                # call hooks
                for h in self.hooks:h.on_validation_end(self, scheduler_loss)
                
                # only rank 0 will update scheduler & logger
                if self.local_rank==0:
                    if self.csv_writer is not None:
                        if self.epoch == 1:
                            self.csv_writer.info(",".join(log_name))
                        self.csv_writer.info(",".join([str("{:.8f}".format(p)) for p in log_val]))
                    ### for tensorboar Logger
                    for i in range(len(log_name)):  
                        # Tensorboard
                        if self.tf_writer is not None:
                            self.tf_writer.add_scalar(log_name[i], log_val[i], self.epoch)
                        # Wandb
                        if self.wandb:
                            wandb.log({log_name[i]:log_val[i]}, step=self.epoch)
                        
                        # some m is smaller the better, some m is larger the better
                        # Record the best m
                        if log_name[i].endswith("loss") or ('MAE' in log_name[i]) or ('MSE' in log_name[i]):     
                            if self.best_logger[log_name[i]] >= log_val[i]:
                                self.best_logger[log_name[i]] = log_val[i]
                                if self.wandb:
                                    wandb.run.summary['best/' + log_name[i]] = self.best_logger[log_name[i]]                            
                        else:
                            if self.best_logger[log_name[i]] <= log_val[i]:
                                self.best_logger[log_name[i]] = log_val[i]
                                if self.wandb:
                                    wandb.run.summary['best/' + log_name[i]] = self.best_logger[log_name[i]]                                
                    
                    
                # Early stopping  
                if self.early_stop_counter >= self.early_stop_patience:
                    if self.early_stop:
                        print("Early stopping")
                        self._stop = True 
                # sync stop between processes
                self._stop = torch.Tensor([self._stop]).to(self.device)
                dist.all_reduce(self._stop, op=dist.ReduceOp.SUM)
                self._stop = bool(self._stop.cpu().numpy().item())
                if self._stop:
                    break
                
            #
            # Training Ends
            #& store checkpoint
            if self.local_rank == 0:
                self.store_checkpoint()

        except Exception as e:
            raise e


    
    def test(self, loader = None):
        """Train the model for the given number of epochs on a specified device.

        Note: Depending on the `hooks`, training can stop earlier than `n_epochs`.

        """
        # TODO: check reload best model works or not
        # self._load_model_state_dict(torch.load(self.best_model))
        
        if loader is None:
            loader = self.test_loader

        test_loss = self.evaluate_epoch(loader,loader_name="test_loader")
        log_name = ["test_loss"]
        log_val = [test_loss]
        
        self.update_metrics2log(log_name,log_val,name_prefix = "test",clear_metrics=True)
        if self.local_rank==0:
            if self.csv_writer is not None:
                self.csv_writer.info(",".join([str("{:.8f}".format(p)) for p in log_val]))
            ### for tensorboar Logger
            if self.tf_writer is not None:
                for i in range(len(log_name)):
                    self.tf_writer.add_scalar(log_name[i], log_val[i], self.epoch)
                    if self.wandb:
                        wandb.log({log_name[i]:log_val[i]}, step=self.epoch)
                    if log_name[i].endswith("loss") or ('MAE' in log_name[i]) or ('MSE' in log_name[i]):
                        if self.best_logger[log_name[i]] > log_val[i]:
                            self.best_logger[log_name[i]] = log_val[i]
                            if self.wandb:
                                wandb.run.summary['best/' + log_name[i]] = self.best_logger[log_name[i]]

        return log_name,log_val
        
    def update_metrics2log(self,log_name,log_val,name_prefix='',clear_metrics = True):
        for m in self.metrics:
            val = m.aggregate()            
            if self.wandb:
                values = m.values.numpy()   #reduce_cat(m.values.to(self.device), self.world_size).cpu()
                if self.local_rank == 0:
                    wandb.log({f"histogram/{name_prefix}/{m.name}": wandb.Histogram(values)}, step=self.epoch)
            val = torch.Tensor([val]).to(self.device)
            val = reduce_mean(val.to(self.device),self.world_size).cpu()
            log_val.append(val.item())
            log_name.append("{}/{}".format(name_prefix,m.name))
            if clear_metrics:
                m.reset()