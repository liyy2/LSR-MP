from cgi import test
import os
from re import T
from lightnp.train.hooks import Hook
import torch
import yaml
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import torch.distributed as dist

class TestSetHook(Hook):
    '''
    Validate a model on a test set.
    '''
    def __init__(self, test_loader, val_metrics, writer = None, log_path = None, test_start = 1, test_interval=1):
        self.test_loader = test_loader
        self.test_start = test_start
        self.test_interval = test_interval
        self.val_metrics = val_metrics
        assert writer is not None or log_path is not None, 'writer or log_path should be provided'
        self.writer = writer if writer is not None else SummaryWriter(log_path)

    def on_epoch_end(self, trainer):
        if (trainer.epoch % self.test_interval == 0) and (trainer.epoch >= self.test_start):
        # validation
            trainer._model.eval()
            loop = tqdm(self.test_loader, leave = False, total=len(self.test_loader))
            for val_batch in loop:
                # append batch_size
                if not trainer.train_force:
                    with torch.no_grad():
                        result, loss = trainer.forward_batch(val_batch)
                else:
                    result, loss = trainer.forward_batch(val_batch)
                    trainer.optimizer.zero_grad()
                
                for metric in self.val_metrics:
                    metric.add_batch(val_batch, result)
                # this part should be implemented in Hook class
                # if trainer.do_analysis and trainer.epoch > 250:
                #     trainer.val_detail = update_analysis(trainer.val_detail, val_batch, diff)
                loop.set_description(f'Epoch [{trainer.epoch}]')
            
            
            for metric in self.val_metrics:
                m = metric.aggregate()
                if np.isscalar(m):
                    self.writer.add_scalar(
                    "test_metrics/%s" % metric.name, float(m), trainer.epoch)
                
                m = loop.write(f'TEST {metric.name}: {m}')
            

class Checkpoints_Hook(Hook):
    '''
    Loads the state dict of a model from a checkpoint.
    '''

    def __init__(self, restore = False, checkpoint_path = None):
        super(Checkpoints_Hook, self).__init__()
        self.restore = restore

    def on_train_begin(self, trainer):
        if self.restore:
            trainer.restore_checkpoint()
        else:
            if trainer.local_rank == 0:
                os.makedirs(trainer.checkpoint_path)
                trainer.store_checkpoint()
            dist.barrier()
            if trainer.local_rank != 0:
                trainer.restore_checkpoint()


