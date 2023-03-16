import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from lightnp.train.hooks import Hook

class LRWarmupHook(Hook):
    '''
    Warm up the learning rate.
    '''
    def __init__(self, warmup_start_lr, warmup_end_lr, warmup_steps):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.warmup_steps = warmup_steps
        self.warmup_factor = (warmup_end_lr - warmup_start_lr) / warmup_steps
        self.warmup_lr = warmup_start_lr

    @property
    def state_dict(self):
        return {'warmup_lr': self.warmup_lr}
    
    @state_dict.setter
    def state_dict(self, state_dict):
        self.warmup_lr = state_dict['warmup_lr']
    
    def on_train_begin(self, trainer):
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = self.warmup_lr

    def on_train_batch_end(self, trainer, *args):
        if trainer.step < self.warmup_steps:
            self.warmup_lr += self.warmup_factor
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = self.warmup_lr

class EarlyStoppingHook(Hook):
    r"""Hook to stop training if validation loss fails to improve.

    Args:
        patience (int): number of epochs which can pass without improvement
            of validation loss before training ends.
        threshold_ratio (float, optional): counter increases if
            curr_val_loss > (1-threshold_ratio) * best_loss

    """

    def __init__(self, patience, threshold_ratio=0.0001):
        self.best_loss = float("Inf")
        self.counter = 0
        self.threshold_ratio = threshold_ratio
        self.patience = patience

    @property
    def state_dict(self):
        return {"counter": self.counter, "best_loss": self.best_loss}

    @state_dict.setter
    def state_dict(self, state_dict):
        self.counter = state_dict["counter"]
        self.best_loss = state_dict["best_loss"]

    def on_validation_end(self, trainer, val_loss):
        if val_loss > (1 - self.threshold_ratio) * self.best_loss:
            self.counter += 1
        else:
            self.best_loss = val_loss
            self.counter = 0

        if self.counter > self.patience:
            trainer._stop = True




class MaxStepHook(Hook):
    """Hook to stop training when a maximum number of steps is reached.

    Args:
        max_steps (int): maximum number of steps.

    """

    def __init__(self, max_steps):
        self.max_steps = max_steps

    def on_train_batch_end(self, trainer, train_batch):
        """Log at the ending of train batch.

        Args:
            trainer (Trainer): instance of lightnp.train.trainer.Trainer class.
            train_batch (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        """
        # stop training if max_steps is reached
        if trainer.step > self.max_steps:
            trainer._stop = True


class LRScheduleHook(Hook):
    """Base class for learning rate scheduling hooks.

    This class provides a thin wrapper around torch.optim.lr_schedule._LRScheduler.

    Args:
        scheduler (torch.optim.lr_schedule._LRScheduler): scheduler.
        each_step (bool, optional): if set to True scheduler.step() is called every
            step, otherwise every epoch.

    """

    def __init__(self, scheduler, each_step=False):
        self.scheduler = scheduler
        self.each_step = each_step

    @property
    def state_dict(self):
        return {"scheduler": self.scheduler.state_dict()}

    @state_dict.setter
    def state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def on_train_begin(self, trainer):
        self.scheduler.last_epoch = trainer.epoch - 1

    def on_train_batch_end(self, trainer, train_batch):
        """Log at the beginning of train batch.

        Args:
            trainer (Trainer): instance of lightnp.train.trainer.Trainer class.
            train_batch (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        """
        if self.each_step:
            self.scheduler.step()
        else:
            if self.epoch!=trainer.epoch:
                self.epoch = trainer.epoch
                self.scheduler.step()



class ReduceLROnPlateauHook(Hook):
    r"""
    !!!per epoch call!!!
    Hook for reduce plateau learning rate scheduling.

    This class provides a thin wrapper around
    torch.optim.lr_schedule.ReduceLROnPlateau. It takes the parameters
    of ReduceLROnPlateau as arguments and creates a scheduler from it whose
    step() function will be called every epoch.

    Args:
        patience (int, optional): number of epochs with no improvement after which
            learning rate will be reduced. For example, if `patience = 2`, then we
            will ignore the first 2 epochs with no improvement, and will only
            decrease the LR after the 3rd epoch if the loss still hasn't improved then.
        factor (float, optional): factor by which the learning rate will be reduced.
            new_lr = lr * factor.
        min_lr (float or list, optional): scalar or list of scalars. A lower bound on
            the learning rate of all param groups or each group respectively.
        window_length (int, optional): window over which the accumulated loss will
            be averaged.
        stop_after_min (bool, optional): if enabled stops after minimal learning rate
            is reached.

    """

    def __init__(
        self,
        patience=25,
        factor=0.5,
        min_lr=1e-6,
        window_length=1,
        stop_after_min=False,
        start_steps = 0.
    ):  
        self.start_steps = start_steps
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.scheduler = None
        self.window_length = window_length
        self.stop_after_min = stop_after_min
        self.window = []

    def on_init_end(self, trainer):
        if self.scheduler is None:
            self.scheduler = ReduceLROnPlateau(
                trainer.optimizer, patience=self.patience, factor=self.factor, min_lr=self.min_lr
            )
    
    @property
    def state_dict(self):
        assert self.scheduler is not None, "No scheduler found. Please set one first."
        return {"scheduler": self.scheduler.state_dict()}

    @state_dict.setter
    def state_dict(self, state_dict):
        if self.scheduler is None:
            self.on_train_begin(None)
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def on_validation_end(self, trainer, val_loss):
        if trainer.step > self.start_steps:
            self.window.append(val_loss)
            if len(self.window) > self.window_length:
                self.window.pop(0)
            accum_loss = np.mean(self.window)

            self.scheduler.step(accum_loss)

            if self.stop_after_min:
                for i, param_group in enumerate(self.scheduler.optimizer.param_groups):
                    old_lr = float(param_group["lr"])
                    if old_lr <= self.scheduler.min_lrs[i]:
                        trainer._stop = True


class ExponentialDecayHook(Hook):
    """Hook for reduce plateau learning rate scheduling.

    This class provides a thin wrapper around torch.optim.lr_schedule.StepLR.
    It takes the parameters of StepLR as arguments and creates a scheduler
    from it whose step() function will be called every
    step.

    Args:
        gamma (float): Factor by which the learning rate will be
            reduced. new_lr = lr * gamma
        step_size (int): Period of learning rate decay.

    """

    def __init__(self, optimizer, gamma=0.96, step_size=100000):
        self.scheduler = StepLR(optimizer, step_size, gamma)

    def on_train_batch_end(self, trainer, *args):
        self.scheduler.step()


