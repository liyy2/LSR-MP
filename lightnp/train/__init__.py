r"""
Classes to manage the training process.

lightnp.train.Trainer encapsulates the training loop. It also can automatically monitor the performance on the
validation set and contains logic for checkpointing. The training process can be customized using Hooks which derive
from lightnp.train.Hooks.

"""

from .ddp_trainer import DDPTrainer
from .loss import *
from .hooks import *
# from .metrics import *
# from .loss_analysis import update_analysis
