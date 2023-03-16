class Hook:
    """Base class for hooks."""

    @property
    def state_dict(self):
        return {}

    @state_dict.setter
    def state_dict(self, state_dict):
        pass
    
    def on_init_end(self, trainer):
        pass

    def on_train_begin(self, trainer):
        pass

    def on_train_ends(self, trainer):
        pass

    def on_train_failed(self, trainer):
        pass

    def on_epoch_begin(self, trainer):
        """Log at the beginning of train epoch.

        Args:
            trainer (Trainer): instance of lightnp.train.trainer.Trainer class.

        """
        pass

    def on_epoch_end(self, trainer):
        pass
    


    def on_train_batch_end(self, trainer, train_batch, result, loss):
        pass



    def on_validation_end(self, trainer, val_loss):
        pass
    


