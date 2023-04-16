import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
# this function encapsulate the stuff (logger, callbacks, parameters)
# needed to perform the pytorch lightning training for each model
def train_model(data, model, experiment_name, patience, metric_to_monitor, mode, epochs):
    logger =  WandbLogger()
    logger.experiment.watch(model, log = None, log_freq = 100000)
    # To limit overfitting and avoid much more epochs than needed to complete
    # the training, we use the early stopping regulation technique which is
    # very powerful since it controls whether there is an improvement in the
    # model or not. If there is no improvement in the model performances for a 
    # given number of epochs (patience) on a certain metric (metric_to_monitor)
    # then the training stops.
    early_stop_callback = EarlyStopping(
        monitor=metric_to_monitor, mode=mode, min_delta=0.00,
        patience=patience, verbose=True)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3, monitor=metric_to_monitor, mode=mode, dirpath="models",
        filename=experiment_name +
        "-{epoch:02d}-{f1_score:.4f}", verbose=True)
    # the trainer collect all the useful informations so far for the training
    n_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        logger=logger, max_epochs=epochs, log_every_n_steps=1, gpus=n_gpus,
        callbacks=[early_stop_callback, checkpoint_callback],
        num_sanity_val_steps=0,
        # auto_lr_find = True,
        # auto_scale_batch_size=True,
        )
    trainer.fit(model, data)
    return trainer
