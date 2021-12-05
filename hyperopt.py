import os

os.environ["SLURM_JOB_NAME"] = "bash"
os.environ["WANDB_CONSOLE"] = "off"

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback, TuneReportCheckpointCallback)
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

from convmixer import ConvMixerModule
from data_utils import data_generator

pl.seed_everything(42)
cwd = os.getcwd()

BATCH_SIZE = 32
NW = 8
EPOCHS = 200

hyps = {
    "size": tune.choice([x for x in range(5, 25, 2)]),
    "num_blocks": tune.choice([x for x in range(5, 25, 2)]),
    "kernel_size": tune.choice([x for x in range(5, 25, 2)]),
    "patch_size": tune.choice([x for x in range(5, 25, 2)]),
    "num_classes": 5,
}


hyp_print = ""
for key, value in hyps.items():
    hyp_print += f"_{key}_{value}"


def train_loop(hyps):
    train_, val_= data_generator(
        img_dir=os.path.join(cwd, "train") , csv_path=os.path.join(cwd,"train.csv"), bs=BATCH_SIZE, nw=NW
    )
    main_model = ConvMixerModule(**hyps)
    trainer = pl.Trainer(
        # fast_dev_run=True,
        # benchmark=True,
        gpus=1,
        max_epochs=EPOCHS,
        callbacks=[
            TuneReportCallback(
                {"loss": "pt/val_loss", "mean_accuracy": "pt/val_accuracy"},
                on="validation_end",
            )
        ],
        progress_bar_refresh_rate=0,
    )
    trainer.fit(main_model, train_dataloaders=train_, val_dataloaders=val_)


if __name__ == "__main__":
    scheduler = ASHAScheduler(max_t=EPOCHS, grace_period=1, reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["size", "num_blocks", "kernel_size", "patch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"],
    )

    hyperopt = tune.run(
        tune.with_parameters(
            train_loop,
        ),
        resources_per_trial={
            "cpu": 2,
            "gpu": .5,
        },
        metric="loss",
        mode="min",
        config=hyps,
        num_samples=100,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="test1",
        # fail_fast=True,
    )
    print("result ", hyperopt.best_config)
