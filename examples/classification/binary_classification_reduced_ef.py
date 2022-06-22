import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics import MetricCollection, AUROC, AveragePrecision

import yaml
import pandas as pd
import os

from ecgxai.utils.dataset import UniversalECGDataset
from ecgxai.utils.transforms import ApplyGain, ToTensor, To12Lead, Resample
from ecgxai.utils.metrics import TMW
from ecgxai.systems.classification_system import ClassificationSystem
from ecgxai.network.causalcnn.encoder import CausalCNNVEncoder
from ecgxai.utils.loss import BinaryFocalLoss


def run_trainer(params):
    pl.seed_everything(1234)

    # don't forget to update this for your own logger
    api_key = open("neptune_token.txt", "r").read()
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project=params['training']['project_name'],
        tags=params['training']['tags'],
        source_files=['*.py', '*.json', '*.yaml', '../../ecgxai/**/*.py']
    )
    neptune_logger.experiment["model/hyper-parameters"] = params

    # define transforms
    transform = transforms.Compose([Resample(500), ApplyGain(), ToTensor(), To12Lead()])

    # define datasets
    traindf = pd.read_csv(params["paths"]["training_labels"])
    trainset = UniversalECGDataset(
        'umcu',
        params["paths"]["raw_data"],
        traindf,
        transform=transform,
        labels=params["training"]["label_names"]
    )
    train_loader = DataLoader(
        trainset,
        batch_size=params['training']['batch_size'],
        num_workers=8,
        shuffle=True
    )

    valdf = pd.read_csv(params["paths"]["validation_labels"])
    valset = UniversalECGDataset(
        'umcu',
        params["paths"]["raw_data"],
        valdf,
        transform=transform,
        labels=params["training"]["label_names"]
    )
    val_loader = DataLoader(
        valset,
        batch_size=params['training']['batch_size'],
        num_workers=8
    )

    if params['training']['pretrain']:
        model = ClassificationSystem.load_from_checkpoint(params['paths']['pretrain_checkpoint'])
    else:
        encoder = CausalCNNVEncoder(**params['network'])

        metrics = MetricCollection({
            'AUROC': TMW(AUROC(), ['y_hat', 'label'], int_args=['label']),
            'AUPRC': TMW(AveragePrecision(), ['y_hat', 'label'])
        })

        model = ClassificationSystem(
            lr=params['training']['learning_rate'],
            model=encoder,
            train_metrics=metrics,
            val_metrics=metrics,
            loss=BinaryFocalLoss(pos_weight=torch.tensor(params['training']['loss_weights'])),
            mode="binary"
        )

    trainer = pl.Trainer(
        max_epochs=params['training']['epochs'],
        logger=neptune_logger,
        log_every_n_steps=5,
        gpus=1,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_last=True,
                dirpath=os.path.join(params['paths']['checkpoints'], neptune_logger.version),
                filename='epoch={epoch}-step={step}-loss={val_loss:.2f}'
            ),
        ]
    )
    trainer.fit(model, train_loader, val_loader)


with open('binary_classification_reduced_ef.yaml', "r") as stream:
    params = yaml.safe_load(stream)
    
output = run_trainer(params)
