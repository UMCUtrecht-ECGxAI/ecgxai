import sys
sys.path.append('../..')

import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection

from ecgxai.network.legacy.causal_cnn import VAE
from ecgxai.utils.dataset import UniversalECGDataset
from ecgxai.utils.transforms import ApplyGain, ToTensor, To12Lead, Resample
from ecgxai.systems.VAE_system import GaussianVAE
from ecgxai.utils.loss import CombinedLoss, GaussianVAEReconLoss, KLDivergence

def run_trainer(params):
    pl.seed_everything(1234)

    # don't forget to put your own API key here, 
    # or select another platform for logging
    api_key = open("../neptune_token.txt", "r").read()
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project=""
    )

trainDataset = UniversalECGDataset(
    dataset_type="umcu",
    waveform_dir="/raw_data/umcu_median",
    dataset=trainset_df,
    transform=Compose([
        Resample(500),
        ApplyGain(),
        To12Lead(),
        ToTensor()
    ]),
)

testDataset = UniversalECGDataset(
    dataset_type="umcu",
    waveform_dir="/raw_data/umcu_median",
    dataset=testset_df,
    transform=Compose([
        Resample(500),
        ApplyGain(),
        To12Lead(),
        ToTensor()
    ]),
)

api_key = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MmQ0ODMxOS03Zjk0LTRkMGEtYjUyOC1hNjAwNTA5OTUxOTgifQ=='
neptune_logger = NeptuneLogger(
    project="UMCUtrecht/VAE",
    api_key=api_key,
    tags=['UMCU finetune FactorECG', 'beta=64', 'working loss', 'weight qrs 10 p 3']
)

trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=128, shuffle=True)


loss = CombinedLoss(
    [
        ChunkedGaussianLoss(chunk_weights=[1,3,1,10,1,1], reduction={'batch': 'mean', 'channel': 'sum', 'inter_chunk': 'sum', 'intra_chunk': 'sum'}, clip_loss=1e3), 
        #GaussianVAEReconLoss(reduction='mean'),
        KLDivergence(reduction='mean', std_is_log=False)
    ],
    ['+'],
    [1, 128]
)


metrics = MetricCollection(
    {
        'pw_diff_all': PwaveHeightDiff('all', True, 12),
        #**{f'pw_diff_{i}': PwaveHeightDiff(i, True, 12) for i in range(12)},
        'qw_diff_all': QwaveHeightDiff('all', True, 12),
        #**{f'qw_diff_{i}': QwaveHeightDiff(i, True, 12) for i in range(12)},
        'rw_diff_all': RwaveHeightDiff('all', True, 12),
        #**{f'rw_diff_{i}': RwaveHeightDiff(i, True, 12) for i in range(12)},
        'sw_diff_all': SwaveHeightDiff('all', True, 12),
        #**{f'sw_diff_{i}': SwaveHeightDiff(i, True, 12) for i in range(12)},
        'tw_diff_all': TwaveHeightDiff('all', True, 12),
        #**{f'tw_diff_{i}': TwaveHeightDiff(i, True, 12) for i in range(12)},
    }
)

newVAE = GaussianVAE.load_from_checkpoint(
    checkpoint_path="/training/factorecg/final_vae_epoch_37.ckpt",
    loss=loss,
    lr=1e-3,
    train_metrics=metrics,
    val_metrics=metrics,
    std_is_log=False
)

trainer = pl.Trainer(
    max_epochs=20,
    logger=neptune_logger,
    log_every_n_steps=5,
    val_check_interval=0.25,
    gradient_clip_val=1,
    gpus=1,
    callbacks=[
        ReconstructionPlottingCallback(
            dataset=testDataset,
            indices=list(range(10))
        ),
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_last=True
        ),
    ]
)

trainer.fit(newVAE, trainLoader, testLoader)
