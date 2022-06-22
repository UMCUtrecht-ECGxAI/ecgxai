import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from ecgxai.utils.dataset import UniversalECGDataset
from ecgxai.utils.transforms import ApplyGain, ToTensor, To12Lead, Resample
from ecgxai.systems.VAE_system import GaussianVAE
from ecgxai.utils.loss import CombinedLoss, GaussianVAEReconLoss, KLDivergence

def run_trainer(params):
    pl.seed_everything(1234)

    # don't forget to put your own API key here, 
    # or select another platform for logging
    api_key = open("../neptune_token.txt", "r").read()
    neptune_logger=NeptuneLogger(
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

trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=128, shuffle=True)


loss = CombinedLoss(
    [
        GaussianVAEReconLoss(reduction='mean'),
        KLDivergence(reduction='mean', std_is_log=False)
    ],
    ['+'],
    [1, 64]
)


newVAE = GaussianVAE.load_from_checkpoint(
    checkpoint_path="/training/factorecg/final_vae_epoch_37.ckpt",
    loss=loss,
    lr=1e-3,
    std_is_log=False
)

trainer = pl.Trainer(
    max_epochs=20,
    log_every_n_steps=5,
    val_check_interval=0.25,
    gradient_clip_val=1,
    gpus=1,
    callbacks=[
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_last=True
        ),
    ]
)

trainer.fit(newVAE, trainLoader, testLoader)
