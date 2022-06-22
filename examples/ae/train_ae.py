import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.transforms import Compose

import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from ecgxai.utils.dataset import UniversalECGDataset
from ecgxai.network.AE_encoder_decoder import AEDoubleResidualEncoder, DoubleResidualDecoder

from ecgxai.utils.loss import TW
from ecgxai.utils.transforms import ApplyGain, ToTensor, Resample
from ecgxai.systems.AE_system import AE


# Please note that this configuration requires median beat data which is not currently publicly available
params = {
    "median_data_dir": "/median",
    "one_mili_csv": "header_info.csv",
}

pl.seed_everything(0)

# Set Data transforms
train_transform = Compose([ToTensor(), ApplyGain(), Resample(500)])
test_transform = Compose([ToTensor(), ApplyGain(), Resample(500)])

# Get data and split in train and test
full_set_df = pd.read_csv(params['one_mili_csv'])


trainset_df, testset_df = train_test_split(full_set_df, test_size=0.1)

trainset = UniversalECGDataset(
    'umcu',
    params['median_data_dir'],
    trainset_df,
    transform=train_transform,
)

testset = UniversalECGDataset(
    'umcu',
    params['median_data_dir'],
    testset_df,
    transform=test_transform,
)

batchsize = 64
trainLoader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=12)
testLoader = DataLoader(testset, batch_size=batchsize, shuffle=True, num_workers=8)

# Remember to properly configure your logger here
# You can change the neptune logger to any logger supported by pytorch lighting
neptune_logger = NeptuneLogger(
    api_key=open("../neptune_token.txt", "r").read(),
    project="%project%"
)

lr = 0.001

latent_dim = 32
in_sample_dim = 600
out_sample_dim = 600
sample_channels = 8
out_sample_channels = 8

enc_pre_block_1_out_channels = 16
enc_pre_block_1_kernel_size = 3
enc_pre_block_1_bn = True
enc_pre_block_1_dropout_rate = 0.0
enc_pre_block_1_act_func = None

enc_pre_block_2_out_channels = 32
enc_pre_block_2_kernel_size = 3
enc_pre_block2_act_func = None
enc_pre_block_2_dropout_rate = 0.1
enc_pre_block2_bn = True

enc_cnn_num_layers = 12
enc_cnn_kernel_size = 16
enc_cnn_dropout_rate = 0.1
enc_cnn_sub_sample_every = 4
enc_cnn_double_channel_every = 4
enc_cnn_act_func = nn.ReLU()
enc_cnn_bn = True

dec_post_block_1_out_channels = 8
dec_post_block_1_kernel_size = 3
dec_post_block_1_bn = True
dec_post_block_1_dropout_rate = 0.1
dec_post_block_1_act_func = None

dec_post_block_2_out_channels = 8
dec_post_block_2_kernel_size = 17
dec_post_block2_act_func = None
dec_post_block_2_dropout_rate = 0.1
dec_post_block2_bn = True

dec_cnn_num_layers = 3
dec_cnn_kernel_size = 3
dec_cnn_dropout_rate = 0.1
dec_cnn_sub_sample_every = 4
dec_cnn_double_channel_every = 4
dec_cnn_act_func = nn.ReLU()
dec_cnn_bn = True

hyperparameters = dict(
    lr=lr,
    latent_dim=latent_dim,
    in_sample_dim=in_sample_dim,
    out_sample_dim=out_sample_dim,
    sample_channels=sample_channels,
    out_sample_channels=out_sample_channels,

    pre_block_1_out_channels=enc_pre_block_1_out_channels,
    pre_block_1_kernel_size=enc_pre_block_1_kernel_size,
    pre_block_1_bn=enc_pre_block_1_bn,
    pre_block_1_dropout_rate=enc_pre_block_1_dropout_rate,
    pre_block_1_act_func=enc_pre_block_1_act_func,

    pre_block_2_out_channels=enc_pre_block_2_out_channels,
    pre_block_2_kernel_size=enc_pre_block_2_kernel_size,
    pre_block2_act_func=enc_pre_block2_act_func,
    pre_block_2_dropout_rate=enc_pre_block_2_dropout_rate,
    pre_block2_bn=enc_pre_block2_bn,

    enc_cnn_num_layers=enc_cnn_num_layers,
    enc_cnn_kernel_size=enc_cnn_kernel_size,
    enc_cnn_dropout_rate=enc_cnn_dropout_rate,
    enc_cnn_sub_sample_every=enc_cnn_sub_sample_every,
    enc_cnn_double_channel_every=enc_cnn_double_channel_every,
    enc_cnn_act_func=enc_cnn_act_func,
    enc_cnn_bn=enc_cnn_bn
)

print(hyperparameters)

encoder = AEDoubleResidualEncoder(
    latent_dim=latent_dim,
    in_sample_dim=in_sample_dim,
    out_sample_dim=out_sample_dim,
    sample_channels=sample_channels,
    out_sample_channels=out_sample_channels,

    pre_block_1_out_channels=enc_pre_block_1_out_channels,
    pre_block_1_kernel_size=enc_pre_block_1_kernel_size,
    pre_block_1_bn=enc_pre_block_1_bn,
    pre_block_1_dropout_rate=enc_pre_block_1_dropout_rate,
    pre_block_1_act_funct=enc_pre_block_1_act_func,

    pre_block_2_out_channels=enc_pre_block_2_out_channels,
    pre_block_2_kernel_size=enc_pre_block_2_kernel_size,
    pre_block2_act_funct=enc_pre_block2_act_func,
    pre_block_2_dropout_rate=enc_pre_block_2_dropout_rate,
    pre_block2_bn=enc_pre_block2_bn,

    cnn_num_layers=enc_cnn_num_layers,
    cnn_kernel_size=enc_cnn_kernel_size,
    cnn_dropout_rate=enc_cnn_dropout_rate,
    cnn_sub_sample_every=enc_cnn_sub_sample_every,
    cnn_double_channel_every=enc_cnn_double_channel_every,
    cnn_act_func=enc_cnn_act_func,
    cnn_bn=enc_cnn_bn
)

decoder = DoubleResidualDecoder(
    latent_dim=latent_dim,
    in_sample_dim=in_sample_dim,
    out_sample_dim=out_sample_dim,
    sample_channels=sample_channels,
    out_sample_channels=out_sample_channels,

    post_block_1_in_channels=dec_post_block_1_out_channels,
    post_block_1_kernel_size=dec_post_block_1_kernel_size,
    post_block_1_bn=dec_post_block_1_bn,
    post_block_1_dropout_rate=dec_post_block_1_dropout_rate,
    post_block_1_act_func=dec_post_block_1_act_func,

    post_block_2_in_channels=dec_post_block_2_out_channels,
    post_block_2_kernel_size=dec_post_block_2_kernel_size,
    post_block_2_act_func=dec_post_block2_act_func,
    post_block_2_dropout_rate=dec_post_block_2_dropout_rate,
    post_block_2_bn=dec_post_block2_bn,

    cnn_num_layers=dec_cnn_num_layers,
    cnn_kernel_size=dec_cnn_kernel_size,
    cnn_dropout_rate=dec_cnn_dropout_rate,
    cnn_sub_sample_every=dec_cnn_sub_sample_every,
    cnn_double_channel_every=dec_cnn_double_channel_every,
    cnn_act_func=dec_cnn_act_func,
    cnn_bn=dec_cnn_bn
)


model = AE(encoder, decoder, lr=lr, loss=TW(torch.nn.MSELoss(reduction='mean'), input_args=['x', 'reconstruction']))

trainer = pl.Trainer(
    logger=neptune_logger,
    checkpoint_callback=False,
    gradient_clip_val=10,
    max_epochs=50,
    gpus=1 if torch.cuda.is_available() else None,
    callbacks=[
        ModelCheckpoint(
            save_last=True
        ),
    ],
)

trainer.logger.log_hyperparams(hyperparameters)
trainer.fit(model, trainLoader, testLoader)
