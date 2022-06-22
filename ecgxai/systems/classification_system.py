"""
Binary classificatios sytem

DB van de Pavert and MB Vessies
"""

import torch
import torch.nn as nn

from ecgxai.systems.base_system import BaseSystem


class ClassificationSystem(BaseSystem):
    def __init__(
        self,
        model: nn.Module,
        mode: str,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert mode in ['binary', 'multi_class', 'multi_label'], 'mode should be either "binary", "multi_class" or "multi_label"'  

        self.prob_func = nn.Softmax(dim=-1) if mode == 'multi_class' else torch.sigmoid
        
        self.save_hyperparameters()

        self.model = model

    def forward(self, x : torch.Tensor):
        y = self.model(x)
        return y

    @property
    def step_args(self):
        return ['y_hat', 'y_prob', 'label']

    def _step(self, batch):
        x = batch['waveform']
        y_hat = self.forward(x)
        return {
            'y_hat': y_hat,
            'y_prob': self.prob_func(y_hat),
            'label': batch['label']
        }