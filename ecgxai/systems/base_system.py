"""
Base system

MB Vessies
"""

import pytorch_lightning as pl
import torch.optim as optim
from typing import List
from torchmetrics import MetricCollection
import re
import torch


class BaseSystem(pl.LightningModule):
    def __init__(
        self, loss : object,
        lr: float,
        train_metrics : MetricCollection = None,
        val_metrics : MetricCollection = None,
        test_metrics : MetricCollection = None,
        log_filter : List[str] = None,
        log_filter_whitelist : bool = False
    ):
        super().__init__()

        self.__check_loss(loss)

        self.train_metrics = train_metrics.clone(prefix='train_') if train_metrics else None
        self.valid_metrics = val_metrics.clone(prefix='val_') if val_metrics else None
        self.test_metrics = test_metrics.clone(prefix='test_') if test_metrics else None

        self.__check_metrics_input(train_metrics, prefix="(train)")
        self.__check_metrics_input(val_metrics, prefix="(val)")
        self.__check_metrics_input(test_metrics, prefix="(test)")

        self.loss = loss
        self.lr = lr
        self.log_filter = log_filter
        self.log_filter_whitelist = log_filter_whitelist

    def __check_loss(self, loss):
        print(loss.input_args)
        for loss_arg in loss.input_args:
            if loss_arg not in self.step_args:
                raise KeyError(f"Loss of type '{type(loss).__name__}' requires input_arg {loss_arg} but it is not defined as a return argument of the current system")

    def __check_metrics_input(self, metrics, prefix=""):
        k_map = {
            '(train)': self.train_metrics,
            '(val)': self.valid_metrics,
            '(test)': self.test_metrics
        }

        if not k_map[prefix]:
            return

        for metric_name, metric in metrics.items():
            for metric_arg in metric.input_args:
                if metric_arg not in self.step_args:
                    raise KeyError(f"{prefix} Metric '{metric_name}' of type {type(metric).__name__} requires input_arg {metric_arg} but it is not defined as a returned argument of the current system")

    def __retrieve_loss_batch_args(self, batch, step_dict):
        for loss_arg in self.loss.batch_args:
            if loss_arg not in batch.keys():
                raise KeyError(f"Loss '{type(self.loss).__name__}' requires batch_arg {loss_arg} but it is not found in the given batch")
            step_dict[loss_arg] = batch[loss_arg]
        return step_dict

    def __retrieve_metrics_batch_args(self, metrics, batch, step_dict, prefix=""):
        if not metrics:
            return step_dict

        for metric_name, metric in metrics.items():
            for metric_arg in metric.batch_args:
                if metric_arg not in batch.keys():
                    raise KeyError(f"{prefix} Metric '{metric_name}' of type {type(metric).__name__} requires batch_arg {metric_arg} but it is not found in the given batch")
                step_dict[metric_arg] = batch[metric_arg]
        return step_dict

    def __apply_log_filter(self, log_dict):
        if self.log_filter:
            res = {}
            for filt in self.log_filter:
                for k, v in log_dict.items():
                    does_match = bool(re.search(filt, k))

                    if not (does_match ^ self.log_filter_whitelist):
                        res[k] = v
            return res
        else:
            return log_dict

    def configure_optimizers(self):
        """ TODO make configureable """

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": self.optimizer}

    @property
    def step_args(self):
        raise NotImplementedError("Make sure to define the step_args property in each system")

    def _step(self, batch):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        step_dict = self.__retrieve_loss_batch_args(batch, step_dict)

        loss, logs = self.loss(step_dict)
        logs = self.__apply_log_filter(logs)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        if self.train_metrics:
            step_dict = self.__retrieve_metrics_batch_args(self.train_metrics, batch, step_dict, prefix="(train)")
            metric_results = self.train_metrics(step_dict)
            self.log_dict(metric_results, on_step=True, on_epoch=True)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        step_dict = self.__retrieve_loss_batch_args(batch, step_dict)

        loss, logs = self.loss(step_dict)
        logs = self.__apply_log_filter(logs)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

        if self.valid_metrics:
            step_dict = self.__retrieve_metrics_batch_args(self.valid_metrics, batch, step_dict, prefix="(val)")
            metric_results = self.valid_metrics(step_dict)
            self.log_dict(metric_results, on_step=False, on_epoch=True)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        step_dict = self.__retrieve_loss_batch_args(batch, step_dict)
        
        loss, logs = self.loss(step_dict)
        logs = self.__apply_log_filter(logs)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})

        if self.test_metrics:
            step_dict = self.__retrieve_metrics_batch_args(self.test_metrics, batch, step_dict, prefix="(test)")
            metric_results = self.test_metrics(step_dict)
            step_dict.update(metric_results)
            self.log_dict(metric_results, on_step=False, on_epoch=True)

        self.log("test_loss", loss)

        return step_dict

    def predict_step(self, batch, batch_idx):
        return self._step(batch)
