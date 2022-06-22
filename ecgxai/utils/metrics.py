import torch
from torchmetrics import Metric
from typing import Union, List
import ecgxai.utils.exceptions as exceptions


class BaseMetric(Metric):
    @property
    def batch_args(self):
        return []

    @property
    def input_args(self):
        raise NotImplementedError()


class TMW(BaseMetric):
    """ Torch Metric wrapper, wrapper class for default TorchMetrics"""
    def __init__(
        self,
        metric : object,
        input_args : List[str],
        batch_args : List[str] = [],
        int_args : List[str] = [],
        out_labels : List[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.inp_args = input_args
        self.b_args = batch_args
        self.int_args = int_args
        self.metric = metric
        self.out_labels = out_labels

    @property
    def input_args(self):
        return self.inp_args

    @property
    def batch_args(self):
        return self.b_args

    def update(self, args):
        m_args = {k: v for k, v in args.items() if k in self.b_args}
        m_args.update({k: v for k, v in args.items() if k in self.inp_args})

        # Cast args to int if specified
        for k in self.int_args:
            m_args[k] = m_args[k].int()

        self.metric(*m_args.values())

    def compute(self):
        out = self.metric.compute()

        if not self.out_labels:
            return out

        if len(self.out_labels) != len(out):
            raise exceptions.IncorrectNumberOfTMWLabelsException(self.metric, len(out.values()[0]), len(self.out_labels))

        res = {}
        for label, value in zip(self.out_labels, out):
            res[label] = value.detach().cpu()

        return res

