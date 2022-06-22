"""
Custom loss function for ECGs.

MB Vessies and RR van de Leur
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Union, List
import operator
from operator import itemgetter
import numpy as np
from ecgxai.utils.exceptions import SumMeanReductionShapeMismatchException


class BaseLoss(pl.LightningModule):
    """
    Base class, used for inheritance later on.

    Attributes:
        reduction: String defining how to reduced the computed loss per sample 
        on the batch level. Options are: 'mean', 'sum' and 'sum_mean'. This last 
        option only works if the input is three dimension, and uses a sum for the 
        second and third dimension, while the a mean over the batch dimension is taken.
    """

    def __init__(self, reduction : str = 'mean'):
        super().__init__()
        assert reduction in ['sum', 'mean', 'sum_mean'], "Please provide one of the following for the reduction function: 'mean', 'sum' of 'sum_mean'."

        self.reduction = reduction
        self.only_log_num = True

    @property
    def input_args(self):
        raise NotImplementedError("Make sure to define the input_args property in each loss class")

    @property
    def batch_args(self):
        return []

    def apply_reduction(self, loss, skip_if_none : bool = False):
        """
        Reduces the computed loss per sample on the batch level.

        Args:
            loss: The computed loss per sample.
            skip_if_none: TODO
        """

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum_mean':
            if len(loss.shape) != 3:
                raise SumMeanReductionShapeMismatchException(dim=len(loss.shape))
            return loss.sum(dim=(1, 2)).mean()
        # TODO, must be a better way to handle this
        # skip_if_none is generally controlled by the only_log_num attribute to make sure no N-dimensional tensors are logged
        return loss if not skip_if_none else None


class TW(BaseLoss):
    """ 
    This wrapper can convert any standard Pytorch loss into a class
    that is usable in the ECGx.AI package.

    Attributes:
        loss: A PyTorch loss object that needs to be converted.
        input_args: A list of strings containing the input arguments
            that are needed to compute the loss. This differs per 
            loss function, and could be `x` and `y` for the 
            CrossEntropyLoss or `x` and `reconstruction` for the MSE in an auto-encoder.
        batch_args: A list of strings containing the arguments that are
            needed from the the batch dictionary, for example P on- and offsets
            to calculate locations specific losses.
    """
    def __init__(
        self,
        loss : object,
        input_args : List[str],
        batch_args : List[str] = [],
        long_args : List[str] = [],
        arg_max_args : List[str] = [],
        **kwargs
    ):
        super().__init__(**kwargs)

        self.inp_args = input_args
        self.b_args = batch_args
        self.long_args = long_args
        self.arg_max_args = arg_max_args
        self.loss = loss

    @property
    def input_args(self):
        return self.inp_args

    @property
    def batch_args(self):
        return self.b_args

    def __call__(self, args):
        l_args = {k: v for k, v in args.items() if k in self.b_args}
        l_args.update({k: v for k, v in args.items() if k in self.inp_args})

        # Take argmax of last dim of args if specified
        for k in self.arg_max_args:
            l_args[k] = l_args[k].argmax(dim=-1)

        # Cast args to int if specified
        for k in self.long_args:
            l_args[k] = l_args[k].long()

        loss = self.loss(*l_args.values())
        return self.apply_reduction(loss), {}


class CombinedLoss(BaseLoss):
    """ 
    Combines mulitple losses into a single loss class.
    
    Attributes:
        losses: List of loss functions (defined as objects) to combine
        operations: List of operations, or single operation to used to combine the loss functions
            defined as strings. Currently supported are "+", "-", "*" and "/". 
            If a list is supplied it should be the same lenght as losses.
        combine_weigths: Weights to multiply each loss by before applying opperations (optional).
    """
    def __init__(
        self,
        losses : List[object],
        operations : Union[List[str], str] = '+',
        combine_weigths : List[float] = None,
        **kwargs
    ):
        """
        Initializes the combined loss.
        """
        super().__init__(**kwargs)

        self.losses = torch.nn.Sequential(*losses)

        # Dict so that we can change the strings to callable functions
        opp_dict = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv
        }
                
        if isinstance(operations, list):
            assert len(operations) == len(self.losses) - 1, "Operations should be a string or a list of strings of length len(losses) - 1"
        else:
            # If only one operator is given, apply same operation to each loss
            operations = [operations for _ in range(len(self.losses) - 1)]

        # convert operation strings to callable functions
        try:
            self.operations = [opp_dict[opp] for opp in operations] 
        except KeyError:
            KeyError('Undfined operation, supported operations are are "+", "-", "*" and "/".')

        # Add none as first operation, as the zip function later will not work without one
        self.operations.insert(0, None)

        # check length of the given weigths, otherwise just set all the weights to one
        if combine_weigths:
            assert len(combine_weigths) == len(self.losses), "Losses and weights should be of equal length"
            self.combine_weigths = combine_weigths
        else:
            self.combine_weigths = torch.ones(len(self.losses))

        # Allow sub-losses to return 'none' reduced logs (loss and log reduction are controlled by the same parameter), logs are then reduced by combined loss class
        for loss in self.losses:
            loss.only_log_num = False

    @property
    def batch_args(self):
        return {arg for loss in self.losses for arg in loss.batch_args}

    @property
    def input_args(self):
        return {arg for loss in self.losses for arg in loss.input_args}

    def __call__(self, args : dict):
        """
        Internal function to compute the combined loss.

        Args:
            args: Dictionary with possible loss arguments indexed by loss parameter name
        """
        logs = {}

        for idx, (loss_obj, operation, weight) in enumerate(zip(self.losses, self.operations, self.combine_weigths)):
            # Call the loss function with the corresponding arguments and multiply it by correspoinding weight
            try:
                c_loss, c_logs = loss_obj(args)
                weighted_c_loss = weight * c_loss

                logs[f'{type(loss_obj).__name__}'] = self.apply_reduction(c_loss, skip_if_none=self.only_log_num)
                logs[f'{type(loss_obj).__name__}_weighted'] = self.apply_reduction(weighted_c_loss, skip_if_none=self.only_log_num)

                logs.update({f'{type(loss_obj).__name__}_{k}': self.apply_reduction(v, skip_if_none=self.only_log_num) for k, v in c_logs.items()})

            except KeyError as e:
                raise KeyError(f"Failed to fetch loss arguments {e.args} for {type(loss_obj).__name__}, available arguments are {args.keys()}.")

            # Combine previous loss and current loss using set operation if applicable (which is not true for first loss)
            loss = weighted_c_loss if idx == 0 else operation(loss, weighted_c_loss)

        return self.apply_reduction(loss), logs


class KLDivergence(BaseLoss):
    """
    Computes the Kuback-Leibner divergence for training of variational auto-encoders.
    
    Attributes:
        std_is_log: If the standard deviation is provided to the loss function as ln(SD).
        mu_min_clip: Minimum value to clip the mean on.
        mu_max_clip: Maximum value to clip the mean on.
        std_min_clip: Minimum value to clip the standard deviation on.
        std_max_clip: Maximum value to clip the standard deviation on.
    """
    def __init__(
        self,
        std_is_log : bool = True,
        mu_min_clip : float = -100,
        mu_max_clip : float = 100,
        std_min_clip : float = 1e-6,
        std_max_clip : float = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.std_is_log = std_is_log
        self.mu_min_clip = mu_min_clip
        self.mu_max_clip = mu_max_clip
        self.std_min_clip = std_min_clip
        self.std_max_clip = std_max_clip

    @property
    def input_args(self):
        return ['mu', 'std']

    def __call__(self, args):
        mu, std = args['mu'], args['std']
        logs = {}
        if self.std_is_log:
            std = torch.exp(std)

        # clip std and mu
        mu = mu.clip(min=self.mu_min_clip, max=self.mu_max_clip)
        std = std.clip(min=self.std_min_clip , max=self.std_max_clip)

        loss = -0.5 * torch.sum(1 + std.pow(2).log() - mu.pow(2) - std.pow(2), dim=1)

        return self.apply_reduction(loss), logs


class GaussianVAEReconLoss(BaseLoss):
    """
    Computes the Gaussian Reconstruction Loss as proposed by Van de Leur & Bos et al (2022).

    For this loss to work, your model should return both a mean and standard deviation for
    every timepoint in the ECG.

    Attributes:
        recon_nan_clip_val: Value to replace NaN values in the reconstruction mean or SD with.
    """

    def __init__(
        self,
        recon_nan_clip_val : float = 50,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.recon_nan_clip_val = recon_nan_clip_val

    @property
    def input_args(self):
        return ['x', 'reconstruction_mean', 'reconstruction_std']

    def __call__(self, args):
        x, reconstruction_mean, reconstruction_std = itemgetter(*self.input_args)(args)
        logs = {}

        # set nan to self.recon_nan_clip_val
        # TODO check why we need this, add it to other VAE system?
        reconstruction_mean[reconstruction_mean != reconstruction_mean] = self.recon_nan_clip_val
        reconstruction_std[reconstruction_std != reconstruction_std] = self.recon_nan_clip_val

        x = x.flatten(start_dim=1)
        reconstruction_mean = reconstruction_mean.flatten(start_dim=1)
        reconstruction_std = reconstruction_std.flatten(start_dim=1)

        loss = -torch.sum(
            -(0.5 * np.log(2 * np.pi) + 0.5 * reconstruction_std.pow(2).log())
            - 0.5 * ((x - reconstruction_mean).pow(2) / reconstruction_std.pow(2)), dim=1)

        return self.apply_reduction(loss), logs


class MSELoss(BaseLoss):
    """
    Computes the mean squared error (MSE) loss between the original ECG and its reconstruction.
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    @property
    def input_args(self):
        return ['x', 'reconstruction']
    
    def __call__(self, args):
        logs = {}
        x, reconstruction = args['x'], args['reconstruction']
        loss = (x - reconstruction)**2
        return self.apply_reduction(loss), logs


class BinaryFocalLoss(BaseLoss):
    def __init__(self, gamma=2, alpha=1, pos_weight=None):
        """Initializes the binary version of the focal loss.

        Args:
            gamma (int): The gamma value for the focal loss.
            alpha (int): The alpha value for the focal loss.
            pos_weight (tensor): The weight given to the positive class
                to deal with the class imbalance even more.
        """
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight if pos_weight is not None else pos_weight

    @property
    def input_args(self):
        return ['label', 'y_hat']
        
    def forward(self, args):
        """Calculate the loss."""
        y, y_hat = args['label'], args['y_hat'].squeeze()
        logs = {}
        
        pos_weight = self.pos_weight
        BCE_loss = F.binary_cross_entropy_with_logits(
            y_hat, y, reduction='none', pos_weight=pos_weight)
        # prevents nans when probability 0
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        return self.apply_reduction(focal_loss), logs
