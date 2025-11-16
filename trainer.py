import torch
import torch.nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from typing import Callable


class Trainer:
    def __init__(
            self,
            loss: str,
            log_dir: str
    ):
        if loss not in {'l1', 'l2'}:
            raise ValueError('Unknown loss function')

        self.loss = loss
        
        # TODO: Set up dataloaders

        self.writer = SummaryWriter(log_dir)

    def compute_loss(self, noise: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        if self.loss == 'l1':
            loss = F.l1_loss(noise, predicted_noise, reduction='sum') / noise.shape[0]

        elif self.loss == 'l2':
            loss = F.mse_loss(noise, predicted_noise, reduction='sum') / noise.shape[0]

        return loss

    def log(self):
        # TODO: figure out what to log
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
