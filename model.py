from typing import List, Union, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

from optimizer import LARS, CosineWarmupScheduler

class BarlowTwins(pl.LightningModule):
    def __init__(
            self, 
            projector: Union[List, Tuple] = (4096, 4096),
            per_device_batch_size: int = 1,
            scale_loss: float = 1.0 / 32,
            lr: float = 0.2,
            weight_decay: float = 1e-6,
            lambd: float = 3.9e-6,
            num_warmup_steps_or_ratio: Union[int, float] = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.scale_loss = scale_loss
        self.per_device_batch_size = per_device_batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambd = lambd
        self.num_warmup_steps_or_ratio = num_warmup_steps_or_ratio
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        sizes = [2048] + projector
        layers = []
        for i in range(len(sizes) -2):
            layers.append(nn.Linear(sizes[i], size[i+1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(size[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representation z1 and z2
        self.bn = nn.BatchNorm1d(size[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.per_device_batch_size * self.trainer.num_processes)
        self.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = self.off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        loss = on_diag + self.lambd * off_diag
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()


    def all_reduce(self, c):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)

    def common_step(self, batch, batch_idx):
        (y1, y2), _  = batch
        loss = self(y1, y2)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('validation_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('test_loss', loss)


    def configure_optimizers(self):
        optimizer = LARS(
                self.parameters(),
                lr = 0,
                weight_decay=self.weight_decay,
                weight_decay_filter=exclude_bias_and_norm,
                lars_adaptation_filter=exclude_bias_and_norm
        )

        total_training_steps = self.total_training_steps
        num_warmup_steps = self.compute_warmup(total_training_steps, self.num_warmup_steps_or_ratio)
        lr_scheduler = CosineWarmupScheduler(
                optimizer = optimizer,
                batch_size = self.per_device_batch_size,
                warmup_steps=num_warmup_steps,
                max_steps = total_training_steps,
                lr = self.lr
        )
        return [optimizer], [
                {
                    'scheduler': lr_scheduler,
                    'interval': 'step', # the unit of the scheduler's step size
                }
        ]

        @property
        def total_training_steps(self) -> int:
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            num_devices = self.trainer.tpu_cores if self.trainer.tpu_cores else self.trainer.num_processes
            effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
            max_estimated_steps = (dataset_size // effective_batch_size ) * self.trainer.max_epochs

            if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
                return self.trainer.max_steps
            reutrn max_estimated_steps

        def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
            return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_training_steps

        def exclude_bias_and_norm(p):
            return p.ndim == 1
