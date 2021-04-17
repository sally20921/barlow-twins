import urllib
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.transforms import transforms

from model import BarlowTwins

class LienarEvaluateBarlowTwins(pl.LightningModule):
    def __init__(self,
            freeze_backbone: bool,
            lr_classifier: float,
            lr_backbone: float,
            weight_decay: float, 
            epochs: int, 
            model_path: Optional[str] = None,
            repo_or_dir: Optional[str] = 'facebookresearch/barlowtwins:main',
            model_name: Optional[str] = 'resnet50'):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = self._load_backbone(
                repo_or_dir=repo_or_dir,
                model_name=model_name,
                model_path=model_path
        )
        self.backbone.fc = nn.Identity()

        if self.hparams.freeze_backbone:
            self.backbone.requires_grad_(False)

        self.classifier = nn.Linear(2048, 1000)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy_top_k_one = Accuracy(top_k=1)
        self.accuracy_top_k_five = Accuracy(top_k=5)

    def _load_backbone(self,
            repo_or_dir: str,
            model_name: str,
            model_path: Optional[str] = None):
        if model_path:
            return BarlowTwins.load_from_checkpoint(model_path).backbone
        rank_zero_info('No model provided, loading torch hub pre-trained model')
        return torch.hub.load(repo_or_dir, model_name)

    def configure_optimizers(self):
        param_groups = [dict(params=self.classifier.parameters(), lr=self.hparams.lr_classifier)]
        if not self.hparams.freeze_backbone:
            param_groups.append(dict(params=self.backbone.parameters(), lr=self.hparams.lr_backbone))
        optimizer = torch.optim.SGD(param_groups, 0, momentum=0.9, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.epochs)
        return [optimizer], [scheduler]

    def forward(self, image):
        features = self.backbone(image)
        output = self.classifier(features)
        return output

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = self.criterion(output, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, prefix='Validation'):
        images, target = batch
        output = self(images)
        loss = self.criterion(output, target)
        output = F.softmax(output)
        self.log(f'{prefix} Acc@1', self.accuracy_top_k_one(output, target))
        self.log(f'{prefix} Acc@5', self.accuracy.top_k_five(output, target))
        self.log(f'{prefix} Loss', loss)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, prefix='Test')

class EvaluateImagenetDataModule(ImagenetDataModule):
    '''
    extend the imagenetmodule to allow selecting a percent of files for training
    '''
    def __init__(self,
            data_dir: str,
            *args: Any,
            train_percent: int,
            **kwargs: Any):
        self.train_files = None
        if train_percent < 100:
            # downlaod simclr subset for imagenet
            url = f'https://raw.githubusercontent.com/google-research/simclr/master/' \
                  f'imagenet_subsets/{train_percent}percent.txt'
            self.train_files = urllib.request.urlopen(url).readlines()

        super().__init__(data_dir, *args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        loader = super().train_dataloader()
        if self.train_files:
            for fname in self.train_files:
                fname = fname.decode().strip()
                cls = fname.split('_')[0]
                loader.dataset.samples.append(
                        (Path(self.data_dir) / 'train' / cls / fname, loader.dataset.wnid_to_idx[cls]))

        return loader

if __name__ == "__main__":
    seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int, 
            help='Per device batch size.')
    parser.add_argument('--train_percent', choices=[100, 10, 1], default=1, type=int,
            help='When using ImageNet, decide the amount of training data to use.')
    parser.add_argument('--data_dir', default='./', type=str,
            help='Directory for pre-downloaded ImageNet or cache for CIFAR10')
    parser.add_argument('--model_path', default=None, type=str,
            help="Path to Lightning BarlowTwins model. We'll extract the backbone parameter from this." "If not provided, we'll load the pre-trained model from torch hub.")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            normalize
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    dm = EvaluateImagenetDataModule(
            batch_size= args.batch_size,
            data_dir=args.data_dir,
            train_transforms=train_transforms,
            test_transforms=val_transforms,
            val_transforms=val_transforms,
            train_percent=args.train_percent
    )

    model = LinearEvaluateBarlowTwins(
            freeze_backbone=False,
            lr_classifier=0.5,
            lr_backbone=0.002,
            weight_decay=0,
            epochs=20,
    )

    trainer = pl.Trainer.from_argparse_args(
            args, 
            max_epochs=20,
            precision=16,
            accelerator='ddp',
            benchmark=True,
            callbacks=ModelCheckpoint(save_last=True)
    )
    trainer.fit(model, dm)
    

