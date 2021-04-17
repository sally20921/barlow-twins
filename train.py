import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

from data import Transform
from model import BarlowTwins

if __name__ == "__main__":
    seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int,
            help='Per device batch size.')
    parser.add_argument('--data_dir', default='./', type=str,
            help='Directory for pre-downloaded imagenet or cache for CIFAR10.')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    dm = ImagenetDataModule(
            batch_size = args.batch_size,
            data_dir = args.data_dir,
            train_transforms=Transform(),
            test_transforms=Transform(),
            val_transforms=Transform()
    )

    model = BarlowTwins(
            lr=0.2,
            weight_decay=1e-6,
            lambd=0.0051,
            projector=[8192,8192,8192],
            scale_loss = 0.024,
            per_device_batch_size=args.batch_size
    )

    trainer = pl.Trainer.from_argparse_args(
            args,
            max_epochs=1000,
            precision=16,
            accelerator='ddp',
            sync_batchnorm=True,
            benchmark=True,
            callbacks=ModelCheckpoint(save_last=True)
    )
    trainer.fit(model, dm)
