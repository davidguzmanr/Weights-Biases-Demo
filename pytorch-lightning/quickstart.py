import wandb

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.nn import functional as F
from torch import nn

import torchvision
from torchvision import transforms

from torchmetrics import Accuracy
import random

random.seed(42)

class Model(LightningModule):
    def __init__(
        self,
        epochs=10,
        batch_size=128,
        lr=1e-3,
        dropout=0.0
        ):
        super().__init__()

        # Hyperparameters are all the kwargs passed
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256,10)
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.softmax(dim=1).argmax(dim=1)

        # logging the loss
        self.log('train_loss', loss, on_step=False, on_epoch=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.softmax(dim=1).argmax(dim=1)

        # logging the loss
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        
        # logging a metric
        self.log('val_accuracy', self.val_acc(preds, y), on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.log_image_table(x, preds, y, logits.softmax(dim=1))
        
        return loss

    @staticmethod
    def log_image_table(images, predicted, labels, probs):
        """
        Log a wandb.Table with (img, pred, target, scores)
        """
        table = wandb.Table(columns=['image', 'pred', 'target'] + [f'score_{i}' for i in range(10)])
        for img, pred, targ, prob in zip(images.to('cpu'), predicted.to('cpu'), labels.to('cpu'), probs.to('cpu')):
            table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
            
        wandb.log({'predictions_table':table}, commit=False)

    @property
    def transform(self):
        return transforms.ToTensor()

    def train_dataloader(self):
        full_dataset = torchvision.datasets.MNIST(
            root='.', 
            train=True, 
            transform=self.transform, 
            download=True
        )
        sub_dataset = torch.utils.data.Subset(
            full_dataset, 
            indices=range(0, len(full_dataset), 5)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=sub_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=2
        )

        return dataloader

    def val_dataloader(self):
        full_dataset = torchvision.datasets.MNIST(
            root='.', 
            train=False, 
            transform=self.transform, 
            download=True
        )
        sub_dataset = torch.utils.data.Subset(
            full_dataset, 
            indices=range(0, len(full_dataset), 5)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=sub_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=2
        )

        return dataloader

def main():
    # Launch 10 experiments, trying different dropout rates
    for i in range(10):
        config = {
            'epochs': 10,
            'batch_size': 128,
            'lr': 1e-3,
            'dropout': random.uniform(0.01, 0.80),
        }
        wandb_logger = WandbLogger(
            project='my-test-project',
            entity='davidguzmanr',
            group='PyTorch-Lightning',
            name=f'run-{i}'
        )

        # seed_everything(42, workers=True)
        trainer = Trainer(
            logger=wandb_logger,    # W&B integration
            log_every_n_steps=50,   # set the logging frequency
            gpus=-1,                # use all GPUs
            max_epochs=10,          # number of epochs
            deterministic=True,     # keep it deterministic
        )

        model = Model(**config)

        trainer.fit(model)
        wandb.finish()

if __name__ == '__main__':
    main()