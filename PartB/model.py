import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torchvision
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchmetrics import Accuracy

from typing import Any, Literal
import random

def get_datasets(augmentation: bool=True, path: str = '../inaturalist_12K/'):
    augs = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    ]
    ts = [transforms.RandomResizedCrop((224, 224))]
    if augmentation:
        ts += random.sample(augs, k=3)
    ts += [transforms.ToTensor(), transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])]
    transform = transforms.Compose(ts)
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'val')
    dataset = ImageFolder(train_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = (len(dataset) - train_size) 
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    test_dataset = ImageFolder(test_path, transform=transform)
    return train_dataset, val_dataset, test_dataset

class FineTuneModel(pl.LightningModule):
    def __init__(
        self, learning_rate: float=1e-4, num_classes: int=10, augmentation: bool=True, 
        dataset_path: str="../inaturalist_12K", num_workers: int=2, batch_size: int=32
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.augmentation = augmentation
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in backbone.parameters():
            param.requires_grad = False
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(backbone.fc.in_features, self.num_classes)
        del backbone
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_dataset, self.val_dataset, self.test_dataset = get_datasets(self.augmentation, self.dataset_path)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('test/loss', loss)
        self.log('test/accuracy', accuracy)
        return {'loss': loss, 'y_hat': y_hat, 'y': y}

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        