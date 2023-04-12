import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

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

class ConvolutionBlock(pl.LightningModule):
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int,
            stride: int, padding: int, batch_norm: bool=True,
            activation: Literal['relu', 'gelu', 'silu', 'mish']="relu"
            ):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # if batch_norm:
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        if activation == "relu":
            self.activation = nn.ReLU()
        if activation == "gelu":
            self.activation = nn.GELU()
        if activation == "silu":
            self.activation = nn.SiLU()
        if activation == "mish":
            self.activation = nn.Mish()
        else:
            self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:          
            x = self.bn(x)
        x = self.activation(x)
        x = self.maxpool(x)  
        return x
    
class CNNBase(pl.LightningModule):
    def __init__(
            self, in_channels: int=3, out_channels: int=32,
            kernel_size: int=3, stride: int=1, padding: int=1,
            batch_norm: bool=True, activation: Literal['relu', 'gelu', 'silu', 'mish']="relu",
            kernel_strategy: Literal['same', 'double', 'half'] = 'same'
            ):
        super(CNNBase, self).__init__()
        if kernel_strategy == 'same':
            coeff = 1
        elif kernel_strategy == 'double':
            coeff = 2
        elif kernel_strategy == 'half':
            coeff = 0.5
        for i in range(1, 6):
            setattr(self, f"conv{i}", ConvolutionBlock(in_channels, out_channels, kernel_size, stride, padding, batch_norm=batch_norm, activation=activation))
            in_channels = out_channels
            out_channels = int(out_channels * coeff)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class ClassifierHead(pl.LightningModule):
    def __init__(self, num_classes: int, in_size: int, hidden_size: int, dropout: float=0.0, activation: Literal['relu', 'gelu', 'silu', 'mish']='relu') -> None:
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        if activation == "gelu":
            self.activation = nn.GELU()
        if activation == "silu":
            self.activation = nn.SiLU()
        if activation == "mish":
            self.activation = nn.Mish()
        else:
            self.activation = nn.ReLU()
        self.o_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.dropout(self.fc1(x)))
        x = self.o_activation(self.fc2(x))
        return x
    
class NeuralNetwork(pl.LightningModule):
    def __init__(
            self, in_channels: int, out_channels: int,  # Convolutional Layers
            kernel_size: int, stride: int, padding: int,
            batch_norm: bool=True, activation: Literal['relu', 'gelu', 'silu', 'mish']="relu",
            kernel_strategy: Literal['same', 'double', 'half'] = 'same',
            dropout: float=0.0, num_classes: int=10, hidden_size: int=64,   # Fully-Connected Layers
            dataset_path: str='../inaturalist_12K/', num_workers: int=2, batch_size: int=32,  augmentation: bool=True # Datasets
            ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cnn = CNNBase(in_channels, out_channels, kernel_size, stride, padding, batch_norm, activation, kernel_strategy)
        in_size = self.get_in_size()
        self.classifier = ClassifierHead(num_classes, in_size, hidden_size, dropout, activation)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_dataset, self.val_dataset, self.test_dataset = get_datasets(augmentation, dataset_path)
        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def get_in_size(self):
        x = torch.randn(1, 3, 224, 224)
        x = self.cnn(x)
        return x.numel()
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('train/loss', loss)
        self.log('train/acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('val/loss', loss)
        self.log('val/acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('test/loss', loss)
        self.log('test/acc', acc)
        return loss
    
    def configure_optimizers(self) -> None:
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)