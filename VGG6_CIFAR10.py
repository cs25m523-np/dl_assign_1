"""VGG6_CIFAR10.ipynb
Original file is located at
    https://colab.research.google.com/drive/1ZEPX8Cr1jURuu8vA-g3QxPo0XX7egE1l
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from PIL import Image, ImageEnhance, ImageOps
import random
import numpy as np
import torch

import copy
import csv
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# %pip install -q wandb
import wandb

wandb.login()

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        self.p1 = p1
        self.op1=operation1
        self.magnitude_idx1=magnitude_idx1
        self.p2 = p2
        self.op2=operation2
        self.magnitude_idx2=magnitude_idx2
        self.fillcolor=fillcolor
        self.init = 0

    def gen(self, operation1, magnitude_idx1, operation2, magnitude_idx2, fillcolor):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if self.init == 0:
            self.gen(self.op1, self.magnitude_idx1, self.op2, self.magnitude_idx2, self.fillcolor)
            self.init = 1
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img

class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"

class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

def GetCifar10(batchsize):
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
        Cutout(n_holes=1, length=16),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    # Two views of the same training images: augmented for training,
    # clean for validation.
    train_aug = datasets.CIFAR10(
        './data', train=True, transform=train_transform, download=True
    )
    train_clean = datasets.CIFAR10(
        './data', train=True, transform=eval_transform, download=True
    )
    test_data = datasets.CIFAR10(
        './data', train=False, transform=eval_transform, download=True
    )

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_aug), generator=generator).tolist()
    val_indices = indices[:5000]
    train_indices = indices[5000:]

    train_data = Subset(train_aug, train_indices)
    val_data = Subset(train_clean, val_indices)

    train_loader = DataLoader(
        train_data, batch_size=batchsize, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=batchsize, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=batchsize, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # fixed output size
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_activation(name):
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
     }
    return activations[name.lower()]()


def make_layers(cfg, batch_norm=False, activation="relu"):
      layers = []
      in_channels = 3
      for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), get_activation(activation)]
            else:
                layers += [conv2d, get_activation(activation)]
            in_channels = v
      return nn.Sequential(*layers)

def vgg(cfg, num_classes=10, batch_norm=True, activation="relu"):
      return VGG(make_layers(cfg, batch_norm=batch_norm, activation=activation), num_classes=num_classes)

# VGG-6 configuration
cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

    return total_loss / total, 100.0 * correct / total

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_optimizer(name, model, lr):
    name = name.lower()

    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)

    if name == "nesterov_sgd":
        return optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True
        )

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)

    if name == "adagrad":
        return optim.Adagrad(model.parameters(), lr=lr)

    if name == "rmsprop":
        return optim.RMSprop(
            model.parameters(), lr=lr, momentum=0.9
        )

    if name == "nadam":
        return optim.NAdam(model.parameters(), lr=lr)

    raise ValueError(f"Unknown optimizer: {name}")

criterion = nn.CrossEntropyLoss()
lr = 0.001
epochs = 3

def train_model(model, epochs, optimizer, train_loader, val_loader):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = -1
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

        train_loss = total_loss / total
        train_accuracy = 100.0 * correct / total
        val_loss, val_accuracy = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # W&B stores these values by epoch, producing training/validation curves.
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }, step=epoch + 1)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Train loss: {train_loss:.4f}, "
            f"Train accuracy: {train_accuracy:.2f}%, "
            f"Val loss: {val_loss:.4f}, "
            f"Val accuracy: {val_accuracy:.2f}%"
        )

        # Keep the epoch with the best validation accuracy.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return history

train_loader, val_loader, test_loader = GetCifar10(batchsize=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
lr = 0.001
epochs = 3

activations = ["relu", "sigmoid", "tanh", "silu", "gelu"]
activation_results = []

#Iterating over above activation functions
for activation in activations:
    print(f"\nTraining with activation: {activation}")

    with wandb.init(
        project="vgg6-cifar10",
        group="activation-comparison",
        name=f"activation-{activation}",
        config={
            "activation": activation,
            "optimizer": "adam",
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs": epochs,
        },
    ) as run:
        model = vgg(
            cfg_vgg6,
            num_classes=10,
            batch_norm=True,
            activation=activation
        ).to(device)

        optimizer = make_optimizer("adam", model, lr=0.001)
        history = train_model(
            model, epochs=epochs, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader
        )

        test_loss, test_accuracy = evaluate(model, test_loader)
        best_val_accuracy = max(history["val_accuracy"])

        # W&B scatter plot: validation accuracy for each epoch in this run.
        scatter_table = wandb.Table(
            data=[[epoch + 1, accuracy]
                  for epoch, accuracy in enumerate(history["val_accuracy"])],
            columns=["step", "validation_accuracy"],
        )
        wandb.log({
            "validation_accuracy_vs_step": wandb.plot.scatter(
                scatter_table,
                "step",
                "validation_accuracy",
                title="Validation accuracy vs. step",
            )
        })

        # Summary values are used by W&B run comparison plots.
        run.summary["best_val_accuracy"] = best_val_accuracy
        run.summary["test_accuracy"] = test_accuracy
        run.summary["test_loss"] = test_loss

        activation_results.append({
            "activation": activation,
            "best_validation_accuracy": best_val_accuracy,
            "test_accuracy": test_accuracy
        })

for result in activation_results:
    print(result)

optimizer_names = [
    "sgd", "nesterov_sgd", "adam",
    "adagrad", "rmsprop", "nadam"
]
optimizer_results = []

#Iterating over above optimizer types
for optimizer_name in optimizer_names:
    print(f"\nTraining with optimizer: {optimizer_name}")

    with wandb.init(
        project="vgg6-cifar10",
        group="optimizer-comparison",
        name=f"optimizer-{optimizer_name}",
        config={
            "activation": "relu",
            "optimizer": optimizer_name,
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs": epochs,
        },
    ) as run:
        model = vgg(
            cfg_vgg6,
            num_classes=10,
            batch_norm=True,
            activation="relu"
        ).to(device)

        optimizer = make_optimizer(optimizer_name, model, lr=0.001)
        history = train_model(
            model, epochs=epochs, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader
        )

        test_loss, test_accuracy = evaluate(model, test_loader)
        best_val_accuracy = max(history["val_accuracy"])

        scatter_table = wandb.Table(
            data=[[epoch + 1, accuracy]
                  for epoch, accuracy in enumerate(history["val_accuracy"])],
            columns=["step", "validation_accuracy"],
        )
        wandb.log({
            "validation_accuracy_vs_step": wandb.plot.scatter(
                scatter_table,
                "step",
                "validation_accuracy",
                title="Validation accuracy vs. step",
            )
        })

        run.summary["best_val_accuracy"] = best_val_accuracy
        run.summary["test_accuracy"] = test_accuracy
        run.summary["test_loss"] = test_loss

        optimizer_results.append({
            "optimizer": optimizer_name,
            "best_validation_accuracy": best_val_accuracy,
            "test_accuracy": test_accuracy
        })

for result in optimizer_results:
    print(result)
