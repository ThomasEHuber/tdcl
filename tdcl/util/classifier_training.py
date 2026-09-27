import os
from typing import *

import dill
import torch
from snntorch import utils
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.wrappers import MetricTracker
from tqdm import tqdm


@torch.no_grad
def create_dataloaders(
    data_loader: DataLoader,
    layers: list[nn.Module],
    concatenate: bool,
    shuffle: bool,
    device: str,
) -> DataLoader:
    """
    Runs data through the network to create a new dataloader for
    for classifier training, otherwise it would take ages.
    If concatenate == True: all out mode
    Else: last out mode
    """
    if len(data_loader) == 0:
        return data_loader
    ys = []
    xs = []
    for x, y in tqdm(data_loader):
        if concatenate:
            temp = [x.cpu()]

        for i, layer in enumerate(layers):
            utils.reset(layer)
            x = x.to(device)  # (T, B, N)

            x, _ = layer(x)  # (T, B, N)
            if concatenate:
                temp.append(x.cpu().detach())

        if concatenate:
            temp = torch.concatenate(temp, axis=-1)
        else:
            temp = x.cpu()
        xs.append(temp)
        ys.append(y.cpu().detach().clone())

    xs = torch.concatenate(xs, dim=1)  # (T, B, N)
    xs = torch.swapaxes(xs, 0, 1)  # (B, T, N)
    ys = torch.concatenate(ys)
    print(xs.shape)
    print(ys.shape)
    dataset = TensorDataset(xs, ys)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=2)
    return data_loader


def train_epoch(
    classifier: nn.Linear,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    train_loss_tracker: MetricTracker,
    train_acc_tracker: MetricTracker,
    device: str,
) -> None:
    train_loss_tracker.increment()
    train_acc_tracker.increment()
    for x, y in tqdm(train_loader):
        x = x.to(device)  # (B, T, N)
        y = y.to(device)
        x = torch.swapaxes(x, 0, 1)  # (T, B, N)
        pred = classifier(x)
        pred = pred.mean(dim=0)  # (B, N)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_tracker.update(loss)
        train_acc_tracker.update(pred, y)


@torch.no_grad()
def test_epoch(
    classifier: nn.Linear,
    loss_fn: Callable,
    test_loader: DataLoader,
    test_loss_tracker: MetricTracker,
    test_acc_tracker: MetricTracker,
    device: str,
) -> None:
    test_loss_tracker.increment()
    test_acc_tracker.increment()
    for x, y in tqdm(test_loader):
        x = x.to(device)
        y = y.to(device)

        x = torch.swapaxes(x, 0, 1)  # (T, B, N)
        pred = classifier(x)
        pred = pred.mean(dim=0)  # (B, N)
        loss = loss_fn(pred, y)

        test_loss_tracker.update(loss)
        test_acc_tracker.update(pred, y)


def train_classifier(
    layers: list[nn.Module],
    features: list[int] | int,
    num_classes: int,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    key: str,
    path: str,
    device: str,
) -> None:
    concatenate = isinstance(features, list)
    train_loader = create_dataloaders(
        train_loader, layers, concatenate, shuffle=True, device=device
    )
    val_loader = create_dataloaders(
        val_loader, layers, concatenate, shuffle=False, device=device
    )
    test_loader = create_dataloaders(
        test_loader, layers, concatenate, shuffle=False, device=device
    )

    if concatenate:
        in_features = sum(features)
    else:
        in_features = features

    classifier = nn.Linear(in_features, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(classifier.parameters(), lr=3e-4)

    train_loss_tracker = MetricTracker(MeanMetric(), maximize=False).to(device)
    train_acc_tracker = MetricTracker(
        MulticlassAccuracy(num_classes), maximize=True
    ).to(device)
    val_loss_tracker = MetricTracker(MeanMetric(), maximize=False).to(device)
    val_acc_tracker = MetricTracker(MulticlassAccuracy(num_classes), maximize=True).to(
        device
    )
    test_loss_tracker = MetricTracker(MeanMetric(), maximize=False).to(device)
    test_acc_tracker = MetricTracker(MulticlassAccuracy(num_classes), maximize=True).to(
        device
    )

    for epoch in range(epochs):
        print(f"\t[{epoch+1}/{epochs}] running...")
        train_epoch(
            classifier,
            loss_fn,
            optimizer,
            train_loader,
            train_loss_tracker,
            train_acc_tracker,
            device,
        )
        train_losses = train_loss_tracker.compute_all()
        train_accs = train_acc_tracker.compute_all()

        test_epoch(
            classifier,
            loss_fn,
            val_loader,
            val_loss_tracker,
            val_acc_tracker,
            device,
        )
        val_losses = val_loss_tracker.compute_all()
        val_accs = val_acc_tracker.compute_all()

        test_epoch(
            classifier,
            loss_fn,
            test_loader,
            test_loss_tracker,
            test_acc_tracker,
            device,
        )
        test_losses = test_loss_tracker.compute_all()
        test_accs = test_acc_tracker.compute_all()

        print("train loss is: ", train_losses[-1].item())
        print("val loss is: ", val_losses[-1].item())
        print("test loss is: ", test_losses[-1].item())
        print("train acc is: ", train_accs[-1].item())
        print("val acc is: ", val_accs[-1].item())
        print("test acc is: ", test_accs[-1].item())

        if os.path.exists(path + "final_checkpoint.pkl"):
            checkpoint: dict[str, dict[str, Any]] = dill.load(
                open(path + "final_checkpoint.pkl", "rb")
            )
        else:
            checkpoint = {}

        checkpoint[key] = {}
        checkpoint[key]["train_loss"] = train_losses
        checkpoint[key]["val_loss"] = val_losses
        checkpoint[key]["test_loss"] = test_losses
        checkpoint[key]["train_acc"] = train_accs
        checkpoint[key]["val_acc"] = val_accs
        checkpoint[key]["test_acc"] = test_accs
        checkpoint[key]["classifier_state_dict"] = classifier.state_dict()
        checkpoint[key]["optimizer_state_dict"] = optimizer.state_dict()
        dill.dump(checkpoint, open(path + "final_checkpoint.pkl", "wb"))
