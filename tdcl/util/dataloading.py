from glob import glob

import numpy as np
import pandas as pd
import tonic
import tonic.transforms as transforms
import torch
import torchvision
from tonic import DiskCachedDataset
from tonic.datasets import NMNIST, SHD
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Compose
from tqdm import tqdm


def create_nmnist_dataloaders(
    batch_size: int,
    bins: int,
    filter_time: int,
    save_path: str = "./data",
    cache_path: str = "./cache/nmnist/",
    num_workers: int = 2,
    reset_cache: bool = True,
    drop_last: bool = False,
    split_train: bool = False,
    flatten: bool = False,
) -> tuple[DataLoader, DataLoader] | tuple[DataLoader, DataLoader, DataLoader]:
    sensor_size = NMNIST.sensor_size

    # Denoise removes isolated, one-off events
    frame_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=filter_time),
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=bins),
        ]
    )

    trainset = NMNIST(save_to=save_path, transform=frame_transform, train=True)
    testset = NMNIST(save_to=save_path, transform=frame_transform, train=False)

    if flatten:
        train_transform = Compose(
            [
                torch.from_numpy,
                torchvision.transforms.RandomRotation([-10, 10]),
                lambda x: torch.flatten(x, start_dim=1),
            ]
        )
        test_transform = Compose(
            [torch.from_numpy, lambda x: torch.flatten(x, start_dim=1)]
        )
    else:
        train_transform = Compose(
            [
                torch.from_numpy,
                torchvision.transforms.RandomRotation([-10, 10]),
            ]
        )
        test_transform = Compose(
            [
                torch.from_numpy,
            ]
        )

    if split_train:
        trainset, valset = torch.utils.data.random_split(trainset, [0.9, 0.1])

        cached_trainset = DiskCachedDataset(
            trainset,
            transform=train_transform,
            cache_path=cache_path + "train/",
            reset_cache=reset_cache,
        )

        # no augmentations for the testset
        cached_valset = DiskCachedDataset(
            valset,
            transform=test_transform,
            cache_path=cache_path + "val/",
            reset_cache=reset_cache,
        )
        cached_testset = DiskCachedDataset(
            testset,
            transform=test_transform,
            cache_path=cache_path + "test/",
            reset_cache=reset_cache,
        )

        train_loader = DataLoader(
            cached_trainset,
            batch_size=batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        val_loader = DataLoader(
            cached_valset,
            batch_size=batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        test_loader = DataLoader(
            cached_testset,
            batch_size=batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        return train_loader, val_loader, test_loader
    else:
        cached_trainset = DiskCachedDataset(
            trainset,
            transform=train_transform,
            cache_path=cache_path + "train/",
            reset_cache=reset_cache,
        )
        cached_testset = DiskCachedDataset(
            testset,
            transform=test_transform,
            cache_path=cache_path + "test/",
            reset_cache=reset_cache,
        )

        train_loader = DataLoader(
            cached_trainset,
            batch_size=batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        test_loader = DataLoader(
            cached_testset,
            batch_size=batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        return train_loader, test_loader


def create_shd_dataloaders(
    batch_size: int,
    bins: int = 30,
    filter_time: int = 10000,
    save_path: str = "./data",
    cache_path: str = "./cache/shd_new/",
    num_workers: int = 2,
    reset_cache: bool = True,
    val_split: bool = False,
) -> tuple[DataLoader, DataLoader]:
    shd_sensor_size = SHD.sensor_size
    if filter_time > 0:
        frame_transform = Compose(
            [
                # transforms.Denoise(filter_time=filter_time),
                tonic.transforms.ToFrame(
                    sensor_size=shd_sensor_size,
                    n_time_bins=bins,
                ),
                lambda x: x[:, 0, :],
                torch.from_numpy,
                lambda x: x[None, ...],
                torchvision.transforms.RandomAffine(degrees=0, translate=(0.05, 0.0)),
                lambda x: x[0, ...],
            ]
        )
    else:
        frame_transform = Compose(
            [
                tonic.transforms.ToFrame(
                    sensor_size=shd_sensor_size,
                    n_time_bins=bins,
                ),
                lambda x: x[:, 0, :],
                torch.from_numpy,
                lambda x: x[None, ...],
                torchvision.transforms.RandomAffine(degrees=0, translate=(0.05, 0.0)),
                lambda x: x[0, ...],
            ]
        )

    test_frame_transform = Compose(
        [
            tonic.transforms.ToFrame(
                sensor_size=shd_sensor_size,
                n_time_bins=bins,
            ),
            lambda x: x[:, 0, :],
        ]
    )

    train_dataset = SHD(save_to=save_path, train=True, transform=frame_transform)
    if val_split:
        train_dataset, val_dataset = random_split(train_dataset, lengths=[0.95, 0.05])
    test_dataset = SHD(save_to=save_path, train=False, transform=test_frame_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        num_workers=num_workers,
    )
    if val_split:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            num_workers=num_workers,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        num_workers=num_workers,
    )

    if val_split:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader


def create_wear_dataset(
    batch_size: int,
    window_size: int,
    stride: int,
    val_subj: int,
    num_workers: int = 2,
    shuffle: bool = False,
):
    assert val_subj >= 0 and val_subj <= 17
    window_size += 1

    path = "./data/WEAR/right_w/"
    files = glob(path + "sbj_*.csv")
    label_names = [
        "nan",
        "bench-dips",
        "burpees",
        "jogging",
        "jogging (butt-kicks)",
        "jogging (rotating arms)",
        "jogging (sidesteps)",
        "jogging (skipping)",
        "lunges",
        "lunges (complex)",
        "push-ups",
        "push-ups (complex)",
        "sit-ups",
        "sit-ups (complex)",
        "stretching (hamstrings)",
        "stretching (lumbar rotation)",
        "stretching (lunging)",
        "stretching (shoulders)",
        "stretching (triceps)",
    ]
    labels_map = {}
    for label in label_names:
        if label == "nan":
            labels_map[label] = 0
        elif "jogging" in label:
            labels_map[label] = 1
        elif "stretching" in label:
            labels_map[label] = 2
        else:
            labels_map[label] = 3
    features = ["right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"]

    def convert_to_dataset(files: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        X = np.empty((0, window_size, 3))
        y = []
        for file in tqdm(files):

            df = pd.read_csv(file)
            df = df[features + ["label"]]
            df["label"] = df["label"].fillna("nan")

            for i in range(0, len(df), stride):
                window = df.iloc[i : i + window_size, :]
                if len(window) == window_size:
                    new_data = window[features].values[None, ...]
                    new_data = new_data.astype(np.float32)
                    X = np.concatenate([X, new_data], axis=0)

                    labels = [
                        labels_map[label_name] for label_name in window["label"].values
                    ]
                    labels = np.array(labels)
                    labels, counts = np.unique(labels, return_counts=True)
                    label = labels[np.argmax(counts)]
                    y.append(label)

        y = np.array(y, dtype=np.int32)

        thr0 = 0.6656900063700987
        thr1 = 1.96634368036615
        thr2 = 1.104125298835588
        thr3 = 0.12750019166261672
        thr4 = 0.00800747384317868

        thrs = (thr0, thr1, thr2, thr3, thr4)
        delta = X[:, 1:, :] - X[:, :-1, :]

        # X.shape: (N, T, 3)
        spikes = np.empty((X.shape[0], window_size - 1, 0))
        for thr in thrs:
            spikes = np.concatenate([spikes, (delta > thr)], axis=-1)

        for thr in thrs:
            spikes = np.concatenate([spikes, (delta < -thr)], axis=-1)

        spikes = torch.from_numpy(spikes).float()
        y = torch.from_numpy(y).long()
        return spikes, y

    files.sort(key=lambda file: int(file.split("_")[-1].split(".")[0]))
    train_files = [file for file in files if str(val_subj) not in file]
    test_files = [file for file in files if str(val_subj) in file]

    def time_first(data: tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        x, y = zip(*data)
        return torch.stack(x, axis=1).float(), torch.stack(y).long()

    spikes_train, y_train = convert_to_dataset(train_files)
    train_dataset = TensorDataset(spikes_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=time_first,
    )

    spikes_test, y_test = convert_to_dataset(test_files)
    test_dataset = TensorDataset(spikes_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=time_first,
    )

    return train_loader, test_loader
