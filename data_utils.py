import os
from glob import glob

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def get_submission_dataloader(img_dir, bs, nw, transform=None):
    dataset = RadarSubmission(img_dir, transform=transform)
    return DataLoader(dataset, batch_size=bs, num_workers=nw)


def kfold_generator(img_dir, csv_path, transform=None):
    train_samples = pd.read_csv(os.path.abspath(csv_path))
    return RadarDataset(img_dir, train_samples, transform=transform)


def data_generator(img_dir, csv_path, bs=4, nw=1, transform=None):

    file_path = pd.read_csv(os.path.abspath(csv_path))

    train_samples, val_samples = train_test_split(
        file_path, test_size=0.2, shuffle=True, stratify=file_path["label"]
    )

    return DataLoader(
        RadarDataset(img_dir, train_samples, transform=transform),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
    ), DataLoader(
        RadarDataset(img_dir, val_samples, transform=transform),
        batch_size=bs,
        # shuffle=True,
        num_workers=nw,
        pin_memory=True,
    )


class RadarSubmission(Dataset):
    def __init__(self, img_dir, transform):
        super().__init__()
        self.img_dir = img_dir
        self.img_paths = glob(img_dir)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.img_paths[idx]

    def __len__(self):
        return len(self.img_paths)


class RadarDataset(Dataset):
    def __init__(self, img_dir: str, data_csv: pd.DataFrame, transform=None):
        super().__init__()
        self.image_dir = img_dir
        self.labels = dict(data_csv.values)
        self.imgs = data_csv["id"].values
        self.n_samples = len(self.imgs)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        # print(img_name)
        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img / 255
        if self.transform:
            img = self.transform(img)
        label = self.labels[img_name] - 1

        # print(label)
        return img, label

    def __len__(self):
        return self.n_samples
