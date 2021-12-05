import pandas as pd
from glob import glob
import cv2
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def get_submission_dataloader(img_dir, bs, nw):
    dataset = RadarSubmission(img_dir)
    return DataLoader(dataset, batch_size=bs, num_workers=nw)


def data_generator(img_dir, csv_path, bs=4, nw=1):

    file_path = pd.read_csv(os.path.abspath(csv_path))

    train_samples, val_samples = train_test_split(
        file_path, test_size=0.2, shuffle=True, stratify=file_path["label"]
    )

    return DataLoader(
        RadarDataset(img_dir, train_samples),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
    ), DataLoader(
        RadarDataset(img_dir, val_samples),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
    )


class RadarSubmission(Dataset):
    def __init__(self, img_dir) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_paths = glob(img_dir)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        return img, self.img_paths[idx]

    def __len__(self):
        return len(self.img_paths)


class RadarDataset(Dataset):
    def __init__(self, img_dir: str, data_csv: pd.DataFrame) -> None:
        super().__init__()
        self.image_dir = img_dir
        self.labels = dict(data_csv.values)
        self.imgs = data_csv["id"].values
        self.n_samples = len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        # print(img_name)
        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        label = self.labels[img_name] - 1
        # print(label)
        return img, label

    def __len__(self):
        return self.n_samples
