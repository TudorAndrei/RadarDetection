import os
from glob import glob

import cv2
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lightning_models import get_regression_output


def classif_report(module, train, val):
    """Generate an sklearn classification report for train and val

    Args:
        module: trained model
        train: train Dataloader
        val: validation Dataloader
    """

    module.eval()
    module.freeze()

    train_preds = []
    train_gt = []
    for img, label in tqdm(train):
        if module.op == "classification":
            output = module(img)
            predict = output.argmax(1)
        else:
            label = label.float()
            output = module(img)
            output = torch.squeeze(output, dim=1)
            predict = get_regression_output(output)
        predict = predict.int()
        label = label.int()
        for p, l in zip(predict, label):
            train_preds.append(p)
            train_gt.append(l)
    print(classification_report(train_gt, train_preds))

    preds = []
    gt = []
    for img, label in tqdm(val):
        if module.op == "classification":
            output = module(img)
            predict = output.argmax(1)
        else:
            label = label.float()
            output = module(img)
            output = torch.squeeze(output, dim=1)
            predict = get_regression_output(output)
        predict = predict.int()
        label = label.int()
        for p, l in zip(predict, label):
            preds.append(p)
            gt.append(l)
    print(classification_report(gt, preds))


def get_submission_dataloader(img_dir, bs, nw, transform=None):
    """Generate a Dataloader ready for submission

    Args:
        img_dir: the path to the image directory
        bs: batch size
        nw: number of workers
        transform: a torchvision transformation
    """
    dataset = RadarSubmission(img_dir, transform=transform)
    return DataLoader(dataset, batch_size=bs, num_workers=nw)


def kfold_generator(img_dir, csv_path, transform=None):
    """Generate a dataset for the Kfold CorssVal search

    Args:
        img_dir: the path to the image directory
        csv_path: the path to the csv file with labels
        transform: a torchvision transformation
    """
    train_samples = pd.read_csv(os.path.abspath(csv_path))
    return RadarDataset(img_dir, train_samples, transform=transform)


def data_generator(img_dir, csv_path, bs=4, nw=1, transform=None):
    """Generate Dataloaders for train and validation

    [TODO:description]

    Args:
        img_dir: path to the image directory
        csv_path: path the csv file with labels
        bs: batch size
        nw: number of workers
        transform: torchvision transformation
    """

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


class RadarDataset(Dataset):
    def __init__(self, img_dir: str, data_csv: pd.DataFrame, transform=None):
        super().__init__()
        self.image_dir = img_dir
        self.labels = dict(data_csv.values)
        # get a list of all the image names
        self.imgs = data_csv["id"].values
        # the number of images in the directory
        self.n_samples = len(self.imgs)
        self.transform = transform

    def __getitem__(self, idx):
        # get the image file name
        img_name = self.imgs[idx]
        # read the image from the image_dir
        img_path = os.path.join(self.image_dir, img_name)
        # load the image with opencv
        img = cv2.imread(img_path)
        # BGR to RGB conversion
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            # apply tranfromations
            img = self.transform(img)
        # Convert the label from [1,5] to [0,4]
        label = self.labels[img_name] - 1
        return img, label

    def __len__(self):
        return self.n_samples


class RadarSubmission(Dataset):
    def __init__(self, img_dir, transform):
        super().__init__()
        self.img_dir = img_dir
        self.img_paths = glob(img_dir)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.img_paths[idx]

    def __len__(self):
        return len(self.img_paths)
