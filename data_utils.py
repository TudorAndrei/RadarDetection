import pandas as pd
import cv2
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class RadarDataset(Dataset):
    def __init__(self, img_path: str, data_csv: pd.DataFrame) -> None:
        super().__init__()
        self.image_paths = img_path
        self.labels = dict(data_csv.values)
        self.imgs = data_csv['id'].values
        self.n_samples = len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        # print(img_name)
        img_path = os.path.join(self.image_paths, img_name)
        img = cv2.imread(img_path)
        label = self.labels[img_name] - 1
        # print(label)
        return img, label

    def __len__(self):
        return self.n_samples


class DataGenerator:
    def __init__(self, img_path, path, bs=4, nw=1) -> None:
        self.img_path = img_path
        self.bs = bs
        self.nw = nw
        file_path = pd.read_csv(path)
        self.train_samples, self.val_samples = train_test_split(
            file_path, test_size=0.2, shuffle=True, stratify=file_path["label"]
        )
        # Check if the split is stratified
        # print(self.train_samples['label'].value_counts())
        # print(self.val_samples['label'].value_counts())

    def get_train(self):
        return DataLoader(
            RadarDataset(self.img_path, self.train_samples),
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
        )

    def get_val(self):
        return DataLoader(
            RadarDataset(self.img_path, self.val_samples),
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
        )
