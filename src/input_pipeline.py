import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.filter import filter
from src.utils.preprocess import normlize


def base_dataset(cfg, split):
    batch_size = cfg["general"]["batch_size"]
    if split == "train":
        shuffle = True
    else:
        shuffle = False

    dataset = DataHelper(cfg, split)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
    )
    return data_loader


class DataHelper(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.mean = cfg["data"]["mean"]
        self.std = cfg["data"]["std"]
        self.enable_filter = cfg["data"]["enable_filter"]
        self.sample_rate = cfg["data"]["sample_rate"]
        self.low_freq = cfg["data"]["low_freq"]
        self.high_freq = cfg["data"]["high_freq"]
        file_path = cfg[split]["dataset_path"]
        with open(file_path, "r") as f:
            self.infos = f.readlines()

    def to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float)

    def prepare_input(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        if self.enable_filter:
            data = np.concatenate([data, data], axis=1)
            data = filter(data, self.low_freq, self.high_freq, self.sample_rate)
            data = data[:, int(data.shape[1] / 2) :]
        data = normlize(data, self.mean, self.std)
        data = self.to_tensor(data)
        return data

    def prepare_target(self, target_path):
        with open(target_path, "r") as f:
            target = [float(f.read())]
        target = self.to_tensor(target)
        return target.squeeze()

    def __getitem__(self, index):
        info = self.infos[index]
        info = json.loads(info)
        data_path = info["patch_path"]
        target_path = info["target_path"]
        data = self.prepare_input(data_path)
        target = self.prepare_target(target_path)

        return data, target

    def __len__(self):
        return len(self.infos)
