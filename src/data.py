import json

import numpy as np
import torch
from scipy import signal
from torch.utils.data import DataLoader, Dataset


def filter(data, low_freq, high_freq, sample_rate):
    b, a = signal.butter(4, [low_freq, high_freq], btype="bandpass", fs=sample_rate)
    filtered_signal = signal.lfilter(b, a, data, axis=1)
    return filtered_signal


def descale(data):
    return (data - 8388608) / 8388608 * 5000000 / 50


def normlize(data, mean, std):
    data = np.clip(data, a_min=-50, a_max=50)
    data = (data + 50) / 100
    # for i in range(np.array(data).shape[0]):
    #     data[i,:] = (data[i,:]-mean[i])/std[i]
    return data


def sliding_window(data, win_size, step_size):
    if data is None:
        return []
    length = int((data.shape[-1] - win_size) / step_size + 1)
    patchs = []
    for i in range(length):
        patch = data[:, i * step_size : i * step_size + win_size]
        patchs.append(patch)
    return patchs


def base_dataset(cfg, split):
    batch_size = cfg["general"]["batch_size"]
    shuffle = split == "train"

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


class MakeIdentityDatabase:
    def __init__(self, cfg):
        self.std = cfg["data"]["std"]
        self.mean = cfg["data"]["mean"]
        self.enable_filter = cfg["data"]["enable_filter"]
        self.low_freq = cfg["data"]["low_freq"]
        self.high_freq = cfg["data"]["high_freq"]
        self.sample_rate = cfg["data"]["sample_rate"]

        sample_size = cfg["inference"]["sample_size"]
        self.identity_file = cfg["inference"]["feature_file"]
        self.valid_identity = cfg["inference"]["valid_identity"]
        self.identity_map = cfg["identity_map"]

        with open(self.identity_file, "r") as f:
            lines = f.readlines()
        self.lines = lines[::sample_size]

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
        data = torch.unsqueeze(data, 0)
        return data

    def run(self, onnx_session):
        identity_database = []
        for line in self.lines:
            line = json.loads(line)
            data_path = line["patch_path"]
            try:
                identity_name = data_path.split("/")[-4]
            except:
                identity_name = data_path.split("\\")[-4]
            identity_id = self.identity_map[identity_name]

            if identity_name not in self.valid_identity:
                continue

            data = self.prepare_input(data_path)

            score, feature = onnx_session.run(None, {"input": data.cpu().numpy()})
            feature = np.squeeze(feature)
            identity_database.append(
                {
                    "identity_name": identity_name,
                    "identity_id": identity_id,
                    "feature": feature,
                }
            )

        return identity_database
