import json
import os
import shutil

import numpy as np
import scipy.io as sio

from src.config import CONFIG_MAT
from src.data import sliding_window

PROFILE = "mat"
DATA_FILE = ""
SAVE_FILE = ""


class ConvertData:
    def __init__(self, cfg) -> None:
        self.win_size = cfg["data"]["win_size"]
        self.step_size = cfg["data"]["win_size"]
        self.channel_num = cfg["general"]["input_channel"]
        self.data_file = DATA_FILE
        self.save_file = SAVE_FILE
        self.low_freq = cfg["data"]["low_freq"]
        self.high_freq = cfg["data"]["high_freq"]
        self.sample_rate = cfg["data"]["sample_rate"]

        self.class_id_map = cfg["identity_map"]

    def read_data(self, data_path):
        info = sio.loadmat(data_path, squeeze_me=False)
        data = info["EEG"][0][0][1]
        label = info["EEG"][0][0][5]
        return data, label

    def collect_data(self, data, label):
        results = []
        for i in range(label.shape[0]):
            start_idx = label[i][0][0][0][0]
            end_idx = start_idx + 5000
            results.append(data[:, start_idx:end_idx])
        if len(results) == 0:
            return None
        results = np.concatenate(results, axis=1)
        return results

    def prepare_data(self, data, data_dir, label):
        for cls_name, cls_id in self.class_id_map.items():
            if cls_name in data_dir:
                break
        data = self.collect_data(data, label)
        patchs = sliding_window(data, self.win_size, self.step_size)
        targets = [cls_id for i in range(len(patchs))]

        patch_dir = os.path.join(data_dir, "data")
        target_dir = os.path.join(data_dir, "label")
        if os.path.exists(patch_dir):
            shutil.rmtree(patch_dir)
        os.makedirs(patch_dir)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)

        paths = []
        for i in range(len(patchs)):
            patch_path = os.path.join(patch_dir, str(i) + ".npy")
            target_path = os.path.join(target_dir, str(i) + ".txt")
            np.save(patch_path, patchs[i])
            with open(target_path, "w+") as f:
                f.write(str(targets[i]))
            paths.append({"patch_path": patch_path, "target_path": target_path})
        return paths

    def run(
        self,
    ):
        with open(self.data_file, "r") as f:
            lines = f.readlines()
            paths = []
            for line in lines:
                data_dir = line.strip()  # subject

                data_path = os.path.join(data_dir, "filter_data.mat")
                raw_data_path = os.path.join(data_dir, "raw_data.mat")
                if os.path.exists(data_path) is False:
                    print(f"data_path: {data_path} not exists")
                    continue
                if os.path.exists(raw_data_path) is False:
                    print(f"raw_data_path: {raw_data_path} not exists")
                    continue

                data, _ = self.read_data(data_path)
                _, label = self.read_data(raw_data_path)

                assert data.shape[0] == self.channel_num

                paths_tmp = self.prepare_data(data, data_dir, label)
                paths.extend(paths_tmp)

        with open(self.save_file, "w+") as f:
            for path_ in paths:
                json.dump(path_, f)
                f.write("\n")


if __name__ == "__main__":
    cfg = CONFIG_MAT if PROFILE == "mat" else None
    if cfg is None:
        raise ValueError(f"Unknown profile: {PROFILE}")
    if not DATA_FILE or not SAVE_FILE:
        raise ValueError("Set DATA_FILE and SAVE_FILE before running.")
    convert_mapper = ConvertData(cfg)
    convert_mapper.run()
