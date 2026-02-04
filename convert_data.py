import json
import os
import shutil
from glob import glob

import numpy as np

from src.runtime_config import CONFIG_DEFAULT
from src.utils.preprocess import descale
from src.utils.sliding import sliding_window

PROFILE = "default"
DATA_FILE = ""
SAVE_FILE = ""


class ConvertData:
    def __init__(self, cfg) -> None:
        self.win_size = cfg["data"]["win_size"]
        self.step_size = cfg["data"]["win_size"]
        self.channel_num = cfg["general"]["input_channel"]
        self.data_file = DATA_FILE
        self.save_file = SAVE_FILE
        self.ssvep_cls_num = cfg["data"]["ssvep_cls_num"]
        self.low_freq = cfg["data"]["low_freq"]
        self.high_freq = cfg["data"]["high_freq"]
        self.sample_rate = cfg["data"]["sample_rate"]

        self.class_id_map = cfg["class_id_map"]

    def read_data(self, data_dir, prefix):
        data_dir = os.path.join(data_dir, prefix)
        file_paths = []
        for file_path in glob(data_dir + "*.txt"):
            file_paths.append(file_path)
        if len(file_paths) > 1:
            file_paths = sorted(
                file_paths,
                key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]),
            )

        arrs = []
        for file_path in file_paths:
            arr = np.genfromtxt(file_path, delimiter=" ")
            arrs.append(arr)
        if len(arrs) > 1:
            data = np.stack([arrs], axis=0)
            data = data.squeeze()
            data = descale(data)
        else:
            data = arrs[0]

        return data.squeeze()

    def prepare_data(self, data, marker, data_type, data_dir):
        cls_id = -1
        for cls_id_tmp, cls_names in self.class_id_map.items():
            for cls_name in cls_names:
                if cls_name in data_dir:
                    cls_id = cls_id_tmp
                    break
            if cls_id >= 0:
                break
        if data_type == "by":
            patchs = sliding_window(data[:, 10000:], self.win_size, self.step_size)
            targets = [cls_id for i in range(len(patchs))]
        elif data_type == "ssvep":
            patchs = []
            targets = []
            for ssvep_id in range(1, self.ssvep_cls_num + 1):
                poss = np.where(marker == ssvep_id)[0]
                patch = []
                for pos in poss:
                    patch.append(data[:, pos : pos + 4999])
                patch = np.concatenate(patch, axis=1)
                patch = sliding_window(patch, self.win_size, self.step_size)
                target = [cls_id for i in range(len(patch))]
                patchs.extend(patch)
                targets.extend(target)
        else:
            print(f"data_type={data_type} not been supported")
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
                line = line.strip()  # subject
                for data_type in ["by", "ssvep"]:  # type
                    data_dir = os.path.join(line, data_type)
                    if os.path.exists(data_dir) == False:
                        continue

                    data = self.read_data(data_dir, "Channel")
                    marker = self.read_data(data_dir, "marker")

                    assert data.shape[0] == self.channel_num
                    assert marker.shape[0] > 0

                    paths_tmp = self.prepare_data(data, marker, data_type, data_dir)
                    paths.extend(paths_tmp)

        with open(self.save_file, "w+") as f:
            for path_ in paths:
                json.dump(path_, f)
                f.write("\n")


if __name__ == "__main__":
    cfg = CONFIG_DEFAULT if PROFILE == "default" else None
    if cfg is None:
        raise ValueError(f"Unknown profile: {PROFILE}")
    if not DATA_FILE or not SAVE_FILE:
        raise ValueError("Set DATA_FILE and SAVE_FILE before running.")
    convert_mapper = ConvertData(cfg)
    convert_mapper.run()
