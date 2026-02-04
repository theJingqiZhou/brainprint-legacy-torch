import json

import numpy as np


def run(data_file):
    with open(data_file, "r") as f:
        lines = f.readlines()
        paths = []
        for line in lines:
            line = line.strip()
            info = json.loads(line)
            patch_path = info["patch_path"]
            data = np.load(patch_path)
            paths.append(data)

    paths = np.concatenate(paths, axis=1)
    print(f"min = {np.min(paths, axis=1)}")
    print(f"max = {np.max(paths, axis=1)}")


if __name__ == "__main__":
    data_file = "data2/train.jsonl"
    run(data_file)
